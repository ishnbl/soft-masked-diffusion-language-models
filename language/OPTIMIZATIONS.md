# Soft-Mask Forward-Pass Optimizations

Three targeted memory/compute optimizations applied to the `slerp_sm` training path.
None change model outputs or gradient semantics — they reduce wasted work for operations
whose results were already being discarded.

---

## 1. Pass 1 uses `torch.no_grad()` instead of `.detach()` (`algo.py`)

**Before**
```python
log_x_theta_pass1 = self.forward(xt, sigma=sigma).detach()
```

**After**
```python
with torch.no_grad():
    log_x_theta_pass1 = self.forward(xt, sigma=sigma)
```

**Why**: `.detach()` drops the output tensor from the autograd graph but the full
backward graph is still _built_ during the forward pass and then thrown away.
`torch.no_grad()` skips graph construction entirely, saving activation memory and
graph-building time for a forward whose output is only ever used as data (feedback
logits for the transparency head).

With two full backbone forwards per soft-mask step, Pass 1 is ~50% of the per-step
forward cost. Making it graph-free can free enough activation memory to support a
larger batch or longer sequence length.

---

## 2. Top-k selection in `slerp_sm_feedback` operates on masked positions only (`transparency_head.py`)

**Before**: `torch.topk` ran on the full `(B, T, V)` logit tensor, computing top-k
for every token position (including unmasked ones whose results were immediately
discarded by the final `torch.where`).

**After**: Masked positions are gathered first (`logits[mask_pos]` → `(M, V)`), then
`torch.topk`, `frechet_mean_sphere`, and SLERP all operate on shape `(M, …)` where
`M ≤ B×T`. Results are scattered back into a `(B, T, D)` output via `torch.where`.

**Why this is safe**: The final output at unmasked positions is `E[input_ids]`
regardless — the SLERP value there was always discarded. Computing SLERP only for
positions that actually use it produces identical outputs.

**Savings**: `torch.topk` over `(B, T, V)` is `O(B·T·V·log k)`. Gathering first
reduces this to `O(M·V·log k)`. At a typical masking rate of 30–70% the savings are
substantial; at high masking rates (near `sm_t_max`) the benefit is smaller but the
operation is also cheaper in absolute terms.

---

## 3. Entropy / softmax computed only at masked positions for `slerp_sm` and `mixinputs_with_topk` (`transparency_head.py`)

**Before**: `get_neg_entropy_and_probabilities` computed `softmax((B,T,V))` at the
start of every `TransparencyHead.forward` call, producing both `neg_entropy (B,T)`
and `p_full (B,T,V)`. For the `slerp_sm` and `mixinputs_with_topk` paths, `p_full`
was never used.

**After**: For those two paths, logits are first gathered at masked positions
(`logits_prelim[mask_positions]` → `(M, V)`), entropy is computed there only, and
the result is scattered back into a `(B, T)` zeros tensor. `p_full` is set to `None`
and never allocated.

**Why this is safe**: `p_full` is only consumed in the generic interpolation (`else`)
branch. `neg_entropy` is used to compute `lambda_tensor`, but only its values at
masked positions matter (non-masked positions are zeroed by `calculate_lambda_tensor`).

In Pass 2 the feedback logits (`log_p_x0`) come from Pass 1 which is now wrapped in
`torch.no_grad()`, so `neg_entropy` carries no grad regardless — the optimization
does not affect the gradient path to `scale / steepness / centre`.

**Savings**: Avoids a `softmax` over `(B, T−M, V)` unmasked positions every step.
For a vocabulary of 50k tokens this is the single largest FLOP consumer in the
transparency-head forward; eliminating it for unmasked positions is meaningful.

# SLERP Soft-Masking Pathway — Parallelism & Optimization Report

**Scope:** Only the `slerp_sm` transparency pathway (spherical feedback in embedding
space). The `lerp`/`mixinputs_*` pathways are out of scope and only referenced for
contrast. No code was modified — this is analysis only.

---

## 1. What the SLERP pathway actually does

### Call graph

```
MDLM_SM.nll()                              algo.py:207
  └─ forward(xt, sigma)            → pass 1, .detach()        algo.py:237
  └─ forward(xt, sigma, log_p_x0)  → pass 2, gradients        algo.py:239
        └─ MDLM_SM.forward()                                  algo.py:175
             ├─ embedding_matrix = backbone.vocab_embed.embedding
             ├─ TransparencyHead.forward(xt, log_p_x0, E)     transparency_head.py:197
             │     └─ slerp_sm_feedback(...)                  transparency_head.py:48
             │           └─ frechet_mean_sphere(...)          transparency_head.py:18
             └─ backbone(p_x0_sm, sigma)  # p_x0_sm is (B,T,D) inputs_embeds
                   └─ EmbeddingLayer.forward  → pass-through   (dit.patch, ndim==3 branch)
```

### Per-step math (`slerp_sm_feedback`, transparency_head.py:48–108)

For **every** sequence position `(B, T)`:
1. `topk(logits, k)` → top-k token ids + softmax weights `pi`  (k default = `mixinputs_k = 3`).
2. `vhat = normalize(E[topk_indices])` → (B, T, k, D) unit vectors.
3. `frechet_mean_sphere(vhat, pi, n_iter=3)` → Karcher mean `mu*` on the unit sphere
   (an iterative loop with `acos`/`sin`/`normalize` each iteration).
4. SLERP between normalized mask embedding `m_hat` and `mu*` using angle `omega = acos(cos)`.
5. Rescale by the mask-token norm, then `torch.where(mask_positions, slerp, E[input_ids])`.

All compute is forced to `float32` (`compute_dtype = torch.float32`) for the
`acos`/`sin` numerics; the outer `MDLM_SM.forward` also wraps the backbone in a
`float32` autocast region (algo.py:189).

---

## 2. Is it parallelizable across GPUs?

**Short answer: Yes — the SLERP pathway is fully data-parallel safe. There is nothing
in it that blocks DDP (or FSDP). The current launcher just doesn't use multi-GPU.**

### 2.1 Why it is safe

Every operation in `slerp_sm_feedback` and `frechet_mean_sphere` is **per-position /
per-sample elementwise** (topk, normalize, acos, sin, gather `E[idx]`, `torch.where`).
There is:
- **No reduction across the batch dimension** — sample `i` never reads sample `j`.
- **No cross-rank communication** inside the head.
- **No data-dependent control flow that diverges between ranks** in the tensor math
  (the gating decisions in §2.3 are the one caveat).

The only **shared / replicated** parameters touched are:
- the backbone embedding matrix `E = backbone.vocab_embed.embedding` (read in the head,
  receives gradient in pass 2), and
- the 4 scalar transparency-head params (`raw_scale`, `raw_centre_neg`, `raw_steep`,
  `raw_temperature`).

Under DDP these are replicated per rank and their gradients are all-reduced by the
standard DDP hook — exactly like any other weight. So **SLERP composes cleanly with
data parallelism**; throughput should scale near-linearly with GPU count, modulo the
two overheads below.

### 2.2 Current configuration does NOT run multi-GPU

The Modal launcher hardcodes single-GPU:

```python
"trainer.devices=1",                    # run_modal.py:227
"strategy.find_unused_parameters=True", # run_modal.py:246
```

`configs/strategy/ddp.yaml` (`find_unused_parameters: false`) and
`configs/strategy/fsdp.yaml` exist, so the framework *supports* multi-GPU; it is simply
not wired into the current SLERP runs. To go multi-GPU you would raise `trainer.devices`
and keep `strategy=ddp` — no change to the SLERP math is required.

### 2.3 The two real DDP overheads (both fixable, neither is a blocker)

**(a) `find_unused_parameters=True` is currently required, and it is a tax on every step.**
The soft-mask path is gated by a Bernoulli draw (`sm_prob`) **and** a time-band check
(`sm_t_min..sm_t_max`):

```python
sm_gate = (not train_mode) or (torch.rand(1).item() < self.config.optim.sm_prob)
in_band = (sm_t_min <= t.mean().item() <= sm_t_max)
use_soft_mask = sm_gate and in_band        # algo.py:227–233
```

On steps where `use_soft_mask` is **False**, the transparency-head params and the SLERP
sub-graph receive **no gradient**. Vanilla DDP raises "expected to mark all parameters
ready" in that situation, which is why the launcher sets `find_unused_parameters=True`.
That flag forces an extra autograd-graph traversal every backward to discover unused
params — a measurable per-step cost (commonly 5–15%).

> **Concern for multi-GPU:** the gate uses `torch.rand(1)` and `t.mean()`. If the RNG /
> timestep sampling is not identical across ranks, **different ranks can take different
> branches on the same step** (some soft-mask, some not). DDP still works with
> `find_unused_parameters=True`, but the set of "ready" params then differs per rank,
> which is the classic recipe for **desync / hangs** in stricter setups. Verify that
> `t` sampling is rank-synchronized (`trainer_base.py:426–427` already chunks `t` by
> node/rank, implying per-rank-distinct `t`), or make the gate a **global batch-level
> decision broadcast from rank 0** so all ranks branch identically. That also lets you
> turn `find_unused_parameters` back **off**.

**(b) Per-step `sync_dist=True` logging all-reduces.** `slerp_angle_mean`,
`lambda_mean/std`, and the 4 head scalars are logged every step with `sync_dist=True`
(algo.py:160–171). Each is a tiny all-reduce; harmless at small scale but pure overhead.
Log every N steps or with `sync_dist=False` for the scalar params (they are identical
across ranks anyway after the grad all-reduce).

---

## 3. Optimization opportunities (single-GPU and multi-GPU)

Ordered roughly by expected payoff.

### 3.1 ★ Compute SLERP only on masked positions (biggest win)

`slerp_sm_feedback` runs the full pipeline — topk, `E[topk_indices]`, the entire
`frechet_mean_sphere` iteration, and the SLERP — for **all** `B×T` positions, then throws
away non-masked positions at the very end:

```python
mask_positions = (input_ids == mask_token_id).unsqueeze(-1)
out = torch.where(mask_positions, slerp, E[input_ids])   # transparency_head.py:101
```

Notably, the sibling `mixinputs_with_topk` path **already** gathers only masked positions
first (`masked_logits = logits_prelim[mask_positions]`, transparency_head.py:230) and
scatters back — SLERP does not. At a typical diffusion mask ratio the head is doing the
expensive `float32` `acos`/`sin` Karcher loop on a large fraction of positions whose
result is discarded. Gathering masked positions up front (flatten to `(N_masked, ...)`,
run Frechet + SLERP, scatter into `E[input_ids]`) would cut the head's FLOPs and the
`(B,T,k,D)` intermediates by roughly `1 / mask_fraction`. This is the single highest-ROI
change and does not affect numerics.

### 3.2 ★ Pass 1 should use `torch.no_grad()`, not just `.detach()`

```python
log_x_theta_pass1 = self.forward(xt, sigma=sigma).detach()   # algo.py:237
```

`.detach()` detaches the *output*, but the forward still builds the full autograd graph
internally and only drops it afterward — wasted activation memory and graph-construction
time for a pass whose gradients are never used. Wrapping pass 1 in `with torch.no_grad():`
avoids allocating the backward graph entirely. With two full backbone forwards per
soft-mask step (pass 1 feedback + pass 2 main), pass 1 is ~half the forward cost; making
it graph-free frees memory that could instead buy a larger batch.

### 3.3 Reconsider the blanket `float32` autocast region

`MDLM_SM.forward` wraps everything — including the **backbone** — in
`torch.cuda.amp.autocast(dtype=torch.float32)` (algo.py:189). The `float32` is genuinely
needed for the `acos`/`sin` SLERP numerics, but it also forces the surrounding glue to
fp32. The backbone re-enters a bf16 autocast internally (`dit.py` blocks), so the net
effect is mostly on the embedding gather / SLERP. Tightening the `float32` region to wrap
*only* `slerp_sm_feedback` (and keeping the backbone in bf16) would reduce fp32 traffic.
Validate that `omega = acos(cos)` stays stable — the code already clamps `cos` to
`[-1+eps, 1-eps]` and guards `sin_omega`, so bf16 *outside* the angle computation is plausible.

### 3.4 `torch.compile` the head

`slerp_sm_feedback` + `frechet_mean_sphere` are elementwise-fusion-friendly
(normalize, mul, sum, acos, sin, where) with a small static `n_iter=3` loop. They are
currently eager. `torch.compile` (or a hand-fused kernel) on the head would fuse the
chain and remove per-op launch overhead — especially valuable once §3.1 shrinks the
working set. Note: the data-dependent gate in §2.3 will cause graph breaks/recompiles if
you compile the whole `nll`; compile the head function in isolation.

### 3.5 Frechet-mean loop: precompute, and consider fewer iterations

In `frechet_mean_sphere` (transparency_head.py:18–45) each of the `n_iter=3` iterations
recomputes `cos`, `omega=acos`, `sin_omega`, the log-map, and a re-`normalize`. Two cheap
wins:
- The Karcher mean is initialized at the **top-1 token** and the weights `pi` are
  softmax-peaked; in practice 1–2 iterations often converge for k=3. Treat `slerp_n_iter`
  as a tunable and check whether `n_iter=1` or `2` matches `3` on val loss — each saved
  iteration removes a full `acos`/`sin` sweep over `(B,T,k,D)`.
- `acos` + `sin(acos(x))` can be replaced by `sin = sqrt(1 - cos²)` (with clamping),
  avoiding the more expensive `acos`/`sin` transcendental pair on the hot path. (Numerical
  parity must be checked, but it's algebraically exact for the `sin` term.)

### 3.6 Avoid redundant normalization / gathers

- `mhat` is `F.normalize(mask_emb)` broadcast to `(B,T,D)` every call — it depends only on
  `E[mask_token_id]`, so it can be computed once per forward (it already is) but note it is
  re-`expand`ed; fine. The `mask_norm` rescale is a scalar — cheap.
- `E.to(compute_dtype)` (transparency_head.py:66) up-casts the **entire** vocab×D embedding
  table to fp32 on every call even though only `topk_indices` and `mask_token_id` rows are
  read. Gather first (`E[topk_indices]`), then cast the small gathered tensor — avoids
  materializing a fp32 copy of the full embedding matrix each step.

### 3.7 Multi-GPU specific

- Turn `find_unused_parameters` **off** by making the soft-mask gate a global, rank-identical
  decision (see §2.3a). This removes the per-step graph-traversal tax for *all* ranks.
- The SLERP head params are tiny (4 scalars + the shared embedding). DDP gradient bucketing
  is dominated by the backbone; the head adds negligible communication. No gradient-compression
  or bucketing tuning needed for the head itself.
- FSDP (`configs/strategy/fsdp.yaml`) is viable for the *backbone*, but the config comment
  flags FSDP-incompatibility with grad clipping; the SLERP head does not change that picture.
  For models at this scale, **DDP is the right default**; reach for FSDP only if the backbone
  outgrows a single GPU's memory.

---

## 4. Summary

| Question | Finding |
|---|---|
| **Is the SLERP pathway parallelizable across GPUs?** | **Yes.** Fully data-parallel; no cross-sample/cross-rank dependency. Composes with DDP out of the box. |
| **Does the current setup use multiple GPUs?** | **No** — `trainer.devices=1` is hardcoded in the launcher. |
| **Main blocker to clean multi-GPU?** | Not a blocker, but `find_unused_parameters=True` (forced by the stochastic soft-mask gate) taxes every step and risks per-rank branch divergence. Make the gate a global decision to fix both. |
| **Top single-GPU wins** | (1) Run SLERP only on masked positions (§3.1). (2) `no_grad` for feedback pass 1 (§3.2). (3) Tighten the fp32 autocast region + `torch.compile` the head (§3.3–3.4). |
| **Tunable knobs** | `slerp_n_iter` (try 1–2), `mixinputs_k`, fp32-vs-bf16 angle math. |

**Key correctness note carried over:** the SLERP output is `(B,T,D)` `inputs_embeds`, fed to
the backbone's patched `EmbeddingLayer.forward` pass-through branch (`x.ndim==3 and
x.shape[-1]==embed_dim`, from `dit.patch`). Any optimization that changes shapes/dtypes of
the head output must preserve that branch's assumptions.

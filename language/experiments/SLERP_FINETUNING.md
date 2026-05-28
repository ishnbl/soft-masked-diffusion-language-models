# SLERP vs top-k LERP — Complete Fine-tuning Procedure

This document covers the end-to-end recipe for fine-tuning `MDLM_SM` to compare
**SLERP** (spherical linear interpolation in embedding space) against **top-k LERP**
(linear mixing in vocabulary-probability space) as the soft-masking feedback
strategy. It supersedes the earlier from-scratch ablations (scripts 06/07) and
the first finetune attempt (script 09).

---

## 1. Background

Soft-masked diffusion (`MDLM_SM`) runs two forward passes per training step on a
fraction `sm_prob` of steps:

```
xt  = q_xt(x0, αt)                         # mask some tokens
Pass 1:  log_p_x0 = forward(xt).detach()   # predictions, gradients blocked
         tran_head(xt, log_p_x0) → soft_input
Pass 2:  log_x_theta = forward(xt, log_p_x0)  # gradient-carrying pass
loss  = MDLM ELBO on log_x_theta
```

The **transparency head** (`transparency_head.py`) decides _how much_ to mix the
Pass-1 predictions back into the masked positions via a scalar weight λ (lambda)
per token, and the `transparency_alg` config key decides _how_:

| `transparency_alg` | Where it operates | Geometry |
|--------------------|-------------------|----------|
| `mixinputs_with_topk` | vocabulary-probability space | **LERP** — convex blend of one-hot current token and top-k predicted distribution |
| `slerp_sm` | unit-embedding space | **SLERP** — great-circle interpolation between the mask-token embedding and the Fréchet (Karcher) mean of the top-k token embeddings on S^{d-1} |

The two methods are geometrically distinct only when λ is non-trivial. The entire
engineering effort below is about ensuring λ actually activates to a meaningful
level so the comparison is real.

---

## 2. The lambda activation problem (why from-scratch training doesn't work)

Training from scratch (script 06) produced λ ≈ 1e-14 for both runs, making the
comparison meaningless. Three compounding factors caused this:

**Factor 1 — vanishing gradient on `raw_scale`.**
The learnable scale is stored as `raw_scale` and mapped through `σ(raw_scale)`.
The config default `init_scale=0.0` sets `raw_scale = logit(1e-6) = -13.82`, giving
`scale = σ(-13.82) ≈ 1e-6`. The gradient `dσ/d(raw_scale) = scale·(1-scale) ≈ 1e-6`
means `raw_scale` essentially never moves.

**Factor 2 — activation threshold too strict.**
The mixing weight is:
```
λ = scale · σ( steepness · (neg_entropy − centre) )
```
The default `init_centre = -0.75` requires `neg_entropy > -0.75`, i.e. the backbone
must put >85% probability mass on one token before λ switches on. A random backbone
gives `neg_entropy ≈ -3.3` (text8, V=27), so the sigmoid factor ≈ 4e-8 and
`λ = 1e-6 × 4e-8 ≈ 4e-14`.

**Factor 3 — cold-start noise loop.**
Because Pass-1 logits from a random backbone are near-uniform, the soft-masked
input fed to Pass 2 is pure noise. The head's gradient pushes `scale` further down
(correct action given the noise), locking in the collapsed state.

**Script 07** (from-scratch, fixed init) addressed factors 1 and 2 with
`init_scale=0.5, init_centre=-2.5`. This was a necessary precondition, but the
cold-start noise loop remained.

---

## 3. Why fine-tuning from a pretrained backbone

Starting from the **released binary MDLM (OpenWebText)** breaks the cold-start loop:
Pass-1 predictions are immediately confident (`neg_entropy → 0`), so with the fixed
init values λ activates at ≈ 0.3–0.5 from step 0 and the gradient pushes `scale`
upward. At λ in the 0.2–0.5 range the arc path (SLERP) and straight-line blend
(LERP) produce visibly different embeddings and the val/bpd curves can separate.

**Checkpoint source:** released binary MDLM (OpenWebText) from the Google Drive
folder linked in `language/README.md`. Same checkpoint as script 01 — a GPT-2-BPE
`small` DiT, so this procedure runs on `model=small` + `data=openwebtext-split`.

**How `finetune_path` loads (`main.py:162–177`):**
```python
model = MDLM_SM(config)            # fresh model; tran_head at __init__() values
old   = torch.load(BASE_CKPT)      # base MDLM state_dict — no tran_head.* keys
model.load_state_dict(old, strict=False)
  → backbone weights loaded from checkpoint  ✓
  → tran_head.* absent in ckpt  → retains init_scale / init_centre below  ✓
```
On first run, log `Weights loaded with N missing keys` — the missing keys should be
exactly `tran_head.*`.

---

## 4. Transparency head internals

Four scalar parameters, each stored in unconstrained `raw_*` form:

| Property | Transform | Init source | Value |
|----------|-----------|-------------|-------|
| `scale` | `σ(raw_scale)` ∈ (0,1) | `logit(init_scale)` | `init_scale=0.5` → `scale=0.5` |
| `centre` | `−softplus(raw_centre_neg) − ε` | `softplus⁻¹(−init_centre)` | `init_centre=−2.5` |
| `steepness` | `softplus(raw_steep) + ε` > 0 | `softplus⁻¹(init_steep)` | default `6.66` |
| `temperature` | `softplus(raw_temperature) + ε` > 0 | `softplus⁻¹(init_temperature)` | `1.0` |

Lambda at a confident backbone (`neg_entropy ≈ 0`):
```
λ_max ≈ 0.5 · σ(6.66 · 2.5) ≈ 0.5 · σ(16.65) ≈ 0.5
```

**`slerp_sm` path** (`transparency_head.py:18–100`):
1. Take top-k token indices from Pass-1 logits; L2-normalise their embeddings onto S^{d-1}.
2. Compute weighted Fréchet (Karcher) mean μ\* via `slerp_n_iter` Riemannian iterations.
3. Compute `SLERP(m̂, μ*, λ) = sin((1−λ)ω)/sinω · m̂ + sin(λω)/sinω · μ*`  
   where `ω` is the angle between the mask embedding and μ\*.
4. Return `(B, T, D)` pre-embedded tensor; the DiT embedding layer passes it through
   unchanged (see `patches/dit.patch` pass-through branch).

**`mixinputs_with_topk` path**: returns sparse `(indices, probs)` tuple consumed by
the DiT's sparse-embedding path (also in `patches/dit.patch`).

---

## 5. The complete fine-tuning recipe

### Step 1 — Download the base checkpoint

Get the released binary MDLM (OpenWebText) from the Google Drive folder in
`language/README.md`. Set its path as `BASE_CKPT`.

### Step 2 — Run the primary A/B (script 10_v2)

```bash
BASE_CKPT=/path/to/mdlm.ckpt SEED=1 MAX_STEPS=5000 \
  bash scripts/10_slerp_vs_topk_finetune_v2.sh
```

This runs two sequential jobs (`topk` then `slerp`) with every variable held fixed
except `transparency_alg`. Same seed → same init and data order → directly
comparable val/bpd curves.

Key hyperparameters (all deltas from the config defaults are marked ★):

| Knob | Value | Note |
|------|-------|------|
| `algo` | `mdlm_sm` | soft-masked MDLM |
| `model` | `small` | DiT hidden=768, 12 blocks, 12 heads, ctx=1024 |
| `data` | `openwebtext-split` | GPT-2 BPE, vocab ≈ 50k |
| `training.finetune_path` | `$BASE_CKPT` | loads backbone; tran_head keeps fresh init |
| `checkpointing.resume_from_ckpt` | `false` | prevents stale `best.ckpt` overriding the load |
| `optim.lr` ★ | `3e-5` | 10× lower than config default; backbone already trained |
| `optim.tran_head_lr` | `0.01` | high — head starts fresh and must learn fast |
| `optim.sm_prob` | `0.8` | 80% of steps run the two-pass soft-mask forward |
| `algo.tran_head.init_scale` ★ | `0.5` | λ_max ≈ 0.5 at confident logits |
| `algo.tran_head.init_centre` ★ | `-2.5` | activates at ~30% top-1 confidence |
| `algo.tran_head.mixinputs_k` | `3` | top-k tokens to mix |
| `algo.tran_head.slerp_n_iter` | `3` | Karcher iterations (slerp run only) |
| `lr_scheduler.num_warmup_steps` | `200` | short — backbone already trained |
| `trainer.max_steps` | `5000` | |
| `trainer.val_check_interval` ★ | `200` | catch early divergence |
| `trainer.log_every_n_steps` | `50` | |
| `eval.compute_generative_perplexity` | `False` | irrelevant to A/B, skips expensive gen-PPL |

The **only difference between the two runs** is:
```
topk run:  algo.tran_head.transparency_alg=mixinputs_with_topk
slerp run: algo.tran_head.transparency_alg=slerp_sm  algo.tran_head.slerp_n_iter=3
```

### Step 3 — Monitor in W&B

Both runs share one `wandb.group`; overlay them and watch:

| Panel | Metric | What to look for |
|-------|--------|-----------------|
| `transparency/scale` | learned scale | should trend **up** from 0.5 |
| `transparency/lambda_mean` | mean λ over masked positions | should reach > 0.1 early; target 0.2–0.5 |
| `transparency/lambda_std` | std of λ over masked positions | should be non-zero; a flat std means the head is becoming constant |
| `transparency/slerp_angle_mean` | mean SLERP angle ω (slerp run only) | typically ~1.5 rad; confirms non-degenerate geometry |
| `val/bpd` | validation bits-per-dim | headline metric; if SLERP helps, curves separate once λ is non-trivial |

### Step 4 — Decision: did lambda activate?

**Lambda is healthy** if `lambda_mean > 0.1` and `lambda_std > 0` within the first
500 steps. Let the run finish; compare `val/bpd` at 5k steps.

**Lambda is collapsing** if `lambda_mean → 0` or `lambda_std` goes flat (head
suppressing itself). Kill the run and proceed to Step 5.

---

## 6. Recovery — freeze-first variant (script 10b)

If lambda collapses, the cause is noisy gradient signal from the moving backbone
reaching the head at the same time. The fix: hold the backbone stationary for the
first 1000 steps so the head settles, then unfreeze and finetune jointly.

```bash
BASE_CKPT=/path/to/mdlm.ckpt SEED=1 MAX_STEPS=5000 FREEZE_UNTIL=1000 \
  bash scripts/10b_slerp_vs_topk_finetune_freeze.sh
```

**How it works (`callbacks.py: FreezeBackboneCallback`):**

- On `train_start`: sets `requires_grad=False` on every param not under `tran_head.*`.
  AdamW sees zero gradients for those params and does not update them.
- Autograd still propagates JVPs _through_ the frozen backbone, so `tran_head.*`
  receives full gradients against a stationary signal.
- At `global_step == FREEZE_UNTIL`: `requires_grad` is restored on the backbone and
  joint fine-tuning begins. No optimizer reconstruction needed — the param groups
  already exist with zero effective updates during the freeze.
- Uses `find_unused_parameters=True` (already set) for DDP compatibility with
  frozen leaves.

Everything else is identical to the primary v2 recipe: same seeds, same
`init_scale=0.5`, same `optim.lr=3e-5`.

---

## 7. Optimizer structure

Two AdamW parameter groups (`algo.py:107–127`):

| Group | Parameters | LR |
|-------|-----------|-----|
| main | everything not under `tran_head.*` | `optim.lr` = `3e-5` |
| head | `tran_head.*` (4 scalars) | `optim.tran_head_lr` = `0.01` |

LR schedule: constant with linear warmup (`lr_scheduler=constant_warmup`,
`num_warmup_steps=200`). Backbone LR ramps 1e-6 → 3e-5 over 200 steps; head LR
ramps 1e-6 → 0.01 over 200 steps. After warmup both are constant.

---

## 8. Architecture: how SLERP inputs reach the backbone

The DiT embedding layer (`patches/dit.patch`) dispatches on input type:

```
x is tuple (indices, probs) → sparse weighted-sum of k+1 embedding vectors   [top-k path]
x.ndim == 2                 → standard token-id lookup                        [plain path]
x.ndim == 3, last dim == D  → pass through unchanged                          [SLERP path]
x.ndim == 3, last dim == V  → dense einsum over full vocabulary               [legacy path]
```

`slerp_sm` returns `(B, T, D)` pre-embedded tensors; the pass-through branch
ensures they flow into the transformer body without re-embedding.
`mixinputs_with_topk` returns a `(indices, probs)` tuple consumed by the sparse
branch.

---

## 9. Compute

| Resource | Requirement |
|----------|-------------|
| GPU | ≥ 1 × A100/H100-class (40–80 GB); one per run |
| Precision | bf16 |
| Context length | 1024 tokens |
| Memory | Close to a single training step (Pass 1 is detached; only Pass-2 activations held) |
| Wall-clock | ~multi-hour per run at 5k steps; run topk and slerp in parallel on two GPUs to halve it (see commented parallel block in each script) |
| Disk / network | One-time download of the base checkpoint + OpenWebText cache (several GB on first run, path set via `DATA_CACHE_DIR`) |

Both scripts include a preflight that fails fast if no GPU is visible:
```bash
python -c "import torch; assert torch.cuda.device_count() > 0"
```
This catches the DDP assertion (`global_batch_size == batch_size * devices * accum`)
before wasting time on dataset loading.

---

## 10. Script and file index

| File | Purpose |
|------|---------|
| `scripts/06_slerp_vs_topk_ablation.sh` | Original from-scratch A/B (broken — lambda collapses) |
| `scripts/07_slerp_vs_topk_fixed.sh` | From-scratch with fixed init (lambda activates, but cold-start noise remains) |
| `scripts/09_slerp_vs_topk_finetune.sh` | First finetune attempt from pretrained backbone (init_scale=0.3) |
| `scripts/10_slerp_vs_topk_finetune_v2.sh` | **Primary recipe** — use this first |
| `scripts/10b_slerp_vs_topk_finetune_freeze.sh` | **Recovery recipe** — use if lambda collapses in v2 |
| `transparency_head.py` | `TransparencyHead`, `slerp_sm_feedback`, `frechet_mean_sphere` |
| `algo.py` | `MDLM_SM`: two-pass forward, two-group optimizer, transparency logging |
| `callbacks.py` | `FreezeBackboneCallback` |
| `configs/callbacks/freeze_backbone.yaml` | Hydra config for the freeze callback |
| `patches/dit.patch` | DiT embedding-layer dispatch for sparse/pre-embedded/dense inputs |

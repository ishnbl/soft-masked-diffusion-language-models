# Experiment 09 — SLERP vs top-k LERP soft-masking (finetune from released OWT MDLM)

Companion doc for [`09_slerp_vs_topk_finetune.sh`](./09_slerp_vs_topk_finetune.sh).

## 1. What this experiment tests

Soft-masked diffusion (`MDLM_SM`) feeds the model's own predictions back into
the masked positions of its input. The **transparency head** decides *how much*
of the prediction to mix in (a per-token weight `λ`) and the `transparency_alg`
decides *how* to mix it:

| `transparency_alg` | Where it mixes | Geometry |
|--------------------|----------------|----------|
| `mixinputs_with_topk` (baseline) | vocabulary-probability space | **LERP** — convex blend of the one-hot current token and the top-k predicted distribution |
| `slerp_sm` | unit-embedding space | **SLERP** — great-circle interpolation between the mask-token embedding and the Fréchet (Karcher) mean of the top-k token embeddings on `Sᵈ⁻¹` |

The two methods are mathematically distinct **only when `λ` is non-trivial**.
At small `λ` the great-circle path and the straight-line path differ by `O(λ²)`,
so both collapse to vanilla MDLM. This experiment is an A/B that holds
everything else fixed (same seed → same init + data order) and asks: *does
SLERP's spherical interpolation beat LERP's linear blend on val bits-per-dim?*

## 2. Why finetune instead of train from scratch

A from-scratch run hits a **cold-start trap** (documented in experiments
06/07):

- A random backbone produces near-uniform Pass-1 logits (`neg_entropy ≈ −3.3`
  on text8), so the fed-back predictions are pure noise.
- The transparency head correctly learns to suppress itself: `scale` drops from
  0.5 → ~0.25, `λ` settles at 0.01–0.04.
- At `λ ≈ 0.02` the SLERP/LERP difference is `O(λ²) ≈ 0` → the two runs produce
  **identical** bpd/ppl/nll. The comparison never actually starts.

Starting from the **released binary MDLM (OpenWebText)** backbone removes the
cold start: Pass-1 predictions are immediately confident (`neg_entropy` near 0),
so `λ` activates at ≈0.3 from step 0 and the gradient pushes `scale` *up*. At
`λ` in the 0.2–0.5 range the two interpolation geometries genuinely diverge and
the comparison is meaningful.

> Checkpoint source: the binary MDLM Google-Drive folder linked in
> [`language/README.md`](../README.md). It is the same checkpoint the repo's own
> continuation recipe (`01_sm_pretraining_cont_owt.sh`) consumes — a GPT-2-BPE
> `small` DiT, hence this experiment runs on `model=small` + `data=openwebtext-split`,
> **not** the `tiny`/`text8` setup of experiments 06/07.

## 3. End-to-end flow per training step

For each step the loss is the standard MDLM ELBO, but the forward is a
**two-pass** procedure on a fraction `sm_prob` of steps
(`algo.py:219–229`):

```
xt  = q_xt(x0, αt)                      # mask some tokens
Pass 1:  log_p_x0 = forward(xt).detach()        # predictions, gradients blocked
         tran_head(xt, log_p_x0) -> soft input  # mix predictions into masks
Pass 2:  log_x_theta = forward(xt, log_p_x0)     # the gradient-carrying pass
loss  = nll_per_token(log_x_theta, ...)
```

- Pass 1 is **detached** (`algo.py:223`), so the transparency head only receives
  gradients through Pass 2.
- On `(1 − sm_prob)` of steps the model does a single plain MDLM forward (no soft
  masking), which keeps the backbone honest.
- `slerp_sm` returns pre-embedded `(B,T,D)` inputs and the DiT embedding layer
  passes them through unchanged (the `patches/dit.patch` pass-through branch);
  `mixinputs_with_topk` returns sparse `(indices, probs)` consumed by the
  backbone's sparse-embedding path.

How the released backbone loads into `MDLM_SM` (`main.py:162–177`,
`strict=False`):

```
model = MDLM_SM(config)            # fresh model; tran_head at __init__() values
old   = torch.load(BASE_CKPT)      # base MDLM state_dict — has NO tran_head.* keys
model.load_state_dict(old, strict=False)
  → backbone weights load from checkpoint  ✓
  → tran_head.* are "missing keys" → keep the init values below  ✓
```

Watch the first-run log line `Weights loaded with N missing keys` — the missing
keys should be exactly the `tran_head.*` parameters.

## 4. The learnable transparency head

Four scalar parameters, each stored in an unconstrained `raw_*` form and mapped
through a monotone transform so it stays in range
(`transparency_head.py:114–145`):

| Property | Transform | `raw_*` init source | Init value here |
|----------|-----------|---------------------|-----------------|
| `scale` | `σ(raw_scale)` ∈ (0,1) | `logit(init_scale)` | `init_scale=0.3` → `scale=0.3` |
| `centre` | `−softplus(raw_centre_neg) − ε` | `softplus⁻¹(−init_centre)` | `init_centre=−2.5` |
| `steepness` | `softplus(raw_steep) + ε` > 0 | `softplus⁻¹(init_steep)` | `init_steep=6.66` |
| `temperature` | `softplus(raw_temperature) + ε` > 0 | `softplus⁻¹(init_temperature)` | `1.0` (unused unless `mixinputs_with_temp`) |

The mixing weight is computed per token from the Pass-1 confidence
(`transparency_head.py:157–168`):

```
neg_entropy = Σ p·log p          # ∈ [−log V, 0]; → 0 = confident, very negative = uniform
λ           = scale · σ( steepness · (neg_entropy − centre) )
λ           = 0   on non-mask positions
```

With `init_scale=0.3`, `init_centre=−2.5`, `init_steep=6.66`: a confident
backbone (`neg_entropy → 0`) gives `λ ≈ 0.3·σ(6.66·2.5) ≈ 0.3`. This is the
**critical fix** — the *config defaults* (`init_scale=0.0` → `scale ≈ 1e-6`,
`init_centre=−0.75`) drive `λ` to ~`1e-14` and the experiment degenerates.

Why `raw_scale` and not `scale` directly: `dσ/d(raw_scale) = scale·(1−scale)`.
At `init_scale=0` this is `~1e-6`, so the gradient on `raw_scale` vanishes and
`scale` can never climb. `init_scale=0.3` gives `σ'(logit(0.3)) ≈ 0.21` — a
healthy gradient.

`slerp_sm` internals (`transparency_head.py:18–100`): top-k token embeddings are
L2-normalised onto the sphere, their Fréchet mean `μ*` is found by `slerp_n_iter`
Riemannian (Karcher) iterations, then the soft input is
`SLERP(m̂, μ*, λ) = sin((1−λ)ω)/sinω · m̂ + sin(λω)/sinω · μ*`,
where `ω` is the angle between the mask embedding and `μ*`.

## 5. Hyperparameters

Defaults set in the script; override via env vars (`SEED`, `MAX_STEPS`,
`MODEL`, `DATA`, `BATCH_SIZE`, `BASE_CKPT`).

| Group | Knob | Value | Notes |
|-------|------|-------|-------|
| **Data / model** | `algo` | `mdlm_sm` | soft-masked MDLM |
| | `model` | `small` | DiT: hidden 768, 12 blocks, 12 heads, ctx 1024 |
| | `data` | `openwebtext-split` | GPT-2 BPE, vocab ≈ 50k |
| **Backbone load** | `training.finetune_path` | `$BASE_CKPT` | released binary MDLM |
| | `checkpointing.resume_from_ckpt` | `false` | stop a stale `best.ckpt` overriding the load |
| **Transparency head** | `algo.tran_head.transparency_alg` | `mixinputs_with_topk` / `slerp_sm` | **the only A/B difference** |
| | `algo.tran_head.init_scale` | `0.3` | activation fix |
| | `algo.tran_head.init_centre` | `−2.5` | activation fix |
| | `algo.tran_head.mixinputs_k` | `3` | top-k tokens mixed |
| | `algo.tran_head.slerp_n_iter` | `3` | Karcher iterations (slerp only) |
| **Optimisation** | optimizer | AdamW | β=(0.9,0.999), eps 1e-8, wd 0 |
| | `optim.lr` (backbone) | `3e-4` | config default; mirrors `01_*`. Lower (e.g. `3e-5`) if you see forgetting |
| | `optim.tran_head_lr` | `0.01` | high — head starts fresh, must learn fast |
| | `optim.sm_prob` | `0.8` | fraction of steps that do the two-pass soft mask |
| | `lr_scheduler.num_warmup_steps` | `200` | short — backbone is already trained |
| **Schedule / batch** | `trainer.max_steps` | `5000` | per run |
| | `trainer.val_check_interval` | `500` | catch early divergence |
| | `loader.global_batch_size` | `512` | config default (effective batch) |
| | `loader.batch_size` | `32` | per-GPU micro-batch; `accumulate = 512/(32·#GPU)` |
| | `eval.compute_generative_perplexity` | `False` | skip expensive gen-PPL (irrelevant to this A/B) |

The optimiser uses **two parameter groups** (`algo.py:107–127`): everything
under `tran_head.*` gets `tran_head_lr`, the rest gets `lr`.

## 6. What success looks like (W&B)

Both runs share one `wandb.group`; overlay them and watch the `transparency/*`
panel (logged each step in `algo.py:156–165`):

- `transparency/scale` — should trend **up** from 0.3 (vs *down* in the scratch runs).
- `transparency/lambda_mean` — should reach **> 0.1** early and climb toward 0.2–0.5.
- `transparency/slerp_angle_mean` — the mean SLERP angle `ω` (slerp run only),
  typically ~1.5 rad; confirms the geometry is non-degenerate.
- `val/bpd` (and ppl/nll) — the headline. **If SLERP helps, its curve separates
  from top-k once `λ` is non-trivial.** Identical curves now mean "no SLERP
  advantage on this task" — a real result — rather than "the head was inert".

## 7. Compute

Dominant cost is the **two forward passes** on ~80% of steps (`sm_prob=0.8`),
on a GPT-2-`small`-scale DiT. SLERP adds 3 Karcher iterations over `k=3`
embeddings — negligible next to a transformer forward.

- **GPU**: one 40–80 GB GPU (A100/H100 class) per run. `bf16` precision, ctx 1024.
- **Memory**: two passes keep Pass-2 activations only (Pass 1 is detached), so
  the footprint is close to a single training step plus the small head.
- **Wall-clock**: ≈ 5000 steps × 2 runs. Sequentially this is a multi-hour to
  overnight job depending on GPU; run the two variants in parallel on two GPUs
  (see the commented block at the foot of the script) to roughly halve it.
- **Hard requirement**: ≥ 1 visible GPU. The dataloader asserts
  `global_batch_size == batch_size · num_nodes · device_count · accum`, which
  reads the **real** `torch.cuda.device_count()`; the script's preflight fails
  fast with a clear message if no GPU is attached.
- **Disk / network**: one-time download of the released MDLM checkpoint plus the
  OpenWebText cache (`data.cache_dir`, several GB on first run).

## 8. How to run

```bash
# 1. Download the binary MDLM (OWT) from the Google-Drive folder in language/README.md
# 2. Point BASE_CKPT at it and launch the A/B:
BASE_CKPT=/path/to/mdlm.ckpt SEED=1 MAX_STEPS=5000 \
  bash scripts/09_slerp_vs_topk_finetune.sh
```

Outputs land in `outputs/slerp_vs_topk_finetune_small_openwebtext-split_seed1/{topk,slerp}`.

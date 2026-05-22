# SLERP vs Top-k Soft-Masking: Convergence Experiment

Tests whether spherical interpolation (`slerp_sm`) converges faster than the
existing convex top-k mixing (`mixinputs_with_topk`), using a small MDLM
(~14 M params, text8, from scratch).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU | ≥ 8 GB VRAM. Tiny model at batch 64 uses < 4 GB. |
| CUDA driver | ≥ 525 (for CUDA 12.4 toolkit). Check: `nvidia-smi` |
| Conda | Miniconda or Anaconda. [Install](https://docs.anaconda.com/miniconda/) |
| W&B account | Free at [wandb.ai](https://wandb.ai) — needed for live metric plots. |
| Internet access | To download DUO source files, pip packages, and text8. |

---

## Step 1 — Navigate to the right directory

All commands below are run from the `language/` subdirectory.

```bash
git clone <repo-url>
cd soft-masked-diffusion-language-models/language
```

---

## Step 2 — Set up the environment

```bash
bash scripts/00_setup_env.sh
```

This does everything in one shot:

1. Creates conda env `sm-env` (Python 3.12)
2. Installs CUDA 12.4 toolkit
3. Installs all Python dependencies (`requirements.txt`)
4. Builds `flash_attn` *(takes 10–20 min; skip with `--skip-flash` if in a hurry)*
5. Logs you into W&B interactively
6. Downloads DUO source files and applies the repo's patches
7. Runs an import + GPU smoke test

**Useful flags:**

```bash
# Skip the flash_attn build (training works fine without it, slightly slower)
bash scripts/00_setup_env.sh --skip-flash

# Non-interactive W&B login (CI / remote machines)
WANDB_API_KEY=<your-key> bash scripts/00_setup_env.sh --skip-wandb

# Custom data cache location
bash scripts/00_setup_env.sh --data-dir /scratch/my_data

# All options
bash scripts/00_setup_env.sh --skip-flash --skip-wandb --env-name my-env --data-dir /scratch/data
```

Activate the environment for subsequent steps:

```bash
conda activate sm-env
```

---

## Step 3 — (Optional) CPU smoke test

Validates the full training pipeline without a GPU. Runs 2 steps only.

```bash
python -u -m main \
  algo=mdlm_sm model=tiny data=text8 \
  algo.tran_head.transparency_alg=slerp_sm \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_steps=2 \
  trainer.num_sanity_val_steps=0 trainer.limit_val_batches=2 \
  trainer.val_check_interval=2 loader.batch_size=4 \
  optim.sm_prob=1.0 sampling.predictor=sm \
  strategy.find_unused_parameters=True \
  data.cache_dir=./data_cache +wandb.offline=true wandb.name=smoke
```

Expect: the run completes without errors and prints `trainer/loss` twice.
Repeat with `algo.tran_head.transparency_alg=mixinputs_with_topk` to confirm
the baseline path too.

---

## Step 4 — Run the A/B experiment

```bash
SEED=1 MAX_STEPS=30000 bash scripts/06_slerp_vs_topk_ablation.sh
```

This launches **two runs sequentially** on a single GPU:

| Run | `transparency_alg` | W&B name |
|---|---|---|
| Baseline | `mixinputs_with_topk` | `topk-tiny-text8-seed1` |
| SLERP | `slerp_sm` | `slerp-tiny-text8-seed1` |

Both share the same seed, batch size, and every other hyperparameter, so the
loss curves are directly comparable. Outputs land in
`outputs/slerp_vs_topk_tiny_text8_seed1/{topk,slerp}/`.

**Knobs (env vars):**

```bash
SEED=1            # random seed — repeat with 2,3 for variance bars
MAX_STEPS=30000   # 30 k gives clear trends; 100 k for cleaner val-ppl numbers
MODEL=tiny        # tiny (~14 M) / small (~125 M) / medium (~355 M)
DATA=text8        # text8 / wikitext2 / ptb
DATA_CACHE_DIR=./data_cache
BATCH_SIZE=64
```

**Two-GPU parallel run** (halves wall-clock time):

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m main \
  algo=mdlm_sm model=tiny data=text8 ... \
  algo.tran_head.transparency_alg=mixinputs_with_topk \
  wandb.name="topk-tiny-text8-seed1" \
  ++hydra.run.dir="outputs/.../topk" &

CUDA_VISIBLE_DEVICES=1 python -u -m main \
  algo=mdlm_sm model=tiny data=text8 ... \
  algo.tran_head.transparency_alg=slerp_sm algo.tran_head.slerp_n_iter=3 \
  wandb.name="slerp-tiny-text8-seed1" \
  ++hydra.run.dir="outputs/.../slerp" &

wait
```

*(Uncomment the parallel block already in `06_slerp_vs_topk_ablation.sh`.)*

**Estimated wall time (single GPU):**

| GPU | Per run | Both runs |
|---|---|---|
| A100 / H100 | ~30–45 min | ~1–1.5 hr |
| RTX 3090/4090 | ~45–70 min | ~1.5–2.5 hr |
| V100 | ~1–1.5 hr | ~2–3 hr |
| T4 | ~2–3 hr | ~4–6 hr |

---

## Step 5 — Monitor in W&B

Open [wandb.ai](https://wandb.ai) → project **`sm-mdlm`** → filter to the run
group `slerp_vs_topk_tiny_text8_seed1`. Both runs appear; overlay them.

### What to watch

**Convergence**

| Metric | What it shows |
|---|---|
| `trainer/loss` | Per-step training loss (logged every 50 steps) |
| `val/nll` | Validation negative log-likelihood (every 1 000 steps) |
| `val/bpd` | Bits per dimension — the canonical text8 metric |

If SLERP helps: its `val/bpd` curve should drop faster (or reach a lower floor)
than the baseline at the same step count.

**Learnable parameters** *(both runs)*

| Metric | What it shows |
|---|---|
| `transparency/scale` | Max mixing weight `ω_s` ∈ (0, 1) |
| `transparency/centre` | Entropy threshold `ω_b` (learned offset) |
| `transparency/steepness` | Sigmoid slope `ω_a` |
| `transparency/temperature` | Softmax temperature (active only for `mixinputs_with_temp`) |

**Realized interpolation** *(live behavior, not just raw knobs)*

| Metric | Run | What it shows |
|---|---|---|
| `transparency/lambda_mean` | Both | Mean realized λ over masked tokens — how strongly feedback is mixed in |
| `transparency/slerp_angle_mean` | SLERP only | Mean angle (radians) between the mask embedding and the Fréchet mean — the geometry of each SLERP step |

A healthy `slerp_angle_mean` is a finite, stable value between ~0.3 and ~2.5 rad.
If it's 0 or π the interpolation has degenerated (mask ≈ prediction or antiparallel).

---

## Step 6 — Interpreting results

| Observation | Interpretation |
|---|---|
| SLERP `val/bpd` lower at same steps | SLERP converges faster ✓ |
| Both curves identical | No effect from interpolation geometry |
| SLERP loss spikes / NaN | Numerical issue — try `slerp_n_iter=1` or reduce `tran_head_lr` |
| `lambda_mean` stays near 0 | Head hasn't learned to mix yet; try `init_scale=0.5` or higher `tran_head_lr` |
| `slerp_angle_mean` ≈ 0 | Mask and Fréchet mean nearly identical — k is too small or embeddings are poorly separated |

**Re-running with more seeds for variance:**

```bash
for SEED in 1 2 3; do
    SEED=$SEED MAX_STEPS=30000 bash scripts/06_slerp_vs_topk_ablation.sh
done
```

Results for all seeds will appear in the same W&B group, making it easy to add
error bars.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'models.dit'`**
→ Run `bash setup.sh` from `language/` to download DUO files.

**`AssertionError` in `get_only_topk_probs`**
→ Unrelated to SLERP; reduce `mixinputs_k` or check batch size.

**flash_attn build fails**
→ Re-run setup with `--skip-flash`; training is unaffected.

**W&B offline / no API key**
→ Add `+wandb.offline=true` to any `python -u -m main ...` call. Logs are saved
locally and can be synced later with `wandb sync`.

**Out of memory**
→ Reduce `BATCH_SIZE=32` or `BATCH_SIZE=16`; the tiny model at batch 16 uses
~1.5 GB VRAM.

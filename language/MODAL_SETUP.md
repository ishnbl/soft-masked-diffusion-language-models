# Running the SLERP vs Top-k A/B Experiment on Modal

## One-time setup

### 1. Install Modal and authenticate
```bash
pip install modal
modal setup        # opens browser for auth
```

### 2. Store your W&B API key
```bash
# Get your key from https://wandb.ai/authorize
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

### 3. Create the three persistent volumes
```bash
modal volume create mdlm-checkpoints   # base MDLM checkpoint
modal volume create mdlm-data          # OWT tokenized dataset cache (~38 GB, downloaded once)
modal volume create mdlm-outputs       # training checkpoints and logs
```

### 4. Upload the base MDLM checkpoint
Download from the Google Drive folder linked in `README.md`, then:
```bash
modal volume put mdlm-checkpoints /path/to/mdlm.ckpt mdlm_owt.ckpt
```
Verify:
```bash
modal volume ls mdlm-checkpoints
# should show: mdlm_owt.ckpt  2.7G
```

---

## Running the experiment

All commands must be run from the `language/` directory:
```bash
cd soft-masked-diffusion-language-models/language/
```

### Smoke test first (always do this before a full run)
Validates the full pipeline in ~30 min, costs ~$1:
```bash
modal run run_modal.py --alg topk --max-steps 100
```
Expected output progression:
1. Image build (first time only, ~5 min — torch + flash_attn download)
2. `[setup] DUO files missing — running setup.sh ...` (first time only, ~10s)
3. `[cache] Valid: /vol/data/owt_cache/openwebtext-train_train_bs1024_wrapped.dat` (after first run)
   OR `Generating new data at: ...` → `Tokenizing ... 100%` (first ever run, ~30 min)
4. `Training: 0%` → runs 100 steps
5. `[train] Done.`

### Full A/B run — parallel (recommended)
Two L40S GPUs simultaneously, ~10–14h, ~$40–55:
```bash
modal run --detach run_modal.py --parallel
```

### Full A/B run — sequential
One L40S GPU, ~20–28h, ~$40–55:
```bash
modal run --detach run_modal.py
```

### Single arm only
```bash
modal run --detach run_modal.py --alg topk
modal run --detach run_modal.py --alg slerp
```

### Faster debug loop
Keep the same microbatch but shrink the effective batch so optimizer steps land
4x sooner:
```bash
modal run run_modal.py --alg topk --max-steps 100 --global-batch-size 128
```

### Recovery variant (if lambda collapses to 0 in W&B)
Freezes backbone for first 1000 steps so head trains against a stable signal:
```bash
modal run --detach run_modal.py --parallel --freeze-until 1000
```

---

## Monitoring

### Stream live logs
```bash
modal app logs slerp-vs-topk-finetune
```
The output is a tqdm progress bar that refreshes in place — watch the microbatch
counter. Prefer W&B for actual progress tracking.

### Check running containers
```bash
modal app list
# look for: slerp-vs-topk-finetune   running   2 containers
```

### Kill a run
```bash
modal app stop slerp-vs-topk-finetune
```

### W&B dashboard
Project: `sm-mdlm` · URL: https://wandb.ai/ishnbl-iit-roorkee/sm-mdlm

Group: `slerp_vs_topk_v2_small_openwebtext-split_seed1`
Runs: `topk-ft-v2-small-openwebtext-split-seed1` and `slerp-ft-v2-small-openwebtext-split-seed1`

**When to expect first data:**

| Event | After launch |
|---|---|
| Run cards appear in W&B | ~2 min |
| First train metrics (loss, lr, lambda_mean, lambda_std) | much sooner now that logs emit every 10 optimizer steps and sanity validation is skipped |
| First val/bpd | ~6–8h (step 200 × 32 grad accum) |

---

## What to watch in W&B

### First 500 steps — is the transparency head activating?

| Metric | Healthy | Red flag |
|---|---|---|
| `transparency/lambda_mean` | rises to 0.2–0.5 by step 500 | stays < 0.05 → abort, rerun with `--freeze-until 1000` |
| `transparency/lambda_std` | > 0 | → 0 = head collapsed to constant |
| `train/loss` | smoothly decreasing | spiking = LR too high |

### Steps 500–5000 — is there signal between arms?

| Metric | What to look for |
|---|---|
| `val/bpd` | two curves visibly separating → SLERP geometry matters |
| `transparency/lambda_mean` | settled around 0.3–0.5 |
| `slerp_angle_mean` (slerp arm only) | 0.8–1.8 rad; near 0 = SLERP degenerate |

### Decision at step 1000
- Both arms learning, lambda healthy, val curves separating → let run to 5000
- Lambda collapsed → `modal app stop` then rerun with `--freeze-until 1000`
- Val curves on top of each other → try `--slerp-n-iter 5`

---

## Downloading results

```bash
# List output files
modal volume ls mdlm-outputs /vol/outputs/

# Download a specific run's outputs
modal volume get mdlm-outputs \
  /vol/outputs/slerp_vs_topk_v2_small_openwebtext-split_seed1 \
  ./local_outputs/
```

---

## GPU and batch size reference

| GPU | VRAM | Batch size | Grad accum | Status |
|---|---|---|---|---|
| A100-40GB | 39.5 GB | 32 | 16 | OOM at logsumexp |
| L40S | 44 GB | 32 | 16 | OOM at logsumexp |
| L40S | 16 | 32 | ✅ running | ~$1.95/h |
| A100-80GB | 80 GB | 32 | 16 | fits, ~$3.40/h |

Current config: **L40S, batch=16, grad_accum=32, global_batch=512**.

---

## Volumes reference

| Volume | Contents | Size |
|---|---|---|
| `mdlm-checkpoints` | `mdlm_owt.ckpt` (base MDLM) | 2.7 GB |
| `mdlm-data` | tokenized OWT `.dat` files + HF Arrow cache | ~40 GB |
| `mdlm-outputs` | Lightning checkpoints + W&B local logs | grows during training |

```bash
modal volume ls mdlm-checkpoints
modal volume ls mdlm-data
modal volume ls mdlm-outputs
```

---

## CLI flags reference for run_modal.py

| Flag | Default | Description |
|---|---|---|
| `--alg` | `both` | `topk`, `slerp`, or `both` |
| `--max-steps` | `5000` | optimizer steps |
| `--parallel` | `False` | spawn two containers simultaneously |
| `--seed` | `1` | random seed |
| `--freeze-until` | `0` | steps to freeze backbone (0 = disabled) |
| `--slerp-n-iter` | `3` | Riemannian iterations for SLERP Fréchet mean |
| `--base-ckpt` | `mdlm_owt.ckpt` | checkpoint filename inside `mdlm-checkpoints` volume |

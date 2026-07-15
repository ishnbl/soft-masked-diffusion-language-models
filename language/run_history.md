# Run History — Last 15 Days (June 30 – July 15, 2026)

All runs are SLERP Soft-Masked MDLM pretraining experiments on **OpenWebText**, executed via **Slurm** on `vitallab1`.  
WandB project: [`slerp-on-smdlm/pretraining`](https://wandb.ai/slerp-on-smdlm/pretraining)

---

## Summary Table

| # | Slurm Job | Date | Type | GPUs | Lambda | SM Prob | Max Steps | Last Step | Outcome | WandB Run |
|---|-----------|------|------|------|--------|---------|-----------|-----------|---------|-----------|
| 1 | `2278` | Jul 10 | SLERP multi-GPU | 2 | **0.3** (fixed) | — | 15,000 | **18,914** / 272,843 | ❌ Cancelled | [`6wjo7v38`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/6wjo7v38) |
| 2 | `2279` | Jul 10–11 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 (unlimited) | **140,735** / 272,843 (52%) | ❌ Time limit | [`bjoqhdku`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/bjoqhdku) |
| 3 | `2296` | Jul 12 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 | 0 (loading data) | ❌ Cancelled instantly | — |
| 4 | `2297` | Jul 12 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 | **771** / 272,843 | ❌ Cancelled | [`gyp01u4n`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/gyp01u4n) |
| 5 | `2299` | Jul 12 | From-scratch (1-GPU) | 1 | **0.0** (fixed) | 0.5 | — | 0 | ❌ Config error | — |
| 6 | `2300` | Jul 12 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 | **2,843** / 272,843 (1%) | ❌ Cancelled | [`wrsi5nhn`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/wrsi5nhn) |
| 7 | `2301` | Jul 12 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 | 0 | ❌ `val_check_interval` error | [`994ucwbs`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/994ucwbs) |
| 8 | `2302` | Jul 12 | From-scratch (1-GPU) | 1 | **0.0** (fixed) | 0→0.5 | -1 | 0 | ❌ `val_check_interval` error | — |
| 9 | `2303` | Jul 12–13 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 | **122,609** / 272,843 (45%) | ❌ Cancelled | [`yery95lv`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/yery95lv) |
| 10 | `2304` | Jul 12 | From-scratch (1-GPU) | 1 | **0.0** (fixed) | 0→0.5 | -1 | **627** / 545,685 | ❌ Cancelled | [`lo65l2q5`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/lo65l2q5) |
| 11 | `2305` | Jul 12–14 | From-scratch (1-GPU) | 1 | **0.0** (fixed) | **0** (no ramp) | -1 | **352,000** / 545,685 (65%) | ⚠️ Crashed (wandb error) | [`lo65l2q5`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/lo65l2q5) |
| 12 | `2311` | Jul 13–14 | SLERP multi-GPU | 2 | **learned** (entropy-gated) | — | -1 | **176,000** / 272,843 (65%) | ⚠️ Crashed (checkpoint save error) | [`w3iken14`](https://wandb.ai/slerp-on-smdlm/pretraining/runs/w3iken14) |

---

## Detailed Run Descriptions

### Run 1 — Job 2278 (Jul 10) — SLERP Multi-GPU, Fixed λ=0.3
- **Script**: [pretrain_scratch_slerp.sbatch](file:///home/aryan_s2/mdlm/soft-masked-diffusion-language-models/language/pretrain_scratch_slerp.sbatch) → `run_verda.sh`
- **Config**: 2 GPUs, batch 16×2 GPUs × 16 accum = 512 global, reliability-conditioned schedule
- **Output dir**: `~/sm_outputs/slerp_mg_2gpu_small_openwebtext-split_seed1`
- **Progress**: Reached step 18,914 (~7%) at 2.0 it/s, then **cancelled manually**
- **Duration**: ~2h 39m

### Run 2 — Job 2279 (Jul 10–11) — SLERP Multi-GPU, Learned λ
- **Config**: Same 2-GPU setup, learned entropy-gated lambda, unlimited steps
- **Progress**: Reached step 140,735 (**52%**) at 1.63 it/s
- **Outcome**: **Hit Slurm time limit** after ~24h
- **Duration**: ~24h

### Run 3 — Job 2296 (Jul 12) — SLERP Multi-GPU, Learned λ
- **Config**: Same as Run 2, re-submitted
- **Outcome**: **Cancelled** during data loading, before any training started
- **Duration**: <1 minute

### Run 4 — Job 2297 (Jul 12) — SLERP Multi-GPU, Learned λ
- **Config**: Same as Run 2, re-submitted
- **Progress**: Reached step 771, then **cancelled manually**
- **Duration**: ~6 minutes

### Run 5 — Job 2299 (Jul 12) — From-Scratch 1-GPU, λ=0.0
- **Script**: [scratch_lam0.0.sbatch](file:///home/aryan_s2/mdlm/soft-masked-diffusion-language-models/language/scratch_lam0.0.sbatch)
- **Outcome**: **Config error** — `Unknown arg: callbacks.checkpoint_every_n_steps.save_top_k=-1`
- **Duration**: instant

### Run 6 — Job 2300 (Jul 12) — SLERP Multi-GPU, Learned λ
- **Config**: Same as Run 2
- **Progress**: Reached step 2,843 (~1%), then **cancelled manually**
- **Duration**: ~23 minutes

### Run 7 — Job 2301 (Jul 12) — SLERP Multi-GPU, Learned λ
- **Config**: Same as Run 2
- **Outcome**: **Config error** — `val_check_interval (20000000) must be less than or equal to training batches (272843)`
- **Duration**: instant

### Run 8 — Job 2302 (Jul 12) — From-Scratch 1-GPU, λ=0.0, SM 0→0.5
- **Script**: [scratch_lam0.0.sbatch](file:///home/aryan_s2/mdlm/soft-masked-diffusion-language-models/language/scratch_lam0.0.sbatch)
- **Config**: 1 GPU, batch 16 × 32 accum = 512, sm_prob ramp 0→0.5 over 2000 steps
- **Outcome**: **Same `val_check_interval` error** as Job 2301
- **Duration**: instant

### Run 9 — Job 2303 (Jul 12–13) — SLERP Multi-GPU, Learned λ
- **Config**: Same as Run 2
- **Progress**: Reached step 122,609 (**45%**) at 1.74 it/s, then **cancelled manually**
- **Duration**: ~19h 36m

### Run 10 — Job 2304 (Jul 12) — From-Scratch 1-GPU, λ=0.0, SM 0→0.5
- **Config**: 1 GPU, sm_prob ramp 0→0.5, fixed λ=0.0
- **Progress**: Reached step 627, then **cancelled manually**
- **Duration**: ~5 minutes

### Run 11 — Job 2305 (Jul 12–14) — From-Scratch 1-GPU, λ=0.0, SM=0 (Baseline)
- **Config**: 1 GPU, **sm_prob = 0 (no soft masking)**, fixed λ=0.0 — this is essentially a **pure MDLM baseline**
- **Output dir**: `~/sm_outputs/scratch_slerp_1gpu_small_openwebtext-split_seed1_sm0_lam0.0`
- **Progress**: Reached step **352,000** / 545,685 (**65%**) at 2.08 it/s
- **Outcome**: **Crashed** — wandb `HandleAbandonedError`
- **Duration**: ~47h
- **Checkpoints saved**: 0-2500 through 0-10500, plus `last.ckpt` (every 500 steps)

### Run 12 — Job 2311 (Jul 13–14) — SLERP Multi-GPU, Learned λ (Longest 2-GPU Run)
- **Config**: 2 GPUs, learned entropy-gated lambda
- **Output dir**: `~/sm_outputs/slerp_mg_2gpu_small_openwebtext-split_seed1`
- **Progress**: Reached step **176,000** / 272,843 (**65%**) at 1.78 it/s
- **Outcome**: **Crashed** during checkpoint save (Lightning `ModelCheckpoint` error)
- **Duration**: ~27h 28m
- **Checkpoints saved**: 0-1000 through 0-11000, plus `last-v1.ckpt` (every 500 steps)

---

## Saved Checkpoints

### `~/sm_outputs/slerp_mg_2gpu_small_openwebtext-split_seed1/checkpoints/`
From Jobs 2303 + 2311 (2-GPU, learned λ):
| Checkpoint | Date | Notes |
|------------|------|-------|
| `0-1000.ckpt` – `0-7500.ckpt` | Jul 13 | From job 2303 (first pass) |
| `0-1000.ckpt` (overwritten) – `0-7500-v1.ckpt` | Jul 14 | From job 2311 (resumed, -v1 variants) |
| `0-8000.ckpt` – `0-11000.ckpt` | Jul 14 | New checkpoints from job 2311 |
| `last-v1.ckpt` | Jul 14 20:18 | Latest state |

### `~/sm_outputs/scratch_slerp_1gpu_small_openwebtext-split_seed1_sm0_lam0.0/checkpoints/`
From Job 2305 (1-GPU, λ=0.0, SM=0 baseline):
| Checkpoint | Date |
|------------|------|
| `0-2500.ckpt` – `0-10500.ckpt` | Jul 13–14 |
| `last.ckpt` | Jul 14 19:54 |

---

## Key Observations

1. **Two main experiment tracks** were running in parallel:
   - **2-GPU SLERP with learned λ** (entropy-gated) — the primary experiment
   - **1-GPU baseline with λ=0.0, sm_prob=0** — a pure MDLM baseline (no soft masking)

2. **Both longest runs (2305 and 2311) crashed at ~65% completion** — 2305 from a wandb error, 2311 from a checkpoint-saving error.

3. **Many short-lived runs** (2296, 2297, 2299, 2300, 2301, 2302, 2304) were configuration debugging iterations — fixing `val_check_interval`, `save_top_k`, and other config issues.

4. **Job 2278** was an earlier experiment with **fixed λ=0.3** (not learned), cancelled after only 7%.

5. **No runs have completed** to 100% in the last 15 days. The furthest progress was 65% on both tracks.

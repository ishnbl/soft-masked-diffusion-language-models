# Running on a Verda GPU VM

Single-node, multi-GPU launchers for the SLERP and top-k soft-masking paths,
mirroring the Modal runners (`run_modal_multigpu.py` / `run_modal_multigpu_topk.py`)
exactly so a VM run reproduces a Modal run. Targets **one Verda GPU instance with
N GPUs** (e.g. 2x/4x/8x A100) — single-node DDP, which is what the DDP-safe
soft-mask gate covers.

| File | Purpose |
|------|---------|
| `setup.sh` | One-time VM bootstrap: Miniconda + Python 3.12, the exact Modal-pinned deps (torch 2.5.1 cu124, prebuilt flash-attn wheel), DUO source files, W&B login. |
| `run_slerp.sh` | Launch `slerp_sm` under N-GPU DDP. |
| `run_topk.sh` | Launch `mixinputs_with_topk` under N-GPU DDP. |

The two launchers are independent (no shared state), same as the two Modal
scripts. None of the existing Modal scripts are touched.

## 1. Provision the instance (Verda console)

Following <https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/>:

1. **Instance type**: Pay-as-you-go (or spot if you can tolerate eviction).
2. **GPU**: pick a multi-GPU card/size (e.g. 4x A100). All visible GPUs are used.
3. **OS**: Ubuntu (the GPU image ships the NVIDIA driver).
4. **Storage**: add a block volume *in the same datacenter* for the dataset cache
   and checkpoints (the OWT cache is tens of GB; keep it on the volume so it
   survives instance teardown). Note its mount path, e.g. `/data`.
5. **SSH key**: add your public key.
6. **(Optional) Startup script**: you can paste the contents of `setup.sh` here to
   bootstrap on first boot, but running it manually after `git clone` is simpler.

SSH in once it's running:

```bash
ssh ubuntu@<instance-ip>
```

## 2. Get the code + checkpoint onto the VM

```bash
# code
git clone <this-repo-url>
cd <repo>/language

# base checkpoint (finetune source) onto the attached volume
#   from your laptop:
scp /local/path/mdlm_owt.ckpt ubuntu@<instance-ip>:/data/mdlm_owt.ckpt
```

## 3. One-time setup

```bash
cd <repo>/language
WANDB_API_KEY=<your-key> bash verda/setup.sh
conda activate sm-env
```

This reproduces the Modal image environment, so runs match Modal. It skips the
~20-min flash-attn build by using the prebuilt cp312 wheel.

## 4. Launch

Both launchers auto-detect the GPU count (override with `NUM_GPUS`) and require a
`BASE_CKPT` (these are finetune runs, like the Modal runners). Outputs and W&B
naming match Modal: project `sm-mdlm`, group `slerp_mg_<N>gpu_..._seed<seed>` /
`topk_mg_<N>gpu_..._seed<seed>`.

```bash
# SLERP, all GPUs, finetune from the checkpoint
BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp.sh

# top-k, fixed lambda = 0.5, seed 2, 4 GPUs
NUM_GPUS=4 SEED=2 FIXED_LAMBDA=0.5 BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_topk.sh

# quick smoke test (few steps, small batch)
MAX_STEPS=20 GLOBAL_BATCH=64 PER_GPU_BATCH=16 \
  BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp.sh
```

### Knobs (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `NUM_GPUS` | all visible | world size; pins `CUDA_VISIBLE_DEVICES` to the first N |
| `SEED` | `1` | seed (also seeds the DDP soft-mask gate); appears in run name |
| `MAX_STEPS` | `5000` | training steps |
| `GLOBAL_BATCH` | `512` | effective batch, held fixed across GPU counts |
| `PER_GPU_BATCH` | `16` | micro-batch per GPU (lower on OOM) |
| `FIXED_LAMBDA` | `-1.0` | `>=0` pins lambda (must be in `[0,1]`); `-1` = learned |
| `SLERP_N_ITER` | `3` | (slerp only) Frechet-mean iterations |
| `MIXINPUTS_K` | `3` | top-k tokens fed to the mix |
| `BASE_CKPT` | — | **required** finetune source `.ckpt` |
| `DATA_CACHE_DIR` | `./data_cache/owt_cache` | put this on the attached volume, e.g. `/data/owt_cache` |
| `OUT_ROOT` | `./outputs` | run/checkpoint output root |

Tip: point `DATA_CACHE_DIR` and `OUT_ROOT` at the attached volume so the OWT
cache and checkpoints persist across instance teardown:

```bash
DATA_CACHE_DIR=/data/owt_cache OUT_ROOT=/data/outputs \
  BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_topk.sh
```

## Notes

- **Spot instances** can be evicted; checkpoints are written every
  `CHECKPOINT_EVERY` (default 100) steps to `OUT_ROOT` — keep that on a volume so
  you can resume.
- **GPU count and the world size stay in lockstep**: this codebase derives the
  world size from `torch.cuda.device_count()`, so the launcher pins
  `CUDA_VISIBLE_DEVICES` to exactly `NUM_GPUS` before any Python runs.
- **Multi-node** (several VMs) is out of scope here — these are single-node N-GPU
  launchers. Multi-node would need a torchrun rendezvous + NCCL-over-network
  setup on top.

#!/bin/bash
#
# Turnkey multi-GPU launcher for the SLERP soft-masking path (`slerp_sm`).
# ---------------------------------------------------------------------------
# Runs the `slerp_sm` transparency pathway under PyTorch-Lightning DDP on N GPUs
# of a single node. Auto-detects the GPU count (override with NUM_GPUS) and
# derives the per-GPU micro-batch / gradient-accumulation so the global batch is
# held fixed regardless of how many GPUs you launch on — i.e. a 2-GPU run and a
# 4-GPU run optimize the SAME effective batch, just faster on 4.
#
# WHY THIS IS DDP-SAFE
# --------------------
# The soft-mask gate (Bernoulli `sm_prob` + time-band) is now decided
# identically on every rank (algo.py MDLM_SM.nll: step-seeded Bernoulli + global
# all-reduced t-mean), so all ranks take the same soft-mask branch every step
# and the DDP gradient all-reduce stays in sync. `find_unused_parameters=True`
# is still required because on non-soft-mask steps the tran_head params legitimately
# receive no gradient on ALL ranks.
#
# USAGE
# -----
#   # 4 GPUs, from scratch:
#   bash scripts/run_slerp_multigpu.sh
#
#   # 2 GPUs, finetuning from the released MDLM checkpoint:
#   NUM_GPUS=2 BASE_CKPT=/path/to/mdlm.ckpt bash scripts/run_slerp_multigpu.sh
#
#   # custom global batch + per-GPU micro-batch:
#   NUM_GPUS=4 GLOBAL_BATCH=512 PER_GPU_BATCH=32 bash scripts/run_slerp_multigpu.sh
#
# KNOBS (env vars, with defaults)
# -------------------------------
NUM_GPUS="${NUM_GPUS:-}"                  # default: all visible GPUs
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"                   # matches the released OWT backbone
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"       # effective batch (held fixed across GPU counts)
PER_GPU_BATCH="${PER_GPU_BATCH:-32}"      # micro-batch per GPU (lower this on OOM)
SLERP_N_ITER="${SLERP_N_ITER:-3}"         # Frechet-mean iterations
MIXINPUTS_K="${MIXINPUTS_K:-3}"           # top-k tokens fed to the Frechet mean
TRAN_HEAD_LR="${TRAN_HEAD_LR:-0.01}"
SM_PROB="${SM_PROB:-0.8}"
BASE_CKPT="${BASE_CKPT:-}"                # optional: finetune from this .ckpt
RUN_DIR="${RUN_DIR:-}"

set -euo pipefail

# --- Resolve GPU count -------------------------------------------------------
if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi

# --- Preflight: GPUs --------------------------------------------------------
python - "$NUM_GPUS" <<'PY'
import sys, torch
want = int(sys.argv[1])
have = torch.cuda.device_count()
if have == 0:
    sys.exit("\n[preflight] No GPU visible to PyTorch. This path requires >=1 GPU.\n")
if want > have:
    sys.exit(f"\n[preflight] Requested NUM_GPUS={want} but only {have} visible.\n")
print(f"[preflight] launching on {want}/{have} visible GPU(s) — OK")
PY

# --- Preflight: global batch must divide cleanly -----------------------------
if (( GLOBAL_BATCH % (NUM_GPUS * PER_GPU_BATCH) != 0 )); then
  echo "[preflight] GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by"
  echo "            NUM_GPUS*PER_GPU_BATCH = ${NUM_GPUS}*${PER_GPU_BATCH}."
  echo "            The dataloader asserts global == per_gpu * gpus * accum."
  echo "            Pick GLOBAL_BATCH as a multiple of $((NUM_GPUS * PER_GPU_BATCH))."
  exit 1
fi
ACCUM=$(( GLOBAL_BATCH / (NUM_GPUS * PER_GPU_BATCH) ))

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="outputs/slerp_sm_${NUM_GPUS}gpu_${MODEL}_seed${SEED}"
fi
mkdir -p "$RUN_DIR"

echo "[config] GPUs=${NUM_GPUS}  per-GPU batch=${PER_GPU_BATCH}  accum=${ACCUM}  global=${GLOBAL_BATCH}"
echo "[config] run dir: ${RUN_DIR}"

ARGS=(
  algo=mdlm_sm
  algo.tran_head.transparency_alg=slerp_sm
  algo.tran_head.slerp_n_iter="$SLERP_N_ITER"
  algo.tran_head.mixinputs_k="$MIXINPUTS_K"
  # lambda-activation init (same rationale as scripts 07/09: config defaults
  # drive lambda->0 and slerp collapses to vanilla MDLM).
  algo.tran_head.init_scale=0.3
  algo.tran_head.init_centre=-2.5
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  # --- multi-GPU DDP ---
  strategy=ddp
  strategy.find_unused_parameters=True
  trainer.devices="$NUM_GPUS"
  trainer.num_nodes=1
  trainer.max_steps="$MAX_STEPS"
  trainer.val_check_interval=500
  trainer.log_every_n_steps=50
  loader.global_batch_size="$GLOBAL_BATCH"
  loader.batch_size="$PER_GPU_BATCH"
  loader.eval_batch_size="$PER_GPU_BATCH"
  optim.tran_head_lr="$TRAN_HEAD_LR"
  optim.sm_prob="$SM_PROB"
  sampling.predictor=sm
  eval.compute_generative_perplexity=False
  wandb.name="slerp-sm-${NUM_GPUS}gpu-${MODEL}-seed${SEED}"
  wandb.group="slerp_sm_multigpu"
)

# Optional finetune-from-checkpoint path.
if [[ -n "$BASE_CKPT" ]]; then
  if [[ ! -f "$BASE_CKPT" ]]; then
    echo "[preflight] BASE_CKPT not found: $BASE_CKPT"; exit 1
  fi
  echo "[config] finetuning from BASE_CKPT=${BASE_CKPT}"
  ARGS+=(
    training.finetune_path="$BASE_CKPT"
    # Don't let a stale best.ckpt in the run dir override finetune_path.
    checkpointing.resume_from_ckpt=false
    lr_scheduler.num_warmup_steps=200
  )
fi

set -x
python -u -m main "${ARGS[@]}" ++hydra.run.dir="$RUN_DIR"

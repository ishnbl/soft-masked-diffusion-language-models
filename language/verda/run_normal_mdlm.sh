#!/bin/bash
#
# verda/run_normal_mdlm.sh — single-node multi-GPU launcher for a normal MDLM run
# with no intermediate step or validation checkpoints (only saving last.ckpt).
#
# USAGE (from language/, env activated):
#   BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_normal_mdlm.sh
#   NUM_GPUS=2 BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_normal_mdlm.sh
#
# KNOBS (env vars):
NUM_GPUS="${NUM_GPUS:-}"                       # default: all visible GPUs
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache/owt_cache}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-3}"
BASE_CKPT="${BASE_CKPT:-}"                      # REQUIRED: finetune-from checkpoint
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
OUT_ROOT="${OUT_ROOT:-./outputs}"

set -euo pipefail

# Run from language/ (this script lives in language/verda/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

# --- Resolve GPU count + pin CUDA_VISIBLE_DEVICES ---------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE_N="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
  if [[ -n "$NUM_GPUS" && "$NUM_GPUS" != "$VISIBLE_N" ]]; then
    echo "[preflight] CUDA_VISIBLE_DEVICES exposes ${VISIBLE_N} GPU(s) but NUM_GPUS=${NUM_GPUS}."
    echo "            Unset one (CUDA_VISIBLE_DEVICES wins for specific GPUs)."; exit 1
  fi
  NUM_GPUS="$VISIBLE_N"
  echo "[preflight] honoring CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPU[s])"
else
  TOTAL_N="$(python -c 'import torch; print(torch.cuda.device_count())')"
  if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS="$TOTAL_N"
    echo "[preflight] NUM_GPUS not set — default to all visible GPUs (${NUM_GPUS})"
  fi
  if [[ "$NUM_GPUS" -gt "$TOTAL_N" ]]; then
    echo "[preflight] requested NUM_GPUS=${NUM_GPUS} but only ${TOTAL_N} GPU(s) exist."; exit 1
  fi
  GPUS_SEQ="$(seq -s, 0 $((NUM_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES="$GPUS_SEQ"
  echo "[preflight] setting CUDA_VISIBLE_DEVICES=${GPUS_SEQ} to restrict device_count()"
fi

if [[ -z "$BASE_CKPT" ]]; then
  echo "[preflight] BASE_CKPT env var is required (path to base .ckpt file)."; exit 1
fi
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[preflight] checkpoint not found: ${BASE_CKPT}"; exit 1
fi

# Assert batch size parameters are divisible
if [[ $(( GLOBAL_BATCH % (NUM_GPUS * PER_GPU_BATCH) )) -ne 0 ]]; then
  echo "[preflight] ERROR: global_batch (${GLOBAL_BATCH}) must be divisible by num_gpus * batch_size (${NUM_GPUS} * ${PER_GPU_BATCH} = $((NUM_GPUS * PER_GPU_BATCH)))"
  exit 1
fi
ACCUM=$(( GLOBAL_BATCH / (NUM_GPUS * PER_GPU_BATCH) ))

# Prepare Output Directories
GROUP="mdlm_mg_${NUM_GPUS}gpu_${MODEL}_${DATA}_seed${SEED}"
RUN_NAME="mdlm-mg-${NUM_GPUS}gpu-${MODEL}-${DATA}-seed${SEED}"
OUT_DIR="${OUT_ROOT}/${GROUP}"
mkdir -p "$OUT_DIR"

echo "[config] alg=mdlm  GPUs=${NUM_GPUS}  per-GPU=${PER_GPU_BATCH}  accum=${ACCUM}  global=${GLOBAL_BATCH}"
echo "[config] seed=${SEED}  steps=${MAX_STEPS}  ckpt=${BASE_CKPT}"
echo "[config] out_dir=${OUT_DIR}"

ARGS=(
  algo=mdlm
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  loader.global_batch_size="$GLOBAL_BATCH"
  loader.eval_global_batch_size="$GLOBAL_BATCH"
  loader.batch_size="$PER_GPU_BATCH"
  loader.eval_batch_size="$PER_GPU_BATCH"
  loader.num_workers="$NUM_WORKERS"
  strategy=ddp
  strategy.find_unused_parameters=False
  trainer.devices="$NUM_GPUS"
  trainer.num_nodes=1
  trainer.max_steps="$MAX_STEPS"
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches=0
  trainer.log_every_n_steps="$LOG_EVERY"
  optim.lr=3e-5
  lr_scheduler.num_warmup_steps=200
  eval.compute_generative_perplexity=False
  eval.generate_samples=False
  training.finetune_path="$BASE_CKPT"
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_every_n_steps.every_n_train_steps="$CHECKPOINT_EVERY"
  wandb.project="finetune-main"
  wandb.entity="slerp-on-smdlm"
  wandb.group="$GROUP"
  wandb.name="$RUN_NAME"
  ++hydra.run.dir="$OUT_DIR"
  ++checkpointing.save_dir="$OUT_DIR"
)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
set -x
python -u -m main "${ARGS[@]}"

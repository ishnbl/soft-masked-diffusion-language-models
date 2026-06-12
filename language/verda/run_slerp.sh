#!/bin/bash
#
# verda/run_slerp.sh — single-node multi-GPU SLERP (`slerp_sm`) launcher for a
# Verda GPU VM. Hydra args mirror run_modal_multigpu.py EXACTLY, so a VM run
# reproduces the Modal run (same init, same soft-mask band, validation disabled,
# fixed-lambda support). This is the slerp counterpart to verda/run_topk.sh and
# shares no state with it.
#
# DDP-safety: the soft-mask gate (algo.py MDLM_SM.nll) decides the branch
# identically on every rank (step-seeded Bernoulli + globally all-reduced
# t-band), so all ranks stay in sync. find_unused_parameters=True is required
# because tran_head params receive no grad on non-soft-mask steps (on ALL ranks).
#
# USAGE (from language/, env activated):
#   BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp.sh
#   NUM_GPUS=4 BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp.sh
#   FIXED_LAMBDA=0.5 BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp.sh
#   SEED=2 MAX_STEPS=5000 GLOBAL_BATCH=512 PER_GPU_BATCH=16 bash verda/run_slerp.sh
#
# KNOBS (env vars, defaults match the Modal runner):
NUM_GPUS="${NUM_GPUS:-}"                       # default: all visible GPUs
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache/owt_cache}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SLERP_N_ITER="${SLERP_N_ITER:-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-3}"
FIXED_LAMBDA="${FIXED_LAMBDA:--1.0}"           # >=0 pins lambda (must be in [0,1])
BASE_CKPT="${BASE_CKPT:-}"                      # REQUIRED: finetune-from checkpoint
OUT_ROOT="${OUT_ROOT:-./outputs}"

set -euo pipefail

# Run from language/ (this script lives in language/verda/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

# --- Resolve GPU count + pin CUDA_VISIBLE_DEVICES ---------------------------
# This codebase derives the world size from torch.cuda.device_count() (the Hydra
# device_count: resolver + the dataloader's global==batch*device_count*accum
# assertion), NOT trainer.devices. So we make device_count() == NUM_GPUS by
# pinning CUDA_VISIBLE_DEVICES before any Python runs. If the caller already set
# it, we honor it and derive NUM_GPUS from it.
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
  [[ -z "$NUM_GPUS" ]] && NUM_GPUS="$TOTAL_N"
  if (( NUM_GPUS < 1 )); then echo "[preflight] No GPU visible to PyTorch."; exit 1; fi
  if (( NUM_GPUS > TOTAL_N )); then
    echo "[preflight] Requested NUM_GPUS=${NUM_GPUS} but only ${TOTAL_N} GPU(s) visible."; exit 1
  fi
  if (( NUM_GPUS < TOTAL_N )); then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
    echo "[preflight] ${TOTAL_N} GPU(s) visible; pinning CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
fi

python - "$NUM_GPUS" <<'PY'
import os, sys, torch
want = int(sys.argv[1]); have = torch.cuda.device_count()
if have != want:
    sys.exit(f"[preflight] expected {want} visible GPU(s) but torch sees {have} "
             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}).")
print(f"[preflight] {want} GPU(s) visible — OK")
PY

# --- Preflight checks (mirror the Modal runner) -----------------------------
if [[ -z "$BASE_CKPT" ]]; then
  echo "[preflight] BASE_CKPT is required (this is a finetune run, like the Modal runner)."
  echo "            e.g. BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp.sh"; exit 1
fi
if [[ ! -f "$BASE_CKPT" ]]; then echo "[preflight] BASE_CKPT not found: $BASE_CKPT"; exit 1; fi

if (( GLOBAL_BATCH % (NUM_GPUS * PER_GPU_BATCH) != 0 )); then
  echo "[preflight] GLOBAL_BATCH=${GLOBAL_BATCH} not divisible by NUM_GPUS*PER_GPU_BATCH"
  echo "            = ${NUM_GPUS}*${PER_GPU_BATCH}. Pick a multiple of $((NUM_GPUS*PER_GPU_BATCH))."; exit 1
fi
ACCUM=$(( GLOBAL_BATCH / (NUM_GPUS * PER_GPU_BATCH) ))

# fixed-lambda range check (bash float compare via awk).
USE_FL="$(awk -v x="$FIXED_LAMBDA" 'BEGIN{print (x>=0)?1:0}')"
if [[ "$USE_FL" == "1" ]]; then
  OK="$(awk -v x="$FIXED_LAMBDA" 'BEGIN{print (x>=0 && x<=1)?1:0}')"
  if [[ "$OK" != "1" ]]; then echo "[preflight] FIXED_LAMBDA must be in [0,1], got $FIXED_LAMBDA"; exit 1; fi
fi

GROUP="slerp_mg_${NUM_GPUS}gpu_${MODEL}_${DATA}_seed${SEED}"
RUN_NAME="slerp-mg-${NUM_GPUS}gpu-${MODEL}-${DATA}-seed${SEED}"
OUT_DIR="${OUT_ROOT}/${GROUP}"
mkdir -p "$OUT_DIR"

echo "[config] alg=slerp_sm  GPUs=${NUM_GPUS}  per-GPU=${PER_GPU_BATCH}  accum=${ACCUM}  global=${GLOBAL_BATCH}"
echo "[config] seed=${SEED}  steps=${MAX_STEPS}  ckpt=${BASE_CKPT}"
[[ "$USE_FL" == "1" ]] && echo "[config] fixed_lambda=${FIXED_LAMBDA}"
echo "[config] out_dir=${OUT_DIR}"

ARGS=(
  algo=mdlm_sm
  algo.tran_head.transparency_alg=slerp_sm
  algo.tran_head.slerp_n_iter="$SLERP_N_ITER"
  algo.tran_head.mixinputs_k=3
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-4.33
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
  strategy.find_unused_parameters=True
  trainer.devices="$NUM_GPUS"
  trainer.num_nodes=1
  trainer.max_steps="$MAX_STEPS"
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches=0
  trainer.log_every_n_steps="$LOG_EVERY"
  optim.lr=3e-5
  optim.tran_head_lr=0.01
  optim.sm_prob=0.8
  optim.sm_t_min=0.2
  optim.sm_t_max=0.8
  lr_scheduler.num_warmup_steps=200
  sampling.predictor=sm
  eval.compute_generative_perplexity=False
  eval.generate_samples=False
  training.finetune_path="$BASE_CKPT"
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_hf_offload.every_n_train_steps="$CHECKPOINT_EVERY"
  wandb.group="$GROUP"
  wandb.name="$RUN_NAME"
  ++hydra.run.dir="$OUT_DIR"
  ++checkpointing.save_dir="$OUT_DIR"
)
[[ "$USE_FL" == "1" ]] && ARGS+=( algo.tran_head.fixed_lambda="$FIXED_LAMBDA" )

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
set -x
python -u -m main "${ARGS[@]}"

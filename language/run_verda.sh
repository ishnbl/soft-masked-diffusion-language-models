#!/usr/bin/env bash
# ============================================================
# run_verda.sh — Drop-in replacement for run_modal_multigpu.py
# Runs the slerp_sm soft-masking MDLM finetune on a Verda VM.
# All hydra args are identical to the Modal multi-GPU runner.
#
# Usage (from inside the language/ directory):
#   bash run_verda.sh [options]
#
# Options (all optional — defaults shown):
#   --num-gpus              2
#   --base-ckpt             /workspace/checkpoints/mdlm_owt.ckpt
#   --seed                  1
#   --max-steps             5000
#   --model                 small
#   --data                  openwebtext-split
#   --slerp-n-iter          3
#   --global-batch-size     512
#   --batch-size            16
#   --num-workers           4
#   --checkpoint-every      100
#   --log-every             3
#   --data-dir              /workspace/data
#   --out-dir               /workspace/outputs
#   --fixed-lambda          -1.0  (disabled by default; set to a value in [0,1]
#                                  to pin lambda and skip the learned lambda head)
#
# Fixed-lambda example:
#   bash run_verda.sh --num-gpus 4 --max-steps 5000 --fixed-lambda 0.5
#
# Quick smoke test (2 GPUs, 20 steps):
#   bash run_verda.sh --num-gpus 2 --max-steps 20 --global-batch-size 64 --batch-size 16
#
# Full 4-GPU run:
#   bash run_verda.sh --num-gpus 4 --max-steps 5000
# ============================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────
NUM_GPUS=2
BASE_CKPT="/workspace/checkpoints/mdlm_owt.ckpt"
SEED=1
MAX_STEPS=5000
MODEL="small"
DATA="openwebtext-split"
SLERP_N_ITER=3
GLOBAL_BATCH_SIZE=512
BATCH_SIZE=16
NUM_WORKERS=4
CHECKPOINT_EVERY=100
LOG_EVERY=3
DATA_DIR="/workspace/data"
OUT_DIR="/workspace/outputs"
FIXED_LAMBDA=-1.0

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus)            NUM_GPUS="$2";           shift 2 ;;
    --base-ckpt)           BASE_CKPT="$2";          shift 2 ;;
    --seed)                SEED="$2";               shift 2 ;;
    --max-steps)           MAX_STEPS="$2";          shift 2 ;;
    --model)               MODEL="$2";              shift 2 ;;
    --data)                DATA="$2";               shift 2 ;;
    --slerp-n-iter)        SLERP_N_ITER="$2";       shift 2 ;;
    --global-batch-size)   GLOBAL_BATCH_SIZE="$2";  shift 2 ;;
    --batch-size)          BATCH_SIZE="$2";         shift 2 ;;
    --num-workers)         NUM_WORKERS="$2";        shift 2 ;;
    --checkpoint-every)    CHECKPOINT_EVERY="$2";   shift 2 ;;
    --log-every)           LOG_EVERY="$2";          shift 2 ;;
    --data-dir)            DATA_DIR="$2";           shift 2 ;;
    --out-dir)             OUT_DIR="$2";            shift 2 ;;
    --fixed-lambda)        FIXED_LAMBDA="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Preflight checks ─────────────────────────────────────────
if [ ! -f "$BASE_CKPT" ]; then
  echo "ERROR: Checkpoint not found at $BASE_CKPT"
  echo "Download it from Google Drive and place it at $BASE_CKPT"
  echo "  (or pass --base-ckpt /your/actual/path/mdlm_owt.ckpt)"
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
  echo "ERROR: CUDA not available. Check your GPU drivers."
  exit 1
fi

VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "ERROR: Requested --num-gpus $NUM_GPUS but only $VISIBLE_GPUS GPU(s) visible."
  exit 1
fi

if [ $(( GLOBAL_BATCH_SIZE % (NUM_GPUS * BATCH_SIZE) )) -ne 0 ]; then
  echo "ERROR: global_batch_size ($GLOBAL_BATCH_SIZE) must be divisible by num_gpus * batch_size ($NUM_GPUS * $BATCH_SIZE = $((NUM_GPUS * BATCH_SIZE)))"
  exit 1
fi

ACCUM=$(( GLOBAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE) ))

# Validate fixed_lambda: must be -1.0 (disabled) or in [0, 1]
if ! python -c "
v = float('${FIXED_LAMBDA}')
if v >= 0.0 and not 0.0 <= v <= 1.0:
    import sys; sys.stderr.write('ERROR: --fixed-lambda must lie in [0, 1], got ' + str(v) + '\n'); sys.exit(1)
"; then
  exit 1
fi

# ── Setup DUO files if missing ────────────────────────────────
if [ ! -f "models/dit.py" ]; then
  echo "[setup] DUO files missing — running setup.sh ..."
  bash setup.sh
else
  echo "[setup] DUO files present — skipping setup.sh"
fi

# ── Directories ───────────────────────────────────────────────
mkdir -p "$DATA_DIR/owt_cache" "$OUT_DIR"

GROUP="slerp_mg_${NUM_GPUS}gpu_${MODEL}_${DATA}_seed${SEED}"
RUN_NAME="slerp-mg-${NUM_GPUS}gpu-${MODEL}-${DATA}-seed${SEED}"
RUN_OUT_DIR="${OUT_DIR}/${GROUP}"
mkdir -p "$RUN_OUT_DIR"

echo ""
echo "========================================"
echo " SLERP Soft-Masked MDLM — Verda Run"
echo "========================================"
echo " GPUs:          $NUM_GPUS"
echo " Per-GPU batch: $BATCH_SIZE"
echo " Accum steps:   $ACCUM"
echo " Global batch:  $GLOBAL_BATCH_SIZE"
echo " Max steps:     $MAX_STEPS"
echo " Checkpoint:    $BASE_CKPT"
echo " Output dir:    $RUN_OUT_DIR"
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  echo " Fixed lambda:  $FIXED_LAMBDA"
else
  echo " Lambda:        learned (entropy-gated)"
fi
echo "========================================"
echo ""

# ── Environment ───────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Launch training ───────────────────────────────────────────
# Build command as array so fixed_lambda can be conditionally appended
# (matches the exact same conditional logic in run_modal_multigpu.py)
CMD=(
  python -u -m main
  algo=mdlm_sm
  algo.tran_head.transparency_alg=slerp_sm
  algo.tran_head.slerp_n_iter="${SLERP_N_ITER}"
  algo.tran_head.mixinputs_k=3
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-4
  model="${MODEL}"
  data="${DATA}"
  data.cache_dir="${DATA_DIR}/owt_cache"
  seed="${SEED}"
  loader.global_batch_size="${GLOBAL_BATCH_SIZE}"
  loader.eval_global_batch_size="${GLOBAL_BATCH_SIZE}"
  loader.batch_size="${BATCH_SIZE}"
  loader.eval_batch_size="${BATCH_SIZE}"
  loader.num_workers="${NUM_WORKERS}"
  strategy=ddp
  strategy.find_unused_parameters=True
  trainer.devices="${NUM_GPUS}"
  trainer.num_nodes=1
  trainer.max_steps="${MAX_STEPS}"
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches=0
  trainer.log_every_n_steps="${LOG_EVERY}"
  optim.lr=3e-5
  optim.tran_head_lr=0.01
  optim.sm_prob=0.8
  optim.sm_t_min=0.2
  optim.sm_t_max=0.8
  lr_scheduler.num_warmup_steps=200
  sampling.predictor=sm
  eval.compute_generative_perplexity=False
  eval.generate_samples=False
  training.finetune_path="${BASE_CKPT}"
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CHECKPOINT_EVERY}"
  wandb.group="${GROUP}"
  wandb.name="${RUN_NAME}"
  ++hydra.run.dir="${RUN_OUT_DIR}"
  ++checkpointing.save_dir="${RUN_OUT_DIR}"
)

# Conditionally append fixed_lambda — only when >= 0.0 (matches Modal logic exactly)
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  CMD+=("algo.tran_head.fixed_lambda=${FIXED_LAMBDA}")
  echo "[train] Fixed lambda override: ${FIXED_LAMBDA}"
fi

"${CMD[@]}"

echo ""
echo "[done] Outputs saved to: $RUN_OUT_DIR"

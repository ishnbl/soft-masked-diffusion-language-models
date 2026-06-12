#!/usr/bin/env bash
# ============================================================
# run_verda_topk_lambda.sh — Top-K soft-masked MDLM finetune.
# Same as run_verda_topk.sh but with init_centre=-4.1 and the
# soft-mask band pinned to t∈[0.2, 0.8].
#
# Learned lambda (default):
#   bash run_verda_topk_lambda.sh
# Fixed lambda = 0:
#   bash run_verda_topk_lambda.sh --fixed-lambda 0
# ============================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────
NUM_GPUS=2
BASE_CKPT="/workspace/checkpoints/mdlm_owt.ckpt"
SEED=1
MAX_STEPS=5000
MODEL="small"
DATA="openwebtext-split"
GLOBAL_BATCH_SIZE=512
BATCH_SIZE=16
NUM_WORKERS=4
CHECKPOINT_EVERY=100
LOG_EVERY=3
DATA_DIR="/workspace/data"
OUT_DIR="/workspace/outputs"
FIXED_LAMBDA=-1.0          # -1.0 = learned lambda; set 0..1 to pin
RESUME=0                   # 1 = resume from <out>/checkpoints/last.ckpt

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)            RESUME=1;               shift   ;;
    --num-gpus)          NUM_GPUS="$2";          shift 2 ;;
    --base-ckpt)         BASE_CKPT="$2";         shift 2 ;;
    --seed)              SEED="$2";              shift 2 ;;
    --max-steps)         MAX_STEPS="$2";         shift 2 ;;
    --model)             MODEL="$2";             shift 2 ;;
    --data)              DATA="$2";              shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --batch-size)        BATCH_SIZE="$2";        shift 2 ;;
    --num-workers)       NUM_WORKERS="$2";       shift 2 ;;
    --checkpoint-every)  CHECKPOINT_EVERY="$2";  shift 2 ;;
    --log-every)         LOG_EVERY="$2";         shift 2 ;;
    --data-dir)          DATA_DIR="$2";          shift 2 ;;
    --out-dir)           OUT_DIR="$2";           shift 2 ;;
    --fixed-lambda)      FIXED_LAMBDA="$2";      shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Preflight checks ─────────────────────────────────────────
if [ ! -f "$BASE_CKPT" ]; then
  echo "ERROR: Checkpoint not found at $BASE_CKPT"; exit 1
fi
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA not available."; exit 1
fi
VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "ERROR: Requested $NUM_GPUS GPU(s) but only $VISIBLE_GPUS visible."; exit 1
fi
if [ $(( GLOBAL_BATCH_SIZE % (NUM_GPUS * BATCH_SIZE) )) -ne 0 ]; then
  echo "ERROR: global_batch_size must be divisible by num_gpus * batch_size."; exit 1
fi
ACCUM=$(( GLOBAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE) ))
if ! python -c "
v = float('${FIXED_LAMBDA}')
import sys
if v >= 0.0 and not 0.0 <= v <= 1.0:
    sys.stderr.write('ERROR: --fixed-lambda must lie in [0, 1]\n'); sys.exit(1)
"; then exit 1; fi

# ── Setup DUO files if missing ────────────────────────────────
if [ ! -f "models/dit.py" ]; then bash setup.sh; fi

# ── Directories ───────────────────────────────────────────────
mkdir -p "$DATA_DIR/owt_cache" "$OUT_DIR"
GROUP="topk_lambda_${NUM_GPUS}gpu_${MODEL}_seed${SEED}"
RUN_NAME="topk-lambda-${NUM_GPUS}gpu-${MODEL}-seed${SEED}"
RUN_OUT_DIR="${OUT_DIR}/${GROUP}"
mkdir -p "$RUN_OUT_DIR"

echo "========================================"
echo " Top-K Soft-Masked MDLM — Verda Run"
echo " GPUs: $NUM_GPUS | per-GPU batch: $BATCH_SIZE | accum: $ACCUM"
echo " Max steps: $MAX_STEPS | Checkpoint: $BASE_CKPT"
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  echo " Lambda: FIXED = $FIXED_LAMBDA"
else
  echo " Lambda: learned (entropy-gated)"
fi
echo " SM band: t in [0.2, 0.8]"
echo "========================================"

# ── Environment ───────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Launch training ───────────────────────────────────────────
CMD=(
  python -u -m main
  algo=mdlm_sm
  algo.tran_head.transparency_alg=mixinputs_with_topk
  algo.tran_head.mixinputs_k=3
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-4.1
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
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CHECKPOINT_EVERY}"
  wandb.group="${GROUP}"
  wandb.name="${RUN_NAME}"
  ++hydra.run.dir="${RUN_OUT_DIR}"
  ++checkpointing.save_dir="${RUN_OUT_DIR}"
)

# Pin lambda only when >= 0.0
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  CMD+=("algo.tran_head.fixed_lambda=${FIXED_LAMBDA}")
fi

# Resume from the rolling last.ckpt (full Lightning state: weights, optimizer,
# step count, LR schedule). Validation is disabled, so best.ckpt is never
# written — we resume from last.ckpt.
if [ "$RESUME" = "1" ]; then
  RESUME_CKPT="${RUN_OUT_DIR}/checkpoints/last.ckpt"
  if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: --resume set but no checkpoint at $RESUME_CKPT"; exit 1
  fi
  echo "[train] Resuming from ${RESUME_CKPT}"
  CMD+=("checkpointing.resume_from_ckpt=true"
        "checkpointing.resume_ckpt_path=${RESUME_CKPT}")
else
  CMD+=("checkpointing.resume_from_ckpt=false")
fi

"${CMD[@]}"
echo "[done] Outputs saved to: $RUN_OUT_DIR"

#!/usr/bin/env bash
# ============================================================
# run_scratch_slerp.sh — Train the SLERP (`slerp_sm`) soft-masking path
# from a FRESHLY INITIALIZED model (random weights), same architecture/data
# as the released mdlm_owt checkpoint (model=small, data=openwebtext-split).
#
# This is the from-scratch counterpart to ../../run_verda.sh: identical
# interface and Hydra overrides, MINUS `training.finetune_path` — no
# checkpoint is loaded, so the DiT backbone starts from random init and the
# transparency head starts at its configured init_scale/init_centre.
#
# COLD-START CAVEAT (read before chasing a "learned lambda" run):
#   experiments/SLERP_FINETUNING.md documents that from-scratch training hits
#   a cold-start trap: a random backbone produces near-uniform Pass-1 logits,
#   so the fed-back predictions are noise and the learned `scale` parameter
#   tends to get driven toward 0 (lambda -> ~0), collapsing slerp_sm to plain
#   MDLM. The init below (init_scale=0.5, init_centre=-2.5) is the validated
#   fix from scripts/07_slerp_vs_topk_fixed.sh: it keeps lambda near-zero
#   while the backbone is still uniform-ish and lets it grow as the backbone
#   becomes confident over training, rather than collapsing immediately.
#   --fixed-lambda sidesteps this entirely (lambda is pinned, not learned).
#
# Usage (from inside the language/ directory):
#   bash scripts/from_scratch/run_scratch_slerp.sh [options]
#
# Options (all optional — defaults shown):
#   --num-gpus              2
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
#   --val-check-interval    500
#   --warmup-steps          2500  (config default — a fresh backbone needs the
#                                  full warmup, unlike finetune's 200)
#   --sm-prob               0.8   (TARGET fraction of steps that run the
#                                  two-pass soft-mask forward once ramped up;
#                                  the rest are plain MDLM)
#   --sm-prob-warmup-steps  2000  (sm_prob ramps LINEARLY from 0 at step 0 up
#                                  to --sm-prob at this step, then holds
#                                  constant; set to 0 to disable the ramp and
#                                  use --sm-prob from step 0, as before)
#   --data-dir              /workspace/data
#   --out-dir               /workspace/outputs
#   --fixed-lambda          -1.0  (disabled by default -> learned entropy-gated
#                                  head; set to a value in [0,1] to pin lambda)
#
# Examples:
#   # Learnt lambda, from scratch, 1 GPU:
#   bash scripts/from_scratch/run_scratch_slerp.sh --num-gpus 1
#
#   # Fixed lambda = 0.5, from scratch, 1 GPU:
#   bash scripts/from_scratch/run_scratch_slerp.sh --num-gpus 1 --fixed-lambda 0.5
#
#   # Quick smoke test (2 GPUs, 20 steps):
#   bash scripts/from_scratch/run_scratch_slerp.sh --num-gpus 2 --max-steps 20 \
#     --global-batch-size 64 --batch-size 16
# ============================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────
NUM_GPUS=2
SEED=1
MAX_STEPS=-1
MODEL="small"
DATA="openwebtext-split"
SLERP_N_ITER=3
GLOBAL_BATCH_SIZE=512
BATCH_SIZE=16
NUM_WORKERS=4
CHECKPOINT_EVERY=500
LOG_EVERY=3
VAL_CHECK_INTERVAL=500
WARMUP_STEPS=2500
SM_PROB=0.8
SM_PROB_WARMUP_STEPS=2000
DATA_DIR="/workspace/data"
OUT_DIR="/workspace/outputs"
FIXED_LAMBDA=-1.0
INIT_CENTRE=-2.5

EXTRA_HYDRA_ARGS=()
# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus)            NUM_GPUS="$2";            shift 2 ;;
    --seed)                SEED="$2";                shift 2 ;;
    --max-steps)           MAX_STEPS="$2";            shift 2 ;;
    --model)               MODEL="$2";                shift 2 ;;
    --data)                DATA="$2";                 shift 2 ;;
    --slerp-n-iter)        SLERP_N_ITER="$2";         shift 2 ;;
    --global-batch-size)   GLOBAL_BATCH_SIZE="$2";    shift 2 ;;
    --batch-size)          BATCH_SIZE="$2";           shift 2 ;;
    --num-workers)         NUM_WORKERS="$2";          shift 2 ;;
    --checkpoint-every)    CHECKPOINT_EVERY="$2";     shift 2 ;;
    --log-every)           LOG_EVERY="$2";            shift 2 ;;
    --val-check-interval)  VAL_CHECK_INTERVAL="$2";   shift 2 ;;
    --warmup-steps)        WARMUP_STEPS="$2";         shift 2 ;;
    --sm-prob)             SM_PROB="$2";              shift 2 ;;
    --sm-prob-warmup-steps) SM_PROB_WARMUP_STEPS="$2"; shift 2 ;;
    --data-dir)            DATA_DIR="$2";             shift 2 ;;
    --out-dir)             OUT_DIR="$2";              shift 2 ;;
    --fixed-lambda)        FIXED_LAMBDA="$2";         shift 2 ;;
    --init-centre)         INIT_CENTRE="$2";          shift 2 ;;
    --resume-from-ckpt)    RESUME_FROM_CKPT="$2";     shift 2 ;;
    *) EXTRA_HYDRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${RESUME_FROM_CKPT:-}" ]]; then
  RESUME_FROM_CKPT="false"
fi

# ── Preflight checks ─────────────────────────────────────────
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

GROUP="scratch_slerp_${NUM_GPUS}gpu_${MODEL}_${DATA}_seed${SEED}_sm${SM_PROB}_lam${FIXED_LAMBDA}"
RUN_NAME="scratch-slerp-${NUM_GPUS}gpu-${MODEL}-${DATA}-seed${SEED}-sm${SM_PROB}-lam${FIXED_LAMBDA}"
RUN_OUT_DIR="${OUT_DIR}/${GROUP}"
mkdir -p "$RUN_OUT_DIR"

echo ""
echo "========================================"
echo " SLERP Soft-Masked MDLM — From Scratch"
echo "========================================"
echo " GPUs:          $NUM_GPUS"
echo " Per-GPU batch: $BATCH_SIZE"
echo " Accum steps:   $ACCUM"
echo " Global batch:  $GLOBAL_BATCH_SIZE"
echo " Max steps:     $MAX_STEPS"
if [ "$SM_PROB_WARMUP_STEPS" -gt 0 ] 2>/dev/null; then
  echo " sm_prob:       0 -> $SM_PROB linearly over $SM_PROB_WARMUP_STEPS steps"
else
  echo " sm_prob:       $SM_PROB (no ramp)"
fi
echo " Backbone init: random (no checkpoint loaded)"
echo " Output dir:    $RUN_OUT_DIR"
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  echo " Fixed lambda:  $FIXED_LAMBDA"
else
  echo " Lambda:        learned (entropy-gated; see cold-start caveat at top of this script)"
fi
echo "========================================"
echo ""

# ── Environment ───────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Launch training ───────────────────────────────────────────
# Note: NO training.finetune_path and NO checkpointing.resume_from_ckpt=false
# override that points at a base checkpoint — this intentionally starts from
# random init. resume_from_ckpt is still forced to false so a stale
# outputs/.../checkpoints/best.ckpt from a previous run in the same RUN_OUT_DIR
# can never silently resume instead of starting fresh.
CMD=(
  python -u -m main
  algo=mdlm_sm
  algo.tran_head.transparency_alg=slerp_sm
  algo.tran_head.slerp_n_iter="${SLERP_N_ITER}"
  algo.tran_head.mixinputs_k=3
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre="${INIT_CENTRE}"
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
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  trainer.log_every_n_steps="${LOG_EVERY}"
  optim.lr=3e-4
  optim.tran_head_lr=0.01
  optim.sm_prob="${SM_PROB}"
  optim.sm_prob_warmup_steps="${SM_PROB_WARMUP_STEPS}"
  optim.sm_t_min=0.2
  optim.sm_t_max=0.8
  lr_scheduler.num_warmup_steps="${WARMUP_STEPS}"
  sampling.predictor=sm
  eval.compute_generative_perplexity=False
  eval.generate_samples=False
  checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT}"
  checkpointing.resume_ckpt_path="${RUN_OUT_DIR}/checkpoints/last.ckpt"
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CHECKPOINT_EVERY}"
  wandb.group="${GROUP}"
  wandb.name="${RUN_NAME}"
  ${WANDB_PROJECT:+wandb.project="${WANDB_PROJECT}"}
  ++hydra.run.dir="${RUN_OUT_DIR}"
  ++checkpointing.save_dir="${RUN_OUT_DIR}"
)

# Conditionally append fixed_lambda — only when >= 0.0
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  CMD+=("algo.tran_head.fixed_lambda=${FIXED_LAMBDA}")
  echo "[train] Fixed lambda override: ${FIXED_LAMBDA}"
fi

"${CMD[@]}" "${EXTRA_HYDRA_ARGS[@]}"

echo ""
echo "[done] Outputs saved to: $RUN_OUT_DIR"

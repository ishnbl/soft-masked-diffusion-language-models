#!/bin/bash
#
# SLERP vs top-k LERP — finetuning from a pretrained base MDLM checkpoint.
#
# WHY FINETUNING INSTEAD OF SCRATCH
# ----------------------------------
# Training from scratch hits a cold-start trap: the random backbone produces
# near-uniform Pass-1 logits (neg_entropy ≈ −3.3), so soft-masked inputs are
# pure noise. The gradient correctly pushes scale DOWN from 0.5 to ~0.25,
# lambda stabilises at 0.01–0.04, and at that mixing weight the geometric
# difference between SLERP's great-circle path and LERP's straight-line blend
# is O(λ²) ≈ 0. Both methods reduce to vanilla MDLM.
#
# With a pretrained backbone, Pass-1 predictions are immediately confident.
# The gradient pushes scale UP (not down), lambda reaches 0.2–0.5 where
# the arc path and linear path produce visibly different embeddings, and
# SLERP/LERP val/bpd curves should separate within the first 500 steps.
#
# HOW finetune_path WORKS (main.py:162–177)
# ------------------------------------------
# model = MDLM_SM(config)          # fresh model, tran_head at __init__() values
# old   = torch.load(BASE_CKPT)    # base MDLM state_dict (no tran_head.* keys)
# model.load_state_dict(old, strict=False)
#   → backbone weights loaded from checkpoint ✓
#   → tran_head.* absent in ckpt  → retains init_scale / init_centre below ✓
#
# HYPERPARAMETER DIFFERENCES FROM SCRIPT 07 (scratch)
# -----------------------------------------------------
#   optim.lr              3e-4  → 3e-5   (10× lower to prevent catastrophic forgetting)
#   optim.tran_head_lr    0.01  → 0.01   (kept high; tran_head starts fresh)
#   lr_scheduler.num_warmup_steps  2500 → 200  (backbone already trained)
#   trainer.max_steps     5000  → 3000   (convergence faster with pretrained backbone)
#   trainer.val_check_interval 500 → 200 (catch early divergence)
#   optim.sm_prob          0.8  → 0.5    (gentler start; protect pretrained backbone)
#   algo.tran_head.init_scale  0.5 → 0.3 (σ'(logit(0.3))≈0.21; healthy gradient)
#   checkpointing.resume_from_ckpt true → false (prevent accidental old-ckpt resume)
#
# GPU REQUIREMENT
# ---------------
# Same as scripts 06/07: at least one visible GPU required.
#
# Usage:
#   BASE_CKPT=outputs/base_pretrain_tiny_text8_seed1/checkpoints/best.ckpt \
#   SEED=1 MAX_STEPS=3000 bash scripts/09_slerp_vs_topk_finetune.sh
#
# Knobs (env vars, with defaults):
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-3000}"
MODEL="${MODEL:-tiny}"
DATA="${DATA:-text8}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_CKPT="${BASE_CKPT:-}"

set -euo pipefail

# Preflight: require BASE_CKPT.
if [[ -z "$BASE_CKPT" ]]; then
  echo ""
  echo "[preflight] BASE_CKPT is not set."
  echo "Run script 08 first, then pass its output checkpoint, e.g.:"
  echo "  BASE_CKPT=outputs/base_pretrain_tiny_text8_seed1/checkpoints/best.ckpt \\"
  echo "  bash scripts/09_slerp_vs_topk_finetune.sh"
  exit 1
fi

if [[ ! -f "$BASE_CKPT" ]]; then
  echo ""
  echo "[preflight] BASE_CKPT file not found: $BASE_CKPT"
  exit 1
fi

# Preflight: require >=1 GPU.
python - <<'PY'
import sys, torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit(
        "\n[preflight] torch.cuda.device_count() == 0 — no GPU visible to PyTorch.\n"
        "This codebase requires at least one GPU (dataloader.py asserts\n"
        "global_batch_size == batch_size * num_nodes * device_count * accum).\n"
        "Attach a GPU and rerun.\n")
print(f"[preflight] {n} GPU(s) visible — OK")
PY

echo "[preflight] BASE_CKPT=${BASE_CKPT} — OK"

GROUP="slerp_vs_topk_finetune_${MODEL}_${DATA}_seed${SEED}"

COMMON=(
  algo=mdlm_sm
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  loader.batch_size="$BATCH_SIZE"
  loader.eval_batch_size="$BATCH_SIZE"
  trainer.max_steps="$MAX_STEPS"
  trainer.val_check_interval=200
  trainer.log_every_n_steps=50
  # Backbone LR 10× lower than scratch to prevent catastrophic forgetting.
  optim.lr=3e-5
  optim.tran_head_lr=0.01
  optim.sm_prob=0.5
  # Backbone already trained; short warmup is enough.
  lr_scheduler.num_warmup_steps=200
  sampling.predictor=sm
  strategy.find_unused_parameters=True
  algo.tran_head.mixinputs_k=3
  # Gentler init_scale than script 07 scratch (0.5→0.3); healthy gradient
  # without risking disruption to the pretrained backbone early on.
  algo.tran_head.init_scale=0.3
  algo.tran_head.init_centre=-2.5
  # Load pretrained backbone weights; tran_head keeps fresh init (strict=False).
  training.finetune_path="$BASE_CKPT"
  # Prevent accidentally resuming from a stale best.ckpt in the output dir.
  checkpointing.resume_from_ckpt=false
  wandb.group="$GROUP"
)

echo "=== [1/2] baseline: mixinputs_with_topk ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=mixinputs_with_topk \
  wandb.name="topk-ft-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/topk"

echo "=== [2/2] slerp: slerp_sm ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=slerp_sm \
  algo.tran_head.slerp_n_iter=3 \
  wandb.name="slerp-ft-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/slerp"

# To run both in parallel on two GPUs:
#   CUDA_VISIBLE_DEVICES=0 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=mixinputs_with_topk \
#     wandb.name="topk-ft-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/topk" &
#   CUDA_VISIBLE_DEVICES=1 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=slerp_sm algo.tran_head.slerp_n_iter=3 \
#     wandb.name="slerp-ft-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/slerp" &
#   wait

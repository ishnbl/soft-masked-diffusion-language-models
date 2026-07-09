#!/bin/bash
#
# SLERP vs top-k LERP — freeze-first recovery variant.
#
# WHEN TO USE THIS INSTEAD OF 10_slerp_vs_topk_finetune_v2.sh
# ----------------------------------------------------------
# Run this if the primary v2 recipe shows lambda collapsing toward 0 on the
# `transparency/lambda_mean` (or `transparency/lambda_std` going flat) — the
# symptom that the moving backbone is giving the head noisy gradients, and the
# head responds by suppressing itself.
#
# WHAT IT DOES DIFFERENTLY
# ------------------------
# Adds `+callbacks/freeze_backbone=freeze_backbone` to the Hydra command:
#   * FreezeBackboneCallback sets `requires_grad=False` on every parameter NOT
#     under `tran_head.*` for the first FREEZE_UNTIL steps (default 1000).
#   * Autograd still propagates JVPs through the frozen backbone, so the
#     transparency head receives full gradients against a stationary backbone.
#   * At step FREEZE_UNTIL the callback unfreezes the backbone and joint
#     finetuning begins. The optimizer already had backbone params in its
#     groups (with zero grads during the freeze), so AdamW just starts using
#     non-zero grads — no reconstruction needed.
#
# Everything else is identical to 10_v2 (lr=3e-5, init_scale=0.5,
# val_check=200, lambda_std logged, same seed → same data order).
#
# Usage:
#   BASE_CKPT=/path/to/mdlm.ckpt SEED=1 MAX_STEPS=5000 FREEZE_UNTIL=1000 \
#     bash scripts/10b_slerp_vs_topk_finetune_freeze.sh

SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_CKPT="${BASE_CKPT:-}"
FREEZE_UNTIL="${FREEZE_UNTIL:-1000}"

set -euo pipefail

if [[ -z "$BASE_CKPT" ]]; then
  echo ""
  echo "[preflight] BASE_CKPT is not set."
  echo "  BASE_CKPT=/path/to/mdlm.ckpt bash scripts/10b_slerp_vs_topk_finetune_freeze.sh"
  exit 1
fi

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[preflight] BASE_CKPT file not found: $BASE_CKPT"
  exit 1
fi

python - <<'PY'
import sys, torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit("\n[preflight] no GPU visible to PyTorch. Attach a GPU and rerun.\n")
print(f"[preflight] {n} GPU(s) visible — OK")
PY

echo "[preflight] BASE_CKPT=${BASE_CKPT} — OK"
echo "[preflight] FREEZE_UNTIL=${FREEZE_UNTIL} steps"

GROUP="slerp_vs_topk_finetune_freeze_${MODEL}_${DATA}_seed${SEED}"

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
  optim.lr=3e-5
  optim.tran_head_lr=0.01
  optim.sm_prob=0.8
  lr_scheduler.num_warmup_steps=200
  sampling.predictor=sm
  strategy.find_unused_parameters=True
  eval.compute_generative_perplexity=False
  algo.tran_head.mixinputs_k=3
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-2.5
  training.finetune_path="$BASE_CKPT"
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100
  # Append the freeze-backbone callback to the default callback group.
  +callbacks/freeze_backbone=freeze_backbone
  callbacks.freeze_backbone.freeze_until_step="$FREEZE_UNTIL"
  wandb.group="$GROUP"
)

echo "=== [1/2] baseline: mixinputs_with_topk (backbone frozen first ${FREEZE_UNTIL} steps) ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=mixinputs_with_topk \
  wandb.name="topk-ft-freeze${FREEZE_UNTIL}-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/topk"

echo "=== [2/2] slerp: slerp_sm (backbone frozen first ${FREEZE_UNTIL} steps) ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=slerp_sm \
  algo.tran_head.slerp_n_iter=3 \
  wandb.name="slerp-ft-freeze${FREEZE_UNTIL}-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/slerp"

#!/bin/bash
#
# SLERP vs top-k LERP — v2 finetuning recipe on the released MDLM (OpenWebText).
#
# WHAT CHANGED VS SCRIPT 09
# -------------------------
# Same A/B as 09, but tightened for a clean 5k-step finetune on a pretrained
# backbone:
#   * trainer.val_check_interval=200          (was 500) — catch divergence sooner.
#   * optim.lr=3e-5                            (config default 3e-4) — backbone is
#       already trained; a 10x lower LR avoids catastrophic forgetting that
#       would otherwise swamp the SLERP/LERP signal.
#   * algo.tran_head.init_scale=0.5            (was 0.3) — push the realised
#       lambda toward ~0.5 at confident logits. At neg_entropy ≈ 0 with
#       centre=-2.5 and steep=6.66, lambda_max ≈ 0.5 · σ(16.65) ≈ 0.5.
#       dσ/draw_scale = 0.25 at scale=0.5, so the raw_scale gradient is still
#       healthy (vs 0.21 at 0.3, but a higher starting lambda → larger
#       SLERP/LERP geometric divergence from step 0).
#   * transparency/lambda_std now logged each step (transparency_head.py +
#       algo.py) alongside lambda_mean. If lambda_mean drifts to 0 OR
#       lambda_std collapses (head is becoming a constant), fall back to
#       10b_slerp_vs_topk_finetune_freeze.sh, which freezes the backbone for
#       the first 1000 steps so the head settles on a stationary signal.
#
# EVERYTHING ELSE MATCHES 09: model=small, data=openwebtext-split, sm_prob=0.8,
# tran_head_lr=0.01, mixinputs_k=3, finetune_path=$BASE_CKPT,
# resume_from_ckpt=false, find_unused_parameters=True, sampling.predictor=sm.
#
# Usage:
#   BASE_CKPT=/path/to/mdlm.ckpt SEED=1 MAX_STEPS=5000 \
#     bash scripts/10_slerp_vs_topk_finetune_v2.sh

SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_CKPT="${BASE_CKPT:-}"

set -euo pipefail

if [[ -z "$BASE_CKPT" ]]; then
  echo ""
  echo "[preflight] BASE_CKPT is not set."
  echo "Download the released binary MDLM (OpenWebText) from the Google Drive"
  echo "folder linked in language/README.md, then pass its path, e.g.:"
  echo "  BASE_CKPT=/path/to/mdlm.ckpt bash scripts/10_slerp_vs_topk_finetune_v2.sh"
  exit 1
fi

if [[ ! -f "$BASE_CKPT" ]]; then
  echo ""
  echo "[preflight] BASE_CKPT file not found: $BASE_CKPT"
  exit 1
fi

python - <<'PY'
import sys, torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit(
        "\n[preflight] torch.cuda.device_count() == 0 — no GPU visible to PyTorch.\n"
        "Attach a GPU and rerun.\n")
print(f"[preflight] {n} GPU(s) visible — OK")
PY

echo "[preflight] BASE_CKPT=${BASE_CKPT} — OK"

GROUP="slerp_vs_topk_finetune_v2_${MODEL}_${DATA}_seed${SEED}"

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
  # --- lambda activation, v2: push lambda_max toward ~0.5 at confident logits.
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-3.5
  training.finetune_path="$BASE_CKPT"
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100
  wandb.group="$GROUP"
)

echo "=== [1/2] baseline: mixinputs_with_topk ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=mixinputs_with_topk \
  wandb.name="topk-ft-v2-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/topk"

echo "=== [2/2] slerp: slerp_sm ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=slerp_sm \
  algo.tran_head.slerp_n_iter=3 \
  wandb.name="slerp-ft-v2-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/slerp"

# Parallel on two GPUs:
#   CUDA_VISIBLE_DEVICES=0 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=mixinputs_with_topk \
#     wandb.name="topk-ft-v2-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/topk" &
#   CUDA_VISIBLE_DEVICES=1 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=slerp_sm algo.tran_head.slerp_n_iter=3 \
#     wandb.name="slerp-ft-v2-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/slerp" &
#   wait

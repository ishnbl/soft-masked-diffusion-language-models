#!/bin/bash
#
# Base MDLM pretraining — produces a shared checkpoint for experiment 09.
#
# Train a plain MDLM (no transparency head) from scratch on text8/tiny.
# The resulting checkpoint is then consumed by 09_slerp_vs_topk_finetune.sh
# which loads it into an MDLM_SM model via training.finetune_path + strict=False.
# Because the base MDLM state_dict contains no tran_head.* keys, the
# transparency head naturally retains its fresh __init__() values — no
# manual surgery required.
#
# Usage:
#   SEED=1 MAX_STEPS=3000 bash scripts/08_base_pretrain.sh
#
# Knobs (env vars, with defaults):
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-3000}"
MODEL="${MODEL:-tiny}"
DATA="${DATA:-text8}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-64}"

set -euo pipefail

# Preflight: this codebase needs >=1 GPU.
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

RUN_NAME="base-pretrain-${MODEL}-${DATA}-seed${SEED}"
RUN_DIR="outputs/base_pretrain_${MODEL}_${DATA}_seed${SEED}"

echo "=== Base MDLM pretraining: ${RUN_NAME} ==="
python -u -m main \
  algo=mdlm \
  model="$MODEL" \
  data="$DATA" \
  data.cache_dir="$DATA_CACHE_DIR" \
  seed="$SEED" \
  loader.batch_size="$BATCH_SIZE" \
  loader.eval_batch_size="$BATCH_SIZE" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=50 \
  lr_scheduler.num_warmup_steps=500 \
  wandb.name="$RUN_NAME" \
  ++hydra.run.dir="$RUN_DIR"

echo ""
echo "=== Checkpoint saved to: ${RUN_DIR}/checkpoints/best.ckpt ==="
echo "Pass it to experiment 09 via:"
echo "  BASE_CKPT=${RUN_DIR}/checkpoints/best.ckpt bash scripts/09_slerp_vs_topk_finetune.sh"

#!/bin/bash
#
# SLERP vs top-k soft-masking convergence A/B (language MDLM, from scratch).
#
# Launches two otherwise-identical runs that differ ONLY in transparency_alg:
#   - baseline : mixinputs_with_topk (convex top-k mixing in vocab space)
#   - slerp    : slerp_sm            (spherical interp + Frechet mean in embed space)
# Same seed => same init + data order, so the loss curves are directly comparable.
# Both share one wandb.group so you can overlay them; watch transparency/* live.
#
# Usage:
#   SEED=1 MAX_STEPS=30000 bash scripts/06_slerp_vs_topk_ablation.sh
#
# Knobs (env vars, with defaults):
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
MODEL="${MODEL:-tiny}"            # tiny (~14M) for fast feedback; small/medium for capacity
DATA="${DATA:-text8}"            # small, self-contained dataset
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-64}"

set -euo pipefail

GROUP="slerp_vs_topk_${MODEL}_${DATA}_seed${SEED}"

# Overrides shared by both variants.
COMMON=(
  algo=mdlm_sm
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  loader.batch_size="$BATCH_SIZE"
  loader.eval_batch_size="$BATCH_SIZE"
  trainer.max_steps="$MAX_STEPS"
  trainer.val_check_interval=1000
  trainer.log_every_n_steps=50
  optim.sm_prob=0.8
  optim.tran_head_lr=0.01
  sampling.predictor=sm
  strategy.find_unused_parameters=True
  algo.tran_head.mixinputs_k=3
  wandb.group="$GROUP"
)

echo "=== [1/2] baseline: mixinputs_with_topk ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=mixinputs_with_topk \
  wandb.name="topk-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/topk"

echo "=== [2/2] slerp: slerp_sm ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=slerp_sm \
  algo.tran_head.slerp_n_iter=3 \
  wandb.name="slerp-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/slerp"

# To run both in parallel on two GPUs instead of sequentially, replace the two
# blocks above with (note the trailing & and final `wait`):
#   CUDA_VISIBLE_DEVICES=0 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=mixinputs_with_topk \
#     wandb.name="topk-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/topk" &
#   CUDA_VISIBLE_DEVICES=1 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=slerp_sm algo.tran_head.slerp_n_iter=3 \
#     wandb.name="slerp-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/slerp" &
#   wait

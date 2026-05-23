#!/bin/bash
#
# SLERP vs top-k ablation — fixed transparency-head initialisation.
#
# The original script (06_slerp_vs_topk_ablation.sh) produced identical metrics
# for both runs because three compounding factors drove lambda to ~1e-14:
#
#   1. init_scale=0.0  →  raw_scale = logit(1e-6) = -13.82
#                         scale = σ(-13.82) ≈ 1e-6
#                         σ'(raw_scale) ≈ 1e-6  (kills gradient on raw_scale)
#
#   2. init_centre=-0.75  →  requires neg_entropy > -0.75 to activate,
#                            i.e. backbone must put >85% prob on one token.
#                            At init, uniform backbone gives neg_entropy ≈ -3.3
#                            (text8, V=27), so sigmoid factor ≈ 4e-8.
#                            Combined: λ = 1e-6 × 4e-8 = 4e-14  (W&B confirmed).
#
#   3. LR warmup: effective tran_head_lr at step 150 was ~7e-4, not the 0.01 target.
#
# Fixes applied here:
#   init_scale=0.5   →  scale = 0.5 from step 0; σ'(0) = 0.25 (healthy gradient)
#   init_centre=-2.5 →  activates when neg_entropy > -2.5, i.e. ~30% top-1 confidence
#                        λ at uniform backbone: 0.5 × σ(6.66×(-3.3+2.5)) ≈ 0.0024
#                        λ at moderate confidence (neg_entropy=-1.0): ≈ 0.5
#
# Usage:
#   SEED=1 MAX_STEPS=5000 bash scripts/07_slerp_vs_topk_fixed.sh
#
# Knobs (env vars, with defaults):
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-tiny}"
DATA="${DATA:-text8}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-64}"

set -euo pipefail

GROUP="slerp_vs_topk_fixed_${MODEL}_${DATA}_seed${SEED}"

COMMON=(
  algo=mdlm_sm
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  loader.batch_size="$BATCH_SIZE"
  loader.eval_batch_size="$BATCH_SIZE"
  trainer.devices=1
  trainer.max_steps="$MAX_STEPS"
  trainer.val_check_interval=500
  trainer.log_every_n_steps=50
  optim.sm_prob=0.8
  optim.tran_head_lr=0.01
  sampling.predictor=sm
  strategy.find_unused_parameters=True
  algo.tran_head.mixinputs_k=3
  # Fixed initialisations
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-2.5
  wandb.group="$GROUP"
)

echo "=== [1/2] baseline: mixinputs_with_topk ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=mixinputs_with_topk \
  wandb.name="topk-fixed-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/topk"

echo "=== [2/2] slerp: slerp_sm ==="
python -u -m main "${COMMON[@]}" \
  algo.tran_head.transparency_alg=slerp_sm \
  algo.tran_head.slerp_n_iter=3 \
  wandb.name="slerp-fixed-${MODEL}-${DATA}-seed${SEED}" \
  ++hydra.run.dir="outputs/${GROUP}/slerp"

# To run both in parallel on two GPUs:
#   CUDA_VISIBLE_DEVICES=0 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=mixinputs_with_topk \
#     wandb.name="topk-fixed-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/topk" &
#   CUDA_VISIBLE_DEVICES=1 python -u -m main "${COMMON[@]}" \
#     algo.tran_head.transparency_alg=slerp_sm algo.tran_head.slerp_n_iter=3 \
#     wandb.name="slerp-fixed-${MODEL}-${DATA}-seed${SEED}" \
#     ++hydra.run.dir="outputs/${GROUP}/slerp" &
#   wait

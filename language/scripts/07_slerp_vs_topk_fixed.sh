#!/bin/bash
#
# SLERP vs top-k ablation — fixed transparency-head initialisation.
#
# WHAT THIS FIXES
# ---------------
# The original script (06_slerp_vs_topk_ablation.sh) produced identical metrics
# for both runs because three compounding factors drove lambda to ~1e-14, so the
# transparency head was effectively inert and both runs reduced to plain MDLM:
#
#   1. init_scale=0.0  →  raw_scale = logit(1e-6) = -13.82
#                         scale = σ(-13.82) ≈ 1e-6
#                         σ'(raw_scale) ≈ 1e-6  (kills the gradient on raw_scale)
#
#   2. init_centre=-0.75  →  requires neg_entropy > -0.75 to activate,
#                            i.e. backbone must put >85% prob on one token.
#                            At init, a uniform backbone gives neg_entropy ≈ -3.3
#                            (text8, V=27), so the sigmoid factor ≈ 4e-8.
#                            Combined: λ = 1e-6 × 4e-8 = 4e-14  (W&B confirmed).
#
#   3. LR warmup: effective tran_head_lr early on is ~7e-4, not the 0.01 target.
#
# Fixes applied here (the ONLY differences from script 06):
#   init_scale=0.5   →  scale = 0.5 from step 0; σ'(0) = 0.25 (healthy gradient)
#   init_centre=-2.5 →  activates when neg_entropy > -2.5, i.e. ~30% top-1 conf.
#                        λ at uniform backbone: 0.5 × σ(6.66×(-3.3+2.5)) ≈ 0.0024
#                        λ at moderate confidence (neg_entropy=-1.0): ≈ 0.5
#   val_check_interval=500 → see the curves diverge before step 1000.
#
# GPU REQUIREMENT
# ---------------
# The duo dataloader hard-asserts (dataloader.py:676):
#     global_batch_size == batch_size * num_nodes * torch.cuda.device_count()
#                          * accumulate_grad_batches
# and trainer.devices resolves to ${device_count:} == torch.cuda.device_count()
# (registered in main.py:15). Both the config formulas AND the assertion read the
# REAL hardware count, so this codebase REQUIRES at least one visible GPU. If
# torch.cuda.device_count() == 0 you get a ZeroDivisionError (resolving
# accumulate_grad_batches) or an AssertionError at dataloader.py:676 — that is an
# environment problem (no GPU attached), not a config problem. The preflight
# check below fails fast with a clear message in that case.
#
# Usage:
#   SEED=1 MAX_STEPS=5000 bash scripts/07_slerp_vs_topk_fixed.sh
#
# Knobs (env vars, with defaults):
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-tiny}"            # tiny (~14M) for fast feedback; small/medium for capacity
DATA="${DATA:-text8}"            # small, self-contained dataset
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-64}"   # per-GPU micro-batch

set -euo pipefail

# Preflight: this codebase needs >=1 GPU (see GPU REQUIREMENT above).
python - <<'PY'
import sys, torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit(
        "\n[preflight] torch.cuda.device_count() == 0 — no GPU visible to PyTorch.\n"
        "This codebase requires at least one GPU (dataloader.py asserts\n"
        "global_batch_size == batch_size * num_nodes * device_count * accum).\n"
        "Attach a GPU to this session (or fix CUDA_VISIBLE_DEVICES / the torch\n"
        "install) and rerun. This is an environment issue, not a config one.\n")
print(f"[preflight] {n} GPU(s) visible — OK")
PY

GROUP="slerp_vs_topk_fixed_${MODEL}_${DATA}_seed${SEED}"

# Overrides shared by both variants. Batch/device config is left exactly as in
# script 06: trainer.devices auto-resolves to device_count(), so the config
# formulas and the dataloader assertion use the SAME GPU count and stay
# consistent on any 1+ GPU machine (effective batch = global_batch_size = 512).
COMMON=(
  algo=mdlm_sm
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  loader.batch_size="$BATCH_SIZE"
  loader.eval_batch_size="$BATCH_SIZE"
  trainer.max_steps="$MAX_STEPS"
  trainer.val_check_interval=500
  trainer.log_every_n_steps=50
  optim.sm_prob=0.8
  optim.tran_head_lr=0.01
  sampling.predictor=sm
  strategy.find_unused_parameters=True
  algo.tran_head.mixinputs_k=3
  # --- the actual fix: transparency-head initialisation ---
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

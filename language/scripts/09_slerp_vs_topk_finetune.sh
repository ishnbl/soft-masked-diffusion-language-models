#!/bin/bash
#
# SLERP vs top-k LERP — finetuning from the RELEASED base MDLM (OpenWebText).
#
# WHY FINETUNING FROM A PRETRAINED BACKBONE
# ------------------------------------------
# Training from scratch hits a cold-start trap: the random backbone produces
# near-uniform Pass-1 logits (neg_entropy ≈ −3.3), so soft-masked inputs are
# pure noise. The gradient correctly pushes scale DOWN from 0.5 to ~0.25,
# lambda stabilises at 0.01–0.04, and at that mixing weight the geometric
# difference between SLERP's great-circle path and LERP's straight-line blend
# is O(λ²) ≈ 0. Both methods reduce to vanilla MDLM and the metrics overlap.
#
# Starting from a strong pretrained backbone removes the cold start: Pass-1
# predictions are immediately confident (neg_entropy near 0), so with the
# init_centre/init_scale below lambda activates at ≈0.3 from step 0 and the
# gradient can push scale UP. At lambda 0.2–0.5 the arc path and the linear
# path produce visibly different embeddings, and SLERP/LERP val/bpd curves
# should separate.
#
# WHICH CHECKPOINT
# ----------------
# Use the released binary MDLM (OpenWebText) referenced in language/README.md:
#   https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau
# Download it, then point BASE_CKPT at the .ckpt file. This is the SAME
# checkpoint that the repo's own continuation recipe (01_sm_pretraining_cont_owt.sh)
# loads. It is a GPT-2-BPE small DiT, so this experiment runs on
# model=small + data=openwebtext-split (NOT the tiny/text8 setup of 06/07).
#
# HOW finetune_path WORKS (main.py:162–177)
# ------------------------------------------
# model = MDLM_SM(config)          # fresh model, tran_head at __init__() values
# old   = torch.load(BASE_CKPT)    # base MDLM state_dict (no tran_head.* keys)
# model.load_state_dict(old, strict=False)
#   → backbone weights loaded from checkpoint ✓
#   → tran_head.* absent in ckpt  → retains init_scale / init_centre below ✓
#
# RECIPE
# ------
# Mirrors 01_sm_pretraining_cont_owt.sh (the repo's validated continuation
# recipe: model=small, data=openwebtext-split, sm_prob=0.8, tran_head_lr=0.01,
# find_unused_parameters=True, sampling.predictor=sm) and layers on:
#   - the A/B over transparency_alg (top-k LERP vs slerp_sm)
#   - the lambda-activation fix that the from-scratch runs needed:
#       algo.tran_head.init_scale=0.3   (σ'(logit(0.3))≈0.21; healthy gradient,
#                                        vs config default 0.0 → scale≈1e-6)
#       algo.tran_head.init_centre=-2.5 (activates at ~30% top-1 conf, vs the
#                                        config default -0.75 which needs >85%)
#   - checkpointing.resume_from_ckpt=false so a stale best.ckpt in the run dir
#     cannot silently override training.finetune_path
#   - a short warmup (backbone is already trained)
# Same seed for both runs ⇒ identical init + data order ⇒ directly comparable.
#
# GPU REQUIREMENT
# ---------------
# Same as scripts 06/07: at least one visible GPU required (dataloader.py asserts
# global_batch_size == batch_size * num_nodes * device_count * accum).
#
# Usage:
#   BASE_CKPT=/path/to/mdlm.ckpt SEED=1 MAX_STEPS=5000 \
#     bash scripts/09_slerp_vs_topk_finetune.sh
#
# Knobs (env vars, with defaults):
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"                  # small matches the released OWT backbone
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
BATCH_SIZE="${BATCH_SIZE:-32}"           # per-GPU micro-batch (matches script 01)
BASE_CKPT="${BASE_CKPT:-}"

set -euo pipefail

# Preflight: require BASE_CKPT (the released binary MDLM checkpoint).
if [[ -z "$BASE_CKPT" ]]; then
  echo ""
  echo "[preflight] BASE_CKPT is not set."
  echo "Download the released binary MDLM (OpenWebText) from the Google Drive"
  echo "folder linked in language/README.md, then pass its path, e.g.:"
  echo "  BASE_CKPT=/path/to/mdlm.ckpt bash scripts/09_slerp_vs_topk_finetune.sh"
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
  trainer.val_check_interval=500
  trainer.log_every_n_steps=50
  optim.tran_head_lr=0.01
  optim.sm_prob=0.8
  # Backbone already trained; minimal warmup needed.
  lr_scheduler.num_warmup_steps=200
  sampling.predictor=sm
  strategy.find_unused_parameters=True
  # OWT generative PPL eval is expensive and irrelevant to this A/B (matches 01).
  eval.compute_generative_perplexity=False
  algo.tran_head.mixinputs_k=3
  # --- lambda-activation fix (same rationale as scripts 07): without this the
  # config defaults (init_scale=0.0 → scale≈1e-6, init_centre=-0.75) drive
  # lambda to ~0 and both methods collapse to vanilla MDLM. ---
  algo.tran_head.init_scale=0.3
  algo.tran_head.init_centre=-2.5
  # Load pretrained backbone weights; tran_head keeps fresh init (strict=False).
  training.finetune_path="$BASE_CKPT"
  # Prevent a stale best.ckpt in the run dir from overriding finetune_path.
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100
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

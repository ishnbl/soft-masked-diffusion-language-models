#!/bin/bash
# ===========================================================================
# slurm_scratch_train.sbatch — submit a single-node multi-GPU from-scratch
# slerp/topk soft-masked MDLM training run (run_scratch_slerp.sh /
# run_scratch_topk.sh — NO checkpoint loaded, random backbone init).
#
# Same shape as ../../verda/slurm_train.sbatch, adapted for the from-scratch
# launchers: no BASE_CKPT (there's nothing to load), and FIXED_LAMBDA /
# SM_PROB are exposed as knobs since both launchers accept them directly.
#
# Targets ONE node with N GPUs (single-node DDP). Lightning spawns one
# process per GPU itself, so we use --ntasks=1 and call plain `bash` (via the
# run_scratch_*.sh launcher); do NOT wrap it in srun with
# ntasks-per-node>1 or Lightning's SLURM integration will fight the launcher
# over the world size.
#
# EDIT THE #SBATCH LINES marked TODO for your cluster, then submit:
#   sbatch scripts/from_scratch/slurm_scratch_train.sbatch
#     # slerp, learnt lambda (defaults)
#   sbatch --export=ALL,ALG=topk scripts/from_scratch/slurm_scratch_train.sbatch
#     # top-k, learnt lambda
#   sbatch --export=ALL,ALG=slerp,FIXED_LAMBDA=0.5 scripts/from_scratch/slurm_scratch_train.sbatch
#     # slerp, fixed lambda = 0.5
#   sbatch --export=ALL,ALG=topk,FIXED_LAMBDA=0.5,SEED=2,SM_PROB=0.5 \
#     scripts/from_scratch/slurm_scratch_train.sbatch
#     # top-k, fixed lambda = 0.5, seed 2, sm_prob 0.5
#
# Change the GPU count by editing --gres=gpu:N below. NUM_GPUS is derived
# from SLURM's CUDA_VISIBLE_DEVICES at runtime — do NOT also set NUM_GPUS.
#
# DATA CACHING: see the "Data / tokenization caching" section in
# scripts/from_scratch/README.md before submitting several of these jobs at
# once — the first job to touch an empty DATA_DIR downloads + tokenizes all
# of OpenWebText (large, slow), and that work is NOT sharded across jobs, so
# running two cold jobs against the same empty DATA_DIR at the same time
# wastes work and can race on the same cache files. Warm the cache with one
# short run first (--max-steps 1), then submit the rest.
# ===========================================================================
#SBATCH --job-name=scratch-sm-mdlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                 # TODO: GPUs per node (also the DDP world size)
#SBATCH --ntasks=1                   # one task; Lightning spawns the per-GPU procs
#SBATCH --cpus-per-task=16           # TODO: ~4 dataloader workers/GPU
#SBATCH --mem=128G                   # TODO: cluster-dependent
#SBATCH --time=07:00:00              # TODO: walltime
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<PARTITION>     # TODO: uncomment + set
##SBATCH --account=<ACCOUNT>         # TODO: uncomment + set if your cluster needs it

set -euo pipefail

# ── User config (override via --export=ALL,VAR=...) ────────────────────────
ALG="${ALG:-slerp}"                                  # slerp | topk
FIXED_LAMBDA="${FIXED_LAMBDA:-}"                     # empty -> learnt lambda; else in [0,1]
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
SM_PROB="${SM_PROB:-0.8}"
MODEL="${MODEL:-small}"
DATA="${DATA:-openwebtext-split}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"                       # TODO: keep divisible into GLOBAL_BATCH_SIZE by NUM_GPUS*BATCH_SIZE
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

# Put these on SHARED, persistent storage (visible from the compute node).
# DATA_DIR is the parent of owt_cache/ (the launchers append that themselves).
export DATA_DIR="${DATA_DIR:-$HOME/sm_data}"
export OUT_DIR="${OUT_DIR:-$HOME/sm_outputs}"

# Compute nodes are frequently air-gapped. Default W&B to offline; sync later
# from the login node with `wandb sync $OUT_DIR/.../wandb/offline-run-*`.
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── Activate the env built by verda/setup_cluster.sh ───────────────────────
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "[job] CONDA_HOME=$CONDA_HOME not found — run verda/setup_cluster.sh first."; exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Run from language/ (this script lives in language/scripts/from_scratch/) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$LANG_DIR"

echo "[job] node=$(hostname)  ALG=${ALG}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ── Derive NUM_GPUS from what SLURM actually gave this job ─────────────────
# run_scratch_{slerp,topk}.sh take an explicit --num-gpus (they don't
# auto-detect), so compute it the same way run_slerp_multigpu.sh's preflight
# does: honor CUDA_VISIBLE_DEVICES if SLURM set it (via --gres), else fall
# back to torch's device count.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
else
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
echo "[job] NUM_GPUS=${NUM_GPUS}"

case "$ALG" in
  slerp) LAUNCHER="scripts/from_scratch/run_scratch_slerp.sh" ;;
  topk)  LAUNCHER="scripts/from_scratch/run_scratch_topk.sh" ;;
  *) echo "[job] ALG must be 'slerp' or 'topk', got '${ALG}'"; exit 1 ;;
esac

ARGS=(
  --num-gpus "$NUM_GPUS"
  --seed "$SEED"
  --max-steps "$MAX_STEPS"
  --sm-prob "$SM_PROB"
  --model "$MODEL"
  --data "$DATA"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --batch-size "$BATCH_SIZE"
  --data-dir "$DATA_DIR"
  --out-dir "$OUT_DIR"
)
if [[ -n "$FIXED_LAMBDA" ]]; then
  ARGS+=(--fixed-lambda "$FIXED_LAMBDA")
fi

echo "[job] launching ${LAUNCHER} ${ARGS[*]}"
bash "$LAUNCHER" "${ARGS[@]}"

echo "[job] done. Outputs under ${OUT_DIR}"
[[ "$WANDB_MODE" == "offline" ]] && \
  echo "[job] W&B was offline — sync later:  wandb sync ${OUT_DIR}/*/wandb/offline-run-*"

#!/bin/bash
# ===========================================================================
# slurm_lambda_zero.sbatch — submit MDLM SM with fixed lambda=0 on a node.
# Only saves the last checkpoint (last.ckpt) to conserve disk space.
# ===========================================================================
#SBATCH --job-name=lambda-zero
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                 # Number of GPUs to use (also the DDP world size)
#SBATCH --ntasks=1                   # one task; Lightning spawns the per-GPU procs
#SBATCH --cpus-per-task=16           # ~8 dataloader workers/GPU
#SBATCH --mem=128G
#SBATCH --time=168:00:00             # 7 days walltime
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# ── User config (override via --export=ALL,VAR=...) ────────────────────────
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-/opt/miniforge3}"

# Persistent storage visible from the compute node.
BASE_CKPT="${BASE_CKPT:-$HOME/mdlm/soft-masked-diffusion-language-models/mdlm.ckpt}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/sm_data/owt_cache}"
export OUT_ROOT="${OUT_ROOT:-$HOME/sm_outputs}"

# Run hyperparameters
export SEED="${SEED:-3}"
export MAX_STEPS="${MAX_STEPS:--1}"           # -1 = unlimited
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Activate env ───────────────────────────────────────────────────────────
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "[job] CONDA_HOME=$CONDA_HOME not found — check miniconda installation."; exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Run from directory where sbatch command was run ────────────────────────
cd "$SLURM_SUBMIT_DIR"

echo "[job] node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "[job] launching verda/run_lambda_zero.sh"
BASE_CKPT="$BASE_CKPT" bash verda/run_lambda_zero.sh

echo "[job] done. Outputs under ${OUT_ROOT}"

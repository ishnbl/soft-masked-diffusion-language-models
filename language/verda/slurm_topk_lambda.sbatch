#!/bin/bash
# ===========================================================================
# slurm_topk_lambda.sbatch — single-node multi-GPU Top-K soft-masked MDLM
# finetune via run_verda_topk_lambda.sh (init_centre=-4.1, SM band t∈[0.2,0.8]).
#
# Single-node DDP: Lightning spawns one process per GPU, so --ntasks=1 and we
# call plain `bash run_verda_topk_lambda.sh` (do NOT wrap in srun with
# ntasks-per-node>1 or Lightning's SLURM integration fights the launcher).
#
# Submit:
#   sbatch verda/slurm_topk_lambda.sbatch                              # learned lambda
#   sbatch --export=ALL,FIXED_LAMBDA=0 verda/slurm_topk_lambda.sbatch  # fixed lambda=0
#   sbatch --export=ALL,FIXED_LAMBDA=0,SEED=2 verda/slurm_topk_lambda.sbatch
#
# Keep --gres=gpu:N and NUM_GPUS below in sync (N = DDP world size).
# ===========================================================================
#SBATCH --job-name=topk-lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                 # TODO: GPUs per node (= NUM_GPUS below)
#SBATCH --ntasks=1                   # one task; Lightning spawns per-GPU procs
#SBATCH --cpus-per-task=16           # ~4 dataloader workers/GPU
#SBATCH --mem=128G                   # TODO: cluster-dependent
#SBATCH --time=07:00:00              # TODO: walltime
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<PARTITION>     # TODO: uncomment + set
##SBATCH --account=<ACCOUNT>         # TODO: uncomment + set if needed

set -euo pipefail

# ── User config (override via --export=ALL,VAR=...) ────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"                            # keep in sync with --gres
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
FIXED_LAMBDA="${FIXED_LAMBDA:--1.0}"                 # -1.0 = learned; 0..1 = pinned
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-/opt/miniforge3}"   # shared miniforge from setup_cluster.sh

# Shared/persistent storage visible from the compute node.
BASE_CKPT="${BASE_CKPT:-$HOME/mdlm/checkpoints/mdlm.ckpt}"
DATA_DIR="${DATA_DIR:-$HOME/mdlm/data}"              # must contain owt_cache/
OUT_DIR="${OUT_DIR:-$HOME/mdlm/outputs}"

# Compute nodes are often air-gapped → default W&B to offline (sync later).
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── Activate conda env ─────────────────────────────────────────────────────
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "[job] CONDA_HOME=$CONDA_HOME not found."; exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Run from language/ (this script lives in language/verda/) ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

echo "[job] node=$(hostname)  NUM_GPUS=${NUM_GPUS}  FIXED_LAMBDA=${FIXED_LAMBDA}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

bash run_verda_topk_lambda.sh \
  --num-gpus "$NUM_GPUS" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --base-ckpt "$BASE_CKPT" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --fixed-lambda "$FIXED_LAMBDA"

echo "[job] done. Outputs under ${OUT_DIR}"
[[ "$WANDB_MODE" == "offline" ]] && \
  echo "[job] W&B offline — sync later:  wandb sync ${OUT_DIR}/*/wandb/offline-run-*"

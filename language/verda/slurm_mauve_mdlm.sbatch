#!/bin/bash
# ===========================================================================
# slurm_mauve_mdlm.sbatch — MAUVE evaluation of *vanilla* MDLM on SLURM.
#
# Single-node, single-GPU. Generates 5000 samples from the base MDLM
# checkpoint (mdlm_owt.ckpt), scores them with MAUVE against 5000 human
# OpenWebText references, and writes the result JSON to OUT_DIR.
#
# EDIT the #SBATCH lines marked TODO for your cluster, then submit:
#   sbatch verda/slurm_mauve_mdlm.sbatch
#
# Optional overrides via --export=ALL,VAR=...:
#   SAMPLING_STEPS   diffusion steps         (default: 128)
#   P_NUCLEUS        top-p                   (default: 0.9)
#   SEED             seed                    (default: 1)
#   NUM_SAMPLES      # of generated samples  (default: 5000)
#   BASE_CKPT        path to mdlm_owt.ckpt
#   DATA_CACHE_DIR   OWT tokenization cache
#   OUT_DIR          where to write the MAUVE JSON
# ===========================================================================
#SBATCH --job-name=mauve-mdlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                 # single GPU — vanilla MDLM is single-process
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # ~8 dataloader workers is plenty for batch=1
#SBATCH --mem=64G                    # TODO: cluster-dependent
#SBATCH --time=08:00:00              # 5000 samples @ seq_len=1024 on a single L40S/A100 is ~5-7h
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<PARTITION>     # TODO: uncomment + set
##SBATCH --account=<ACCOUNT>         # TODO: uncomment + set if your cluster needs it

set -euo pipefail

# ── User config ────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

SAMPLING_STEPS="${SAMPLING_STEPS:-128}"
P_NUCLEUS="${P_NUCLEUS:-0.9}"
SEED="${SEED:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"

BASE_CKPT="${BASE_CKPT:-$HOME/checkpoints/mdlm_owt.ckpt}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/sm_data/owt_cache}"
export OUT_DIR="${OUT_DIR:-$HOME/sm_outputs/mauve/mdlm_vanilla_seed${SEED}}"

# W&B stays offline — sync from the login node later if you want.
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── Activate env (built by setup_cluster.sh) ───────────────────────────────
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "[job] CONDA_HOME=$CONDA_HOME not found — run verda/setup_cluster.sh first."; exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Run from language/ ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

mkdir -p "$OUT_DIR"
GENERATED_SEQS_PATH="${OUT_DIR}/mdlm_T-${SAMPLING_STEPS}_topp-${P_NUCLEUS}.json"

echo "[job] node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[job] BASE_CKPT=$BASE_CKPT"
echo "[job] DATA_CACHE_DIR=$DATA_CACHE_DIR"
echo "[job] OUT_DIR=$OUT_DIR"
echo "[job] NUM_SAMPLES=$NUM_SAMPLES  STEPS=$SAMPLING_STEPS  topp=$P_NUCLEUS  seed=$SEED"

# ── Launch ─────────────────────────────────────────────────────────────────
# mode=sample_eval triggers generation + MAUVE inside main.py (lines 200+).
# algo=mdlm + sampler=mdlm is the vanilla MDLM path (no soft-masking).
python -u -m main \
    data.cache_dir="$DATA_CACHE_DIR" \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    algo=mdlm \
    model.length=1024 \
    eval.checkpoint_path="${BASE_CKPT}" \
    +wandb.offline=true \
    hydra.run.dir="${PWD}" \
    sampling.steps="${SAMPLING_STEPS}" \
    sampling.num_sample_batches="${NUM_SAMPLES}" \
    sampling.p_nucleus="${P_NUCLEUS}" \
    sampling.sampler="mdlm" \
    sampling.generated_seqs_path="${GENERATED_SEQS_PATH}" \
    seed="${SEED}"

echo "[job] done. MAUVE result JSON: ${GENERATED_SEQS_PATH}"
[[ "$WANDB_MODE" == "offline" ]] && \
  echo "[job] W&B was offline — sync later:  wandb sync ${OUT_DIR}/wandb/offline-run-*"
#!/bin/bash
#SBATCH --job-name=mdlm-scratch-slerp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=mdlm-scratch-slerp-%j.out

# Activate conda environment (matches environment setup in README)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm-env

# Navigate to the language experiment directory
cd "$SLURM_SUBMIT_DIR"

# Run SLERP-SM pretraining from scratch with the reliability schedule
bash run_verda.sh \
  --num-gpus 2 \
  --max-steps 15000 \
  --base-ckpt none \
  --reliability-conditioned true \
  --reliability-start 0.05 \
  --reliability-full 0.25 \
  --reliability-beta 0.99 \
  --init-centre -9.7 \
  --data-dir "$HOME/sm_data" \
  --out-dir "$HOME/sm_outputs" \
  "$@"


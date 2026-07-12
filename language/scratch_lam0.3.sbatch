#!/bin/bash
#SBATCH --job-name=lam0.3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm-lam0.3-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm-env
# export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=pretraining
cd "$HOME/mdlm/soft-masked-diffusion-language-models/language"

bash scripts/from_scratch/run_scratch_slerp.sh \
  --num-gpus 1 \
  --seed 1 \
  --max-steps 5000 \
  --sm-prob 0.5 \
  --model small \
  --data openwebtext-split \
  --global-batch-size 512 \
  --batch-size 16 \
  --data-dir /home/aryan_s2/sm_data \
  --val-check-interval 200000 \
  --out-dir /home/aryan_s2/sm_outputs \
  --fixed-lambda 0.3

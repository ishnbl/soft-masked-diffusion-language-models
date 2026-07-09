#!/bin/bash
#SBATCH --job-name=topk-fl0.3-s3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=topk-fl0.3-s3-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm-env
cd "$HOME/mdlm/soft-masked-diffusion-language-models/language"
bash run_verda_topk.sh \
  --num-gpus 2 \
  --fixed-lambda 0.3 \
  --seed 3 \
  --checkpoint-every 500 \
  --base-ckpt "$HOME/mdlm/soft-masked-diffusion-language-models/mdlm.ckpt" \
  --data-dir "$HOME/mdlm/data" \
  --out-dir "$HOME/mdlm/outputs"

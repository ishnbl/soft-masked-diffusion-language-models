#!/bin/bash
#SBATCH --job-name=slerp-fl0.3-s3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slerp-fl0.3-s3-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm-env
cd "$HOME/mdlm/soft-masked-diffusion-language-models/language"
bash run_verda.sh \
  --num-gpus 2 \
  --fixed-lambda 0.3 \
  --seed 3 \
  --base-ckpt "$HOME/mdlm/soft-masked-diffusion-language-models/mdlm.ckpt" \
  --data-dir "$HOME/mdlm/data" \
  --out-dir "$HOME/mdlm/outputs"

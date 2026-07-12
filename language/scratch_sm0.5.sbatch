#!/bin/bash
#SBATCH --job-name=scratch-sm-mdlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-scratch-sm-mdlm-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm-env
# export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=pretraining
cd "$HOME/mdlm/soft-masked-diffusion-language-models/language"

echo "[job] node=$SLURMD_NODENAME  ALG=slerp  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[job] NUM_GPUS=1"
echo "[job] launching scripts/from_scratch/run_scratch_slerp.sh --num-gpus 1 --seed 1 --max-steps 5000 --sm-prob 0.5 --model small --data openwebtext-split --global-batch-size 512 --batch-size 16 --data-dir /home/aryan_s2/sm_data --out-dir /home/aryan_s2/sm_outputs"

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
  --out-dir /home/aryan_s2/sm_outputs

#!/bin/bash
#SBATCH --job-name=base0.0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm-base0.0-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm-env
export WANDB_API_KEY=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
export WANDB_PROJECT=pretraining
cd "$HOME/mdlm/soft-masked-diffusion-language-models/language"

bash scripts/from_scratch/run_scratch_slerp.sh \
  --num-gpus 1 \
  --seed 1 \
  --sm-prob 0 \
  --sm-prob-warmup-steps 0 \
  --model small \
  --data openwebtext-split \
  --global-batch-size 512 \
  --batch-size 16 \
  --data-dir /home/aryan_s2/sm_data \
  --val-check-interval 200000 \
  --out-dir /home/aryan_s2/sm_outputs \
  --resume-from-ckpt false \
  --fixed-lambda 0.0 \
  trainer.limit_val_batches=0 \
  trainer.max_steps=-1 \
  algo.tran_head.reliability_conditioned=false "$@"

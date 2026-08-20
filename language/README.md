# Language Modeling with Soft-Masked Diffusion Language Models

## Requirements

### Install Dependencies
To get started, create a conda environment containing the required dependencies.

```bash
conda create -n sm-env python=3.12
conda activate sm-env
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

We use **Weights & Biases** (W&B) to track training runs. To install and authenticate W&B, run:

```bash
pip install wandb
wandb login
```

### Download and Apply Patches
To download and apply patches to files which are directly used from DUO, please run the following bash script: 
```bash
bash setup.sh
```

## Checkpoints Language Modeling
You can download the pretrained soft-masking checkpoints for language modeling (OpenWebText) from our [Hugging Face repository](https://huggingface.co/lavanyanigam/soft-masking-checkpoints). They contain models in the iso-compute and iso-update regimes, both with pretraining continuation or training from scratch. 

## Training the Models

For training continuation, you have to first download the binary MDLM from this [Google drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing) and specify the checkpoint's location in the [training script](./scripts/01_sm_pretraining_cont_owt.sh). Then, run

```bash
./scripts/01_sm_pretraining_cont_owt.sh
```

To train SM on OWT from scratch, run the following script:
```bash
./scripts/02_sm_pretraining_scratch_owt.sh
```


### Multi-GPU (2 or 4 GPUs) — SLERP soft-masking

The `slerp_sm` soft-masking path runs data-parallel under Lightning DDP. A turnkey
launcher auto-detects the GPU count and keeps the **global batch fixed** across GPU
counts (it derives the per-GPU micro-batch / gradient-accumulation for you), so a
2-GPU and a 4-GPU run optimize the same effective batch:

```bash
# 4 GPUs (auto-detected), from scratch:
bash scripts/run_slerp_multigpu.sh

# 2 GPUs, finetuning from the released MDLM checkpoint:
NUM_GPUS=2 BASE_CKPT=/path/to/mdlm.ckpt bash scripts/run_slerp_multigpu.sh

# lower the per-GPU micro-batch on OOM (accumulation is recomputed automatically):
NUM_GPUS=4 PER_GPU_BATCH=16 bash scripts/run_slerp_multigpu.sh
```

The soft-mask gate (Bernoulli `sm_prob` + time band) is decided **identically on every
rank** (step-seeded Bernoulli + globally-reduced `t` mean in `MDLM_SM.nll`), so all ranks
take the same soft-mask branch each step and the DDP gradient all-reduce stays in sync.

`NUM_GPUS` smaller than the number of visible GPUs is handled by pinning
`CUDA_VISIBLE_DEVICES` to the first `NUM_GPUS` devices, because this codebase derives the
world size from `torch.cuda.device_count()` (not `trainer.devices`). To select *specific*
GPUs, set `CUDA_VISIBLE_DEVICES` yourself and the launcher will honor it
(`CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_slerp_multigpu.sh`).

#### On Modal (multi-GPU)

To test the multi-GPU `slerp_sm` path on [Modal](https://modal.com) (N GPUs in one
container = N-way DDP on a single node), use `run_modal_multigpu.py`. It reuses the
image / volumes / dataset-cache setup from `run_modal.py` and exercises the DDP-safe
gate under real multi-process DDP.

```bash
pip install modal && modal setup
modal secret create wandb-secret WANDB_API_KEY=<your-key>
modal volume put mdlm-checkpoints /local/path/mdlm_owt.ckpt mdlm_owt.ckpt

# GPU count/type are set via NUM_GPUS / GPU_TYPE env vars (fixed at import time).
# quick 2-GPU smoke test (20 steps, small global batch so it finishes fast):
NUM_GPUS=2 GPU_TYPE=L40S modal run run_modal_multigpu.py \
    --max-steps 20 --global-batch-size 64

# full 4-GPU finetune:
NUM_GPUS=4 GPU_TYPE=A100 modal run run_modal_multigpu.py \
    --max-steps 5000 --global-batch-size 512 --batch-size 32
```

## Evaluation

For unconstrained generation experiments, specify the checkpoint's location in the [evaluation script](./scripts/03_gen_ppl_owt_mdlm_sm.sh). For standard MDLM sampling `NFE` number of evaluations, run
```bash
./scripts/03_gen_ppl_owt_mdlm_sm.sh <NFE>
```
The generated samples as well as the metrics (generative perplexity, entropy, MAUVE) are in the `generations` folder. 

For experimenting with REMDM, run 
```bash
./scripts/04_gen_ppl_owt_remdm_cap_sm.sh <NFE>
```
for `NFE` in 128, 256, and 512. For `NFE=1024`, run 

```bash
./scripts/05_gen_ppl_owt_remdm_loop_sm.sh 1024
```

## Spherical Soft-Masking (SLERP-SM)

To address the norm-collapse failure of standard LERP soft-masking, you can configure the model to use **Spherical Soft-Masking (SLERP-SM)**. Under this configuration, the feedback is computed on the unit hypersphere using the weighted Fréchet mean of the top-k predictions and Spherical Linear Interpolation (SLERP), followed by rescaling to the mask token's original norm.

### Configuration Hyperparameters
Modify `configs/algo/mdlm_sm.yaml` (or specify in your run arguments):
* `transparency_alg`: `slerp_sm` (use `mixinputs_with_topk` or `mixinputs_with_temp` for linear/LERP baselines)
* `mixinputs_k`: Number of top-k tokens to aggregate (default: `3`)
* `slerp_n_iter`: Number of iterative Karcher flow steps to compute the Fréchet mean (default: `3`)
* `epsilon`: Numerical tolerance for clipping cosine similarity and preventing division by zero (default: `1e-6`)

### Language Modeling Comparison (COLM 2026 results)

#### 1. Validation Performance on OpenWebText (5,000 steps)
| Feedback Configuration | val/BPD | val/NLL | val/PPL |
|---|---|---|---|
| Vanilla MDLM | 4.5194 | 3.1326 | 22.9336 |
| **Fixed Lambda = 0.3 Regimes**: | | | |
| LERP SM | 4.9719 | 3.4462 | 31.3821 |
| **Our SLERP SM** | **4.7050** | **3.2614** | **26.0860** |
| **Learned Lambda Regimes**: | | | |
| LERP SM | 4.5300 | 3.1399 | 23.1030 |
| **Our SLERP SM** | **4.5370** | **3.1450** | **23.2270** |

#### 2. Unconstrained Generation Quality (at Step 5,000)
| Feedback | Gen PPL | Entropy | MAUVE |
|---|---|---|---|
| Vanilla MDLM | 60.17 | 5.51 | 0.0201 |
| **Fixed Lambda = 0.3**: | | | |
| LERP SM | 48.25 | 5.52 | 0.0551 |
| **SLERP SM (Ours)** | **26.91** | **5.35** | **0.2588** |
| **Learned Lambda**: | | | |
| LERP SM | 118.91 | 5.67 | 0.0071 |
| **SLERP SM (Ours)** | **51.38** | **5.49** | **0.0293** |

---

## Acknowledgements
This part of the repository was built on top of [Duo](https://github.com/s-sahoo/duo) and ReMDM [ReMDM](https://github.com/kuleshov-group/remdm). 

## Citation
If you use the work released here for your research, please consider citing our papers:
```bibtex
@article{lost_in_interpolation_2026,
  title={Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models},
  author={Nigam, Lavanya and contributors},
  journal={Conference on Language Modeling (COLM)},
  year={2026}
}

@inproceedings{hersche_softmasking_2026,
  title={Soft-Masked Diffusion Language Models},
  author={Hersche, Michael and Moor-Smith, Samuel and Hofmann, Thomas and Rahimi, Abbas},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=Gba02UMvrG}
}
```



# Step-by-step: VM setup on Verda

## Step 1 — Spin up the VM

On Verda's dashboard, pick a VM image with Ubuntu 22.04 + CUDA 12.4. CUDA 12.4 is critical — the PyTorch and flash-attention wheels are built specifically for it. If 12.4 isn't available, 12.1 or 12.2 will also work with these wheels (they're cu12-compatible), but 12.4 is safest.

## Step 2 — SSH in and verify CUDA

```bash
nvidia-smi
# You must see CUDA Version: 12.x in the top right
# You must see your GPU(s) listed

```

## Step 3 — Install Python 3.12

Ubuntu 22.04 ships with Python 3.10. You need 3.12 specifically (the flash-attn prebuilt wheel is cp312 — it will refuse to install on any other Python version).

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-dev python3.12-venv git git-lfs curl patch

```

## Step 4 — Create a virtual environment

```bash
python3.12 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

```

Add `source ~/venv/bin/activate` to your `~/.bashrc` so you don't have to run it every session.

## Step 5 — Install packages in the correct order

Order matters here. Do not combine these into one command.

```bash
# 1. numpy first
pip install "numpy<2"

# 2. PyTorch with CUDA 12.4 wheels (must use the index URL — do NOT use plain pip install torch)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. triton (must match torch 2.5)
pip install triton==3.1.0

# 4. Everything else
pip install \
    datasets==2.15.0 einops==0.7.0 "fsspec==2023.10.0" h5py==3.10.0 \
    hydra-core==1.3.2 lightning==2.2.1 omegaconf==2.3.0 packaging==23.2 \
    pandas==2.2.1 rich==13.7.1 scikit-learn==1.4.0 transformers==4.38.2 \
    torchmetrics==1.6.1 timm wandb huggingface-hub hf_transfer mauve-text \
    ipdb==0.13.13 nvitop==1.3.2

# 5. Flash attention — prebuilt wheel ONLY, do not build from source
pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

```

## Step 6 — Verify the install

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must print: 2.5.1+cu124  and  True

python -c "import flash_attn; print(flash_attn.__version__)"
# Must print: 2.7.4.post1

python -c "import mauve; print('mauve ok')"
# Must print: mauve ok

```

## Step 7 — Upload your codebase

```bash
# From your local machine, copy the language/ directory to the VM
scp -r ./language/ user@<verda-vm-ip>:/workspace/language/

```

## Step 8 — Run setup.sh

```bash
cd /workspace/language
bash setup.sh
# This downloads models/dit.py, models/ema.py, dataloader.py, metrics.py, utils.py
# from github.com/s-sahoo/duo branch ch-1 and applies your patches

```

## Step 9 — Download and place the MDLM checkpoint

Download `mdlm_owt.ckpt` from Google Drive (use a browser, gdown, or the Drive API), then:

```bash
mkdir -p /workspace/checkpoints
# Copy the file you downloaded to:
cp mdlm_owt.ckpt /workspace/checkpoints/mdlm_owt.ckpt

```

The run script defaults to `/workspace/checkpoints/mdlm_owt.ckpt`. If you put it elsewhere, pass `--base-ckpt /your/path/mdlm_owt.ckpt`.

## Step 10 — Set your W&B key

```bash
export WANDB_API_KEY=your_key_here
# Add to ~/.bashrc to persist it

```

## Step 11 — Run

```bash
cd /workspace/language

# Smoke test (fast, 2 GPUs, 20 steps)
bash run_verda.sh --num-gpus 2 --max-steps 20 --global-batch-size 64 --batch-size 16

# Full run (4 GPUs)
bash run_verda.sh --num-gpus 4 --max-steps 5000

```

## If Verda gives you a container deployment instead of a VM

Use the Dockerfile above. Build it from inside your `language/` directory:

```bash
docker build -t slerp-mdlm .
docker run --gpus all \
    -v /your/local/checkpoints:/workspace/checkpoints \
    -v /your/local/data:/workspace/data \
    -v /your/local/outputs:/workspace/outputs \
    -e WANDB_API_KEY=your_key \
    slerp-mdlm \
    bash run_verda.sh --num-gpus 4

```

## One thing to confirm with your collaborator

When they push their changes to `main.py` / `algo.py`, they may change the hydra config args in the run command (things like `algo.tran_head.*`). The `run_verda.sh` uses the same args as the current Modal script — if they add new config keys, you'll need to add those to `run_verda.sh` as well. Ask them to flag any new hydra args they add.
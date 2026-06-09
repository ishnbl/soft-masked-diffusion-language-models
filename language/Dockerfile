# ============================================================
# Dockerfile — SLERP / Soft-Masked Diffusion LLM
# Matches run_modal_multigpu.py image spec exactly.
# Base: CUDA 12.4.1 + cuDNN 9 + Ubuntu 22.04
# Python: 3.12
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Speeds up HuggingFace downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── 1. System packages ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    patch \
    git-lfs \
    software-properties-common \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Make python3.12 the system default ────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ── 3. pip ───────────────────────────────────────────────────
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && pip install --upgrade pip

# ── 4. numpy (must precede torch) ────────────────────────────
RUN pip install "numpy<2"

# ── 5. PyTorch stack — CUDA 12.4 wheels ──────────────────────
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 6. Triton (must match torch 2.5) ─────────────────────────
RUN pip install triton==3.1.0

# ── 7. All other ML dependencies ─────────────────────────────
RUN pip install \
    datasets==2.15.0 \
    einops==0.7.0 \
    "fsspec==2023.10.0" \
    h5py==3.10.0 \
    hydra-core==1.3.2 \
    lightning==2.2.1 \
    omegaconf==2.3.0 \
    packaging==23.2 \
    pandas==2.2.1 \
    rich==13.7.1 \
    scikit-learn==1.4.0 \
    transformers==4.38.2 \
    torchmetrics==1.6.1 \
    timm \
    wandb \
    huggingface-hub \
    hf_transfer \
    mauve-text \
    ipdb==0.13.13 \
    nvitop==1.3.2

# ── 8. Flash Attention — prebuilt wheel ──────────────────────
# --no-deps: avoids re-installing torch (matches Modal exactly).
# Wheel is built for: CUDA 12.x | torch 2.5 | cxx11abi=False | Python 3.12
RUN pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# ── 9. Copy codebase ─────────────────────────────────────────
WORKDIR /workspace/language
COPY . .

# ── 10. Run setup.sh ─────────────────────────────────────────
# Downloads DUO model files from GitHub and applies patches.
# Requires: patches/ dir and models/ dir to exist in your repo.
RUN bash setup.sh

# ── 11. Checkpoint mount point ───────────────────────────────
# Mount your checkpoint directory here at runtime, e.g.:
#   docker run -v /your/local/checkpoints:/workspace/checkpoints ...
RUN mkdir -p /workspace/checkpoints /workspace/data /workspace/outputs

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

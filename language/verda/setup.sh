#!/bin/bash
#
# verda/setup.sh — one-time bootstrap for a Verda GPU VM.
# ---------------------------------------------------------------------------
# Brings a fresh Verda Ubuntu GPU instance to the SAME environment the Modal
# runners use (run_modal_multigpu.py / run_modal_multigpu_topk.py), so a run on
# the VM reproduces a run on Modal byte-for-byte:
#
#   * Python 3.12 (via Miniconda) — required by the prebuilt flash-attn cp312
#     wheel, so we skip the ~20-min source build.
#   * torch 2.5.1 + cu124 wheels (torch bundles its own CUDA runtime — no system
#     CUDA toolkit needed for training).
#   * the exact pinned deps from the Modal image.
#   * DUO source files (models/dit.py etc.) downloaded + patched via setup.sh.
#
# A Verda GPU image already ships the NVIDIA driver; this script does NOT touch
# it. It only builds the userspace Python env.
#
# USAGE (on the VM, after `git clone` + cd into language/):
#   WANDB_API_KEY=<key> bash verda/setup.sh
#
# KNOBS (env vars):
#   ENV_NAME        conda env name              (default: sm-env)
#   WANDB_API_KEY   logs in non-interactively   (optional; else prompted/offline)
#   SKIP_FLASH      =1 to skip flash-attn        (training still works, slower)
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-sm-env}"
PYTHON_VERSION="3.12"
SKIP_FLASH="${SKIP_FLASH:-0}"

# Resolve language/ (this script lives in language/verda/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

if [[ ! -f setup.sh || ! -f main.py ]]; then
  echo "[verda] Expected to run from the repo's language/ dir (found via verda/)."
  echo "        cwd=$LANG_DIR is missing setup.sh/main.py — clone the repo first."
  exit 1
fi

sudo_if_needed() { if [[ $EUID -ne 0 ]] && command -v sudo &>/dev/null; then sudo "$@"; else "$@"; fi; }

echo "=== [verda 1/6] GPU / driver check ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
    | awk -F, '{printf "  GPU %s:%s  driver%s  %s\n",$1,$2,$3,$4}'
else
  echo "  [warn] nvidia-smi not found. A Verda *GPU* instance should have the driver"
  echo "         pre-installed; if this is a CPU instance, training will not run."
fi

echo "=== [verda 2/6] system packages (git, curl, patch, git-lfs) ==="
# DUO setup.sh needs curl + patch; git-lfs for any lfs assets.
if command -v apt-get &>/dev/null; then
  sudo_if_needed apt-get update -y
  sudo_if_needed apt-get install -y git curl patch git-lfs
else
  echo "  [warn] apt-get not found — ensure git/curl/patch are already installed."
fi

echo "=== [verda 3/6] Miniconda + Python ${PYTHON_VERSION} env (${ENV_NAME}) ==="
if ! command -v conda &>/dev/null; then
  echo "  conda not found — installing Miniconda to \$HOME/miniconda3 ..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda.sh
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "  env '${ENV_NAME}' exists — reusing."
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi
conda activate "$ENV_NAME"
echo "  active: $(python --version) @ $CONDA_PREFIX"

echo "=== [verda 4/6] Python deps (exact Modal-image pins) ==="
# Step 1: non-torch deps. numpy<2 because datasets 2.15 / transformers 4.38 are
# not numpy-2 ready.
pip install \
  "numpy<2" \
  "datasets==2.15.0" \
  "einops==0.7.0" \
  "fsspec==2023.10.0" \
  "h5py==3.10.0" \
  "hydra-core==1.3.2" \
  "lightning==2.2.1" \
  "omegaconf==2.3.0" \
  "packaging==23.2" \
  "pandas==2.2.1" \
  "rich==13.7.1" \
  "scikit-learn==1.4.0" \
  "transformers==4.38.2" \
  "torchmetrics==1.6.1" \
  wandb timm huggingface-hub hf_transfer mauve-text

# Step 2: torch LAST with cu124 index so it overrides any transitive CPU torch.
pip install \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" "triton==3.1.0" \
  --extra-index-url https://download.pytorch.org/whl/cu124

echo "=== [verda 5/6] flash-attn ==="
if [[ "$SKIP_FLASH" == "1" ]]; then
  echo "  skipped (SKIP_FLASH=1)."
elif python -c "import flash_attn" 2>/dev/null; then
  echo "  already installed."
else
  # Prebuilt wheel matched to cu124 / torch2.5 / cp312. --no-deps so pip can't
  # downgrade torch to satisfy the wheel's loose dependency pin.
  pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
fi

echo "=== [verda 6/6] DUO source files + patches + W&B ==="
if [[ ! -f models/dit.py ]]; then
  echo "  downloading DUO files (models/dit.py etc.) ..."
  bash setup.sh
else
  echo "  DUO files present — skipping setup.sh."
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY" && echo "  W&B logged in."
else
  echo "  [note] WANDB_API_KEY not set. Either run 'wandb login', export the key,"
  echo "         or pass +wandb.offline=true to the launchers."
fi

echo ""
echo "[verda] Setup complete. Activate with:  conda activate ${ENV_NAME}"
echo "[verda] Then launch (from language/):"
echo "          BASE_CKPT=/path/to/mdlm_owt.ckpt bash verda/run_slerp.sh"
echo "          BASE_CKPT=/path/to/mdlm_owt.ckpt bash verda/run_topk.sh"

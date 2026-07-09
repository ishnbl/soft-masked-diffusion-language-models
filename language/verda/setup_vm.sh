#!/bin/bash
#
# setup_vm.sh — one-shot bootstrap for a GENERIC Ubuntu GPU VM.
# ---------------------------------------------------------------------------
# Identical to verda/setup.sh EXCEPT it does not assume the NVIDIA driver is
# already installed. Verda's GPU image ships the driver; a bare cloud VM
# (AWS/GCP/Azure/bare-metal) often does not. This script:
#
#   * checks for a working driver (nvidia-smi);
#   * if absent: either installs it (INSTALL_DRIVER=1, REBOOT REQUIRED) or stops
#     with instructions — it will NOT silently continue without a GPU;
#   * then builds the exact same Python env as the Modal runners (Miniconda +
#     Python 3.12, torch 2.5.1 cu124, prebuilt flash-attn cp312 wheel, DUO files).
#
# You do NOT need a system CUDA toolkit: torch's cu124 wheels bundle the CUDA
# runtime. Only the kernel-mode driver must be present.
#
# USAGE (on the VM, after git clone + cd into language/):
#   WANDB_API_KEY=<key> bash verda/setup_vm.sh
#
#   # if nvidia-smi is missing and you want this script to install the driver
#   # (Ubuntu only; reboots are required afterward, then re-run this script):
#   INSTALL_DRIVER=1 bash verda/setup_vm.sh
#
# KNOBS (env vars):
#   ENV_NAME         conda env name              (default: sm-env)
#   WANDB_API_KEY    logs in non-interactively   (optional)
#   SKIP_FLASH       =1 to skip flash-attn        (training still works, slower)
#   INSTALL_DRIVER   =1 to apt-install the NVIDIA driver if nvidia-smi is missing
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-sm-env}"
PYTHON_VERSION="3.12"
SKIP_FLASH="${SKIP_FLASH:-0}"
INSTALL_DRIVER="${INSTALL_DRIVER:-0}"

# Resolve language/ (this script lives in language/verda/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

if [[ ! -f setup.sh || ! -f main.py ]]; then
  echo "[vm] Run from the repo's language/ dir (found via verda/). cwd=$LANG_DIR"
  echo "     is missing setup.sh/main.py — clone the repo first."
  exit 1
fi

sudo_if_needed() { if [[ $EUID -ne 0 ]] && command -v sudo &>/dev/null; then sudo "$@"; else "$@"; fi; }

echo "=== [vm 1/7] NVIDIA driver check ==="
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
    | awk -F, '{printf "  GPU %s:%s  driver%s  %s\n",$1,$2,$3,$4}'
else
  echo "  [warn] no working NVIDIA driver (nvidia-smi missing or failing)."
  if [[ "$INSTALL_DRIVER" == "1" ]]; then
    if command -v apt-get &>/dev/null; then
      echo "  INSTALL_DRIVER=1 — installing the recommended driver (Ubuntu) ..."
      sudo_if_needed apt-get update -y
      sudo_if_needed apt-get install -y ubuntu-drivers-common
      sudo_if_needed ubuntu-drivers autoinstall
      echo ""
      echo "  >>> Driver installed. You MUST reboot now, then re-run this script:"
      echo "        sudo reboot"
      echo "        # after it comes back:"
      echo "        WANDB_API_KEY=<key> bash verda/setup_vm.sh"
      exit 0
    else
      echo "  [err] apt-get not found — install the NVIDIA driver via your distro,"
      echo "        reboot, then re-run this script."
      exit 1
    fi
  else
    echo "  This is the ONLY part that differs from a Verda VM (Verda ships the"
    echo "  driver). Install it, reboot, and re-run — or pass INSTALL_DRIVER=1 to"
    echo "  let this script do it (Ubuntu only)."
    exit 1
  fi
fi

echo "=== [vm 2/7] system packages (git, curl, patch, git-lfs) ==="
if command -v apt-get &>/dev/null; then
  sudo_if_needed apt-get update -y
  sudo_if_needed apt-get install -y git curl patch git-lfs
else
  echo "  [warn] apt-get not found — ensure git/curl/patch are installed."
fi

echo "=== [vm 3/7] Miniconda + Python ${PYTHON_VERSION} env (${ENV_NAME}) ==="
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

echo "=== [vm 4/7] Python deps (exact Modal-image pins) ==="
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

pip install \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" "triton==3.1.0" \
  --extra-index-url https://download.pytorch.org/whl/cu124

echo "=== [vm 5/7] flash-attn ==="
if [[ "$SKIP_FLASH" == "1" ]]; then
  echo "  skipped (SKIP_FLASH=1)."
elif python -c "import flash_attn" 2>/dev/null; then
  echo "  already installed."
else
  pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
fi

echo "=== [vm 6/7] DUO source files + patches ==="
if [[ ! -f models/dit.py ]]; then
  echo "  downloading DUO files ..."
  bash setup.sh
else
  echo "  DUO files present — skipping setup.sh."
fi

echo "=== [vm 7/7] sanity check + W&B ==="
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"device_count={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA not available to torch — driver/runtime mismatch."
try:
    import flash_attn; print(f"  flash_attn {flash_attn.__version__}")
except Exception as e:
    print(f"  flash_attn not importable ({e}) — training still runs, slower.")
PY

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY" && echo "  W&B logged in."
else
  echo "  [note] WANDB_API_KEY not set. Run 'wandb login' or pass +wandb.offline=true."
fi

echo ""
echo "[vm] Setup complete. Activate with:  conda activate ${ENV_NAME}"
echo "[vm] Then launch (from language/):"
echo "       BASE_CKPT=/path/to/mdlm_owt.ckpt bash verda/run_slerp.sh"
echo "       BASE_CKPT=/path/to/mdlm_owt.ckpt bash verda/run_topk.sh"

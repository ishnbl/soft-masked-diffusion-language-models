#!/bin/bash
#
# setup_cluster.sh — no-sudo env bootstrap for a SLURM / HPC cluster.
# ---------------------------------------------------------------------------
# Same target environment as the Modal runners, but built WITHOUT root:
#   * NO apt-get / sudo. System tools (git, curl, patch) are assumed present
#     (module-provided or preinstalled); we only verify them.
#   * Miniconda is installed into $HOME (user-writable — no root needed), giving
#     Python 3.12 for the prebuilt flash-attn cp312 wheel.
#   * torch 2.5.1 cu124 + the exact pinned deps + DUO files.
#
# Run this ONCE on a node WITH INTERNET (usually the login node — compute nodes
# are often air-gapped). flash-attn is a prebuilt wheel, so no GPU/compiler is
# needed at install time; this is safe on the login node.
#
# USAGE (login node, from language/):
#   bash verda/setup_cluster.sh
#   # then submit jobs with verda/slurm_train.sbatch (see that file).
#
# KNOBS:
#   ENV_NAME      conda env name              (default: sm-env)
#   CONDA_HOME    where to install Miniconda  (default: $HOME/miniconda3)
#   SKIP_FLASH    =1 to skip flash-attn
#   MODULE_LOAD   optional module(s) to load first, e.g. MODULE_LOAD="cuda/12.4"
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-sm-env}"
PYTHON_VERSION="3.12"
SKIP_FLASH="${SKIP_FLASH:-0}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

if [[ ! -f setup.sh || ! -f main.py ]]; then
  echo "[cluster] Run from the repo's language/ dir. cwd=$LANG_DIR missing setup.sh/main.py."
  exit 1
fi

# Optional environment modules (clusters often expose toolchains this way).
if [[ -n "${MODULE_LOAD:-}" ]] && command -v module &>/dev/null; then
  # shellcheck disable=SC2086
  module load ${MODULE_LOAD} || echo "[cluster] [warn] 'module load ${MODULE_LOAD}' failed — continuing."
fi

echo "=== [cluster 1/5] required tools (no install — verify only) ==="
missing=0
for t in git curl patch; do
  if command -v "$t" &>/dev/null; then
    echo "  ok: $t -> $(command -v "$t")"
  else
    echo "  [err] '$t' not found. Load it (e.g. 'module load git') or ask an admin."
    missing=1
  fi
done
[[ "$missing" == "1" ]] && { echo "[cluster] missing tools — aborting."; exit 1; }

echo "=== [cluster 2/5] Miniconda (\$HOME, no root) + Python ${PYTHON_VERSION} env ==="
if ! command -v conda &>/dev/null; then
  if [[ -x "$CONDA_HOME/bin/conda" ]]; then
    export PATH="$CONDA_HOME/bin:$PATH"
  else
    echo "  installing Miniconda to ${CONDA_HOME} ..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -o /tmp/miniconda.$$.sh
    bash /tmp/miniconda.$$.sh -b -p "$CONDA_HOME"
    rm -f /tmp/miniconda.$$.sh
    export PATH="$CONDA_HOME/bin:$PATH"
  fi
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

echo "=== [cluster 3/5] Python deps (exact Modal-image pins) ==="
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

echo "=== [cluster 4/5] flash-attn (prebuilt wheel — no compile) ==="
if [[ "$SKIP_FLASH" == "1" ]]; then
  echo "  skipped (SKIP_FLASH=1)."
elif python -c "import flash_attn" 2>/dev/null; then
  echo "  already installed."
else
  pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
fi

echo "=== [cluster 5/5] DUO source files + patches ==="
if [[ ! -f models/dit.py ]]; then
  echo "  downloading DUO files ..."
  bash setup.sh
else
  echo "  DUO files present — skipping setup.sh."
fi

# CUDA check is a soft warning: this script usually runs on a GPU-less login node.
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  (cuda_available now={torch.cuda.is_available()} "
      f"— expected False on a login node; the GPU check happens inside the job)")
PY

echo ""
echo "[cluster] Setup complete."
echo "[cluster] In your sbatch script, before launching, do:"
echo "            source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "[cluster] Then submit:  sbatch verda/slurm_train.sbatch"

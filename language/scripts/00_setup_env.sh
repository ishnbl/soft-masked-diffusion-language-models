#!/bin/bash
#
# 00_setup_env.sh — One-shot environment setup for the SLERP-vs-topk experiment.
#
# Must be run from the language/ directory (the one containing requirements.txt
# and setup.sh).
#
# Usage:
#   cd language
#   bash scripts/00_setup_env.sh [options]
#
# Options:
#   --env-name   NAME     Conda env name             (default: sm-env)
#   --skip-flash         Skip flash_attn build       (saves ~20 min; training still works)
#   --skip-wandb         Skip 'wandb login' prompt   (set WANDB_API_KEY env var instead)
#   --data-dir   DIR      Where to cache datasets    (default: ./data_cache)
#   --no-color           Disable coloured output
#

# ── Defaults ──────────────────────────────────────────────────────────────────
ENV_NAME="sm-env"
PYTHON_VERSION="3.12"
CUDA_TOOLKIT_VERSION="12.4.0"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache}"
SKIP_FLASH=0
SKIP_WANDB=0
NO_COLOR=0

# ── Parse CLI flags ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)   ENV_NAME="$2"; shift 2 ;;
        --skip-flash) SKIP_FLASH=1; shift ;;
        --skip-wandb) SKIP_WANDB=1; shift ;;
        --data-dir)   DATA_CACHE_DIR="$2"; shift 2 ;;
        --no-color)   NO_COLOR=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colour helpers ─────────────────────────────────────────────────────────────
if [[ $NO_COLOR -eq 0 ]]; then
    BOLD=$'\033[1m'; CYAN=$'\033[36m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

step() { echo "${CYAN}${BOLD}=== $* ===${RESET}"; }
ok()   { echo "${GREEN}✓  $*${RESET}"; }
warn() { echo "${YELLOW}⚠  $*${RESET}"; }
die()  { echo "${RED}✗  $*${RESET}"; exit 1; }

# ── Guard: must be run from language/ ─────────────────────────────────────────
if [[ ! -f requirements.txt || ! -f setup.sh ]]; then
    die "Run this script from the language/ directory:  cd language && bash scripts/00_setup_env.sh"
fi

echo ""
echo "${BOLD}SLERP vs top-k experiment — environment setup${RESET}"
echo "  Conda env  : $ENV_NAME (Python $PYTHON_VERSION)"
echo "  CUDA kit   : $CUDA_TOOLKIT_VERSION"
echo "  Data cache : $DATA_CACHE_DIR"
echo "  flash_attn : $([ $SKIP_FLASH -eq 1 ] && echo skipped || echo yes)"
echo "  W&B login  : $([ $SKIP_WANDB -eq 1 ] && echo skipped || echo interactive)"
echo ""

# ── 1. GPU check ──────────────────────────────────────────────────────────────
step "1/7  GPU / CUDA driver"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
        | awk -F, '{printf "  GPU %s: %s  driver %s  %s\n",$1,$2,$3,$4}'
    ok "GPU detected"
else
    warn "nvidia-smi not found — training requires a CUDA GPU."
    warn "You can still build the environment; set --skip-flash if there is no CUDA toolkit."
fi

# ── 2. Conda environment ───────────────────────────────────────────────────────
step "2/7  Conda environment ($ENV_NAME)"
if ! command -v conda &>/dev/null; then
    die "conda not found. Install Miniconda/Anaconda first: https://docs.anaconda.com/miniconda/"
fi

if conda env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
    ok "Environment '$ENV_NAME' already exists — skipping creation."
else
    echo "  Creating environment (Python $PYTHON_VERSION)…"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    ok "Environment created."
fi

# Activate inside the script.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
ok "Activated: $(python --version) in $CONDA_PREFIX"

# ── 3. CUDA toolkit ───────────────────────────────────────────────────────────
step "3/7  CUDA $CUDA_TOOLKIT_VERSION toolkit (conda)"
if python -c "import torch; assert torch.version.cuda" 2>/dev/null; then
    ok "PyTorch with CUDA already installed — skipping conda toolkit step."
else
    conda install -y "nvidia/label/cuda-${CUDA_TOOLKIT_VERSION}::cuda-toolkit"
    ok "CUDA toolkit installed."
fi

# ── 4. Python dependencies ────────────────────────────────────────────────────
step "4/7  Python dependencies (requirements.txt)"
pip install -r requirements.txt
ok "Core dependencies installed."

# ── 5. flash_attn ─────────────────────────────────────────────────────────────
step "5/7  flash_attn"
if [[ $SKIP_FLASH -eq 1 ]]; then
    warn "Skipped (--skip-flash). Training works without it but may be slower."
else
    if python -c "import flash_attn" 2>/dev/null; then
        ok "flash_attn already installed."
    else
        echo "  Building flash_attn from source (this takes 10–20 min)…"
        pip install flash_attn==2.7.4.post1 --no-build-isolation \
            || { warn "flash_attn build failed. Training will still run without it."; }
    fi
fi

# ── 6. W&B ────────────────────────────────────────────────────────────────────
step "6/7  Weights & Biases"
pip install wandb -q

if [[ $SKIP_WANDB -eq 1 ]]; then
    warn "Skipping interactive login (--skip-wandb)."
    warn "Set WANDB_API_KEY=<your-key> before running training, or pass +wandb.offline=true."
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
    ok "W&B logged in via WANDB_API_KEY."
else
    echo "  Paste your API key when prompted (find it at https://wandb.ai/authorize)."
    wandb login
fi

# ── 7. DUO source files + patches ─────────────────────────────────────────────
step "7/7  DUO source files and patches"
DUO_FILES=(models/dit.py models/ema.py dataloader.py metrics.py utils.py)
ALL_PRESENT=1
for f in "${DUO_FILES[@]}"; do
    [[ -f "$f" ]] || { ALL_PRESENT=0; break; }
done

if [[ $ALL_PRESENT -eq 1 ]]; then
    ok "DUO files already present — skipping download."
    echo "  To re-download and re-patch, delete them and re-run."
else
    echo "  Downloading from s-sahoo/duo (branch ch-1)…"
    bash setup.sh
    ok "DUO files downloaded and patched."
fi

# ── Data cache dir ────────────────────────────────────────────────────────────
mkdir -p "$DATA_CACHE_DIR"
ok "Data cache directory: $(realpath "$DATA_CACHE_DIR")"
echo "  (text8 will be downloaded here automatically on the first training run)"

# ── Quick smoke test ──────────────────────────────────────────────────────────
echo ""
step "Smoke test — imports and CUDA"
python - <<'PYEOF'
import sys

failures = []

# Core libs
for mod in ("torch", "lightning", "hydra", "wandb", "einops", "transformers"):
    try:
        __import__(mod)
    except ImportError as e:
        failures.append(str(e))

# Local modules (must run from language/)
for mod in ("transparency_head",):
    try:
        __import__(mod)
    except ImportError as e:
        failures.append(f"{mod}: {e}")

import torch
print(f"  torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {p.name}  {p.total_memory // 1024**2} MB")
else:
    print("  (No CUDA GPU — set trainer.accelerator=cpu for testing)")

if failures:
    print("\n  Missing imports:", "\n    ".join(failures), file=sys.stderr)
    sys.exit(1)
else:
    print("  All imports OK")
PYEOF

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "${GREEN}${BOLD}Setup complete!${RESET}"
echo ""
echo "  Activate env:  ${BOLD}conda activate ${ENV_NAME}${RESET}"
echo ""
echo "  Run the A/B experiment (from language/):"
echo "    ${BOLD}DATA_CACHE_DIR=${DATA_CACHE_DIR} SEED=1 MAX_STEPS=30000 bash scripts/06_slerp_vs_topk_ablation.sh${RESET}"
echo ""
echo "  CPU smoke test (no GPU needed):"
echo "    ${BOLD}python -u -m main algo=mdlm_sm model=tiny data=text8 \\"
echo "      algo.tran_head.transparency_alg=slerp_sm \\"
echo "      trainer.accelerator=cpu trainer.devices=1 trainer.max_steps=2 \\"
echo "      trainer.num_sanity_val_steps=0 trainer.limit_val_batches=2 \\"
echo "      trainer.val_check_interval=2 loader.batch_size=4 \\"
echo "      optim.sm_prob=1.0 sampling.predictor=sm \\"
echo "      strategy.find_unused_parameters=True \\"
echo "      data.cache_dir=${DATA_CACHE_DIR} +wandb.offline=true wandb.name=smoke${RESET}"
echo ""

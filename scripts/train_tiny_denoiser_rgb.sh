#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=train-tiny-denoiser-rgb
#SBATCH --output=%x-%j.out

set -euo pipefail

# -------- configurable --------
REPO="$HOME/mtf-aware-deblurring"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K/DIV2K_train_LR_bicubic/X2}"
CKPT_OUT="${CKPT_OUT:-$REPO/src/mtf_aware_deblurring/assets/tiny_denoiser_sigma15.pth}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-3}"
DEPTH="${DEPTH:-10}"
FEATS="${FEATS:-64}"
PATCH="${PATCH:-64}"
MAX_IMGS="${MAX_IMGS:-800}"
PPI="${PPI:-60}"
NOISE_SIGMA="${NOISE_SIGMA:-15}"  # pixel space (0-255)
LOG_INT="${LOG_INT:-50}"
# --------------------------------

VENV_DIR="/tmp/$USER/envs/CSC2529"
PIP_CACHE="/tmp/$USER/pip-cache"
mkdir -p "$PIP_CACHE" "$VENV_DIR" "$HOME/mtf-logs" "$(dirname "$CKPT_OUT")"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR="$PIP_CACHE"

cd "$REPO"
python -m pip install --upgrade pip wheel
pip install --no-cache-dir -e .

python "$REPO/train_tiny_denoiser.py" \
  --div2k-root "$DATA_ROOT" \
  --output "$CKPT_OUT" \
  --max-images "$MAX_IMGS" \
  --patches-per-image "$PPI" \
  --patch-size "$PATCH" \
  --noise-sigma "$NOISE_SIGMA" \
  --depth "$DEPTH" \
  --features "$FEATS" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --log-interval "$LOG_INT"

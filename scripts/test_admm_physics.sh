#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=test-admm-physics
#SBATCH --output=%x-%j.out

set -euo pipefail

# ------------- configurable paths -------------
REPO="${REPO:-$HOME/mtf-aware-deblurring}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K}"
OUT_DIR="${OUT_DIR:-$HOME/mtf-smoke/admm-physics}"
VENV_DIR="${VENV_DIR:-/tmp/$USER/envs/CSC2529}"
PIP_CACHE="${PIP_CACHE:-/tmp/$USER/pip-cache}"
# Denoiser + ADMM hyperparameters (override via env vars as needed)
LIMIT="${LIMIT:-5}"
ADMM_ITERS="${ADMM_ITERS:-60}"
ADMM_RHO="${ADMM_RHO:-0.4}"
ADMM_DENOISER_WEIGHT="${ADMM_DENOISER_WEIGHT:-1.0}"
ADMM_MTF_SCALE="${ADMM_MTF_SCALE:-0.35}"
ADMM_MTF_FLOOR="${ADMM_MTF_FLOOR:-0.05}"
DENOISER_DEVICE="${DENOISER_DEVICE:-cuda}"
DENOISER_TYPE="${DENOISER_TYPE:-drunet_color}"
# ----------------------------------------------

mkdir -p "$PIP_CACHE" "$VENV_DIR" "$OUT_DIR" "$HOME/mtf-logs"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR="$PIP_CACHE"

cd "$REPO"
python -m pip install --upgrade pip wheel
pip install --no-cache-dir -e .

python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root "$DATA_ROOT" \
  --subset train \
  --degradation bicubic \
  --scale X2 \
  --image-mode rgb \
  --limit "$LIMIT" \
  --auto-download \
  --method admm \
  --admm-iters "$ADMM_ITERS" \
  --admm-rho "$ADMM_RHO" \
  --admm-denoiser-weight "$ADMM_DENOISER_WEIGHT" \
  --admm-mtf-scale "$ADMM_MTF_SCALE" \
  --admm-mtf-floor "$ADMM_MTF_FLOOR" \
  --denoiser-type "$DENOISER_TYPE" \
  --denoiser-device "$DENOISER_DEVICE" \
  --use-physics-scheduler \
  --collect-only \
  --output-dir "$OUT_DIR"

#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=test-admm-diffusion
#SBATCH --output=%x-%j.out

set -euo pipefail

# -------- configurable paths --------
REPO="$HOME/mtf-aware-deblurring"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K}"
OUT_DIR="${OUT_DIR:-$HOME/mtf-smoke/admm-diffusion}"
VENV_DIR="/tmp/$USER/envs/CSC2529"
PIP_CACHE="/tmp/$USER/pip-cache"
# Tiny score weights you already use in reconstruct:
DIFF_WEIGHTS="${DIFF_WEIGHTS:-$REPO/src/mtf_aware_deblurring/assets/tiny_score_unet.pth}"
# -----------------------------------

mkdir -p "$PIP_CACHE" "$VENV_DIR" "$OUT_DIR" "$HOME/mtf-logs"

# venv bootstrap on the node-local disk
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR="$PIP_CACHE"

# install repo (no cache in home)
cd "$REPO"
python -m pip install --upgrade pip wheel
pip install --no-cache-dir -e .

# run one-image reconstruct on CUDA
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root "$DATA_ROOT" \
  --subset valid \
  --degradation bicubic --scale X2 \
  --image-mode grayscale \
  --limit 1 \
  --auto-download \
  --method admm_diffusion \
  --diffusion-prior-type tiny_score \
  --diffusion-prior-weights "$DIFF_WEIGHTS" \
  --diffusion-steps 8 \
  --denoiser-device cuda \
  --diffusion-device cuda \
  --output-dir "$OUT_DIR"

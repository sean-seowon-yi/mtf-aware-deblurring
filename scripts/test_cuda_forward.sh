#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=test-admm-denoiser
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# --- config you might tweak ---
VENV_PATH="${VENV_PATH:-/tmp/$USER/envs/CSC2529}"   # or e.g., ~/envs/mtf if you keep a persistent one
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K}"
OUT_DIR="${OUT_DIR:-$HOME/mtf-smoke/admm}"
DENOISER_TYPE="${DENOISER_TYPE:-tiny}"
DENOISER_DEVICE="${DENOISER_DEVICE:-cuda}"
# ------------------------------

mkdir -p "$(dirname "$OUT_DIR")" logs

# Activate your env
source "$VENV_PATH/bin/activate"

# Run the pipeline (CUDA on proximal + denoiser)
cd "$HOME/mtf-aware-deblurring"

python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root "$DATA_ROOT" \
  --subset valid \
  --degradation bicubic --scale X2 \
  --image-mode grayscale \
  --limit 1 \
  --auto-download \
  --method admm \
  --admm-iters 30 \
  --admm-rho 0.45 \
  --admm-denoiser-weight 1.0 \
  --denoiser-type "$DENOISER_TYPE" \
  --denoiser-device "$DENOISER_DEVICE" \
  --output-dir "$OUT_DIR"

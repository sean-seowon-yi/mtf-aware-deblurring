#!/bin/bash
set -euo pipefail

VENV=.venv_admm_baseline
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
pip install --no-cache-dir numpy imageio pillow scipy lpips
pip install --no-cache-dir -e .

export PYTHONPATH=$(pwd)/src

python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 1 --patterns box random legendre \
  --method admm \
  --admm-iters 60 --admm-rho 1.04 --admm-denoiser-weight 0.16 \
  --admm-mtf-weighting-mode combined --admm-mtf-sigma-adapt --use-physics-scheduler \
  --denoiser-type drunet_color --denoiser-device cuda --denoiser-sigma-scale 8.0 \
  --photon-budget 1000 --blur-length 15 --read-noise 0.01 \
  --target-size 256 --collect-only --enable-ssim --enable-lpips --lpips-device cuda \
  --output-dir logs/admm_baseline_1img_local

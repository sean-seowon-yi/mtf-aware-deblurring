#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=train-unet-denoiser-pg
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------- config ----------
REPO="${REPO:-$HOME/mtf-aware-deblurring}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K/DIV2K_train_LR_bicubic/X2}"
CKPT_OUT="${CKPT_OUT:-$REPO/src/mtf_aware_deblurring/assets/unet_denoiser_pg.pth}"

EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
BASEF="${BASEF:-64}"
PATCH="${PATCH:-96}"
MAX_IMGS="${MAX_IMGS:-800}"
PPI="${PPI:-40}"
PHOTONS="${PHOTONS:-800.0}"
READNOISE="${READNOISE:-0.01}"
SEED="${SEED:-0}"
LOG_INT="${LOG_INT:-50}"
# ----------------------------

VENV_DIR="/tmp/$USER/envs/CSC2529"
PIP_CACHE="/tmp/$USER/pip-cache"
mkdir -p "$PIP_CACHE" "$VENV_DIR" "$REPO/logs" "$(dirname "$CKPT_OUT")"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR="$PIP_CACHE"

cd "$REPO"
python -m pip install --upgrade pip wheel
pip install --no-cache-dir -e .
pip install --no-cache-dir scipy

echo "[CUDA]" && nvidia-smi || true
python - <<'PY'
import torch; print("torch.cuda.is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

python "$REPO/scripts/train_unet_denoiser.py" \
  --dataset-root "$DATA_ROOT" \
  --output "$CKPT_OUT" \
  --max-images "$MAX_IMGS" \
  --patches-per-image "$PPI" \
  --patch-size "$PATCH" \
  --photon-budget "$PHOTONS" \
  --read-noise "$READNOISE" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --device cuda \
  --base-features "$BASEF" \
  --seed "$SEED" \
  --log-interval "$LOG_INT"

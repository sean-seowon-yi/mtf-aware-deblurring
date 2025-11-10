#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=train-large-score
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------- config ----------
REPO="${REPO:-$HOME/mtf-aware-deblurring}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K/DIV2K_train_LR_bicubic/X2}"
CKPT_OUT="${CKPT_OUT:-$REPO/src/mtf_aware_deblurring/assets/large_score_unet.pth}"

EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-32}"          # larger model → slightly smaller batch
LR="${LR:-1e-4}"
BASE_CH="${BASE_CH:-128}"
DEPTH="${DEPTH:-8}"
PATCH="${PATCH:-96}"
MAX_IMGS="${MAX_IMGS:-1000}"
PPI="${PPI:-60}"
SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-0.5}"
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

echo "[CUDA]" && nvidia-smi || true
python - <<'PY'
import torch; print("torch.cuda.is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

python "$REPO/scripts/train_tiny_score_model.py" \
  --dataset-root "$DATA_ROOT" \
  --output "$CKPT_OUT" \
  --max-images "$MAX_IMGS" \
  --patches-per-image "$PPI" \
  --patch-size "$PATCH" \
  --sigma-min "$SIGMA_MIN" \
  --sigma-max "$SIGMA_MAX" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --device cuda \
  --base-channels "$BASE_CH" \
  --depth "$DEPTH" \
  --seed "$SEED" \
  --log-interval "$LOG_INT"

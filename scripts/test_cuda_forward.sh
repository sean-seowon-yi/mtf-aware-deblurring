# scripts/test_admm_diffusion.sh
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=test-admm-diffusion
#SBATCH --output=logs/%x-%j.out

# --- config you might tweak ---
VENV_PATH="${VENV_PATH:-/tmp/$USER/envs/CSC2529}"   # or e.g., ~/envs/mtf if you keep a persistent one
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/DIV2K}"
OUT_DIR="${OUT_DIR:-$HOME/mtf-smoke/admm-diffusion}"
DIFF_WEIGHTS="${DIFF_WEIGHTS:-$HOME/mtf-aware-deblurring/src/mtf_aware_deblurring/assets/tiny_score_unet.pth}"
# ------------------------------

set -euo pipefail
mkdir -p "$(dirname "$OUT_DIR")" logs

# Activate your env
source "$VENV_PATH/bin/activate"

# Run the pipeline (CUDA on both proximal and score model)
cd "$HOME/mtf-aware-deblurring"

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

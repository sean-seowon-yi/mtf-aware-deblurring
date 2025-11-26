#!/usr/bin/env bash

# Sweep rho/denoiser-weight combinations for physics-aware ADMM + DRUNet.
# Defaults target DIV2K train bicubic X2, RGB, limit=3 images.
# Outputs: $OUT_BASE/rho<rho>-w<weight>/admm_metrics.csv (collect-only, no PNGs).

set -euo pipefail

DIV2K_ROOT="${DIV2K_ROOT:-$HOME/datasets/DIV2K}"
PATTERNS="${PATTERNS:-box random legendre}"
LIMIT="${LIMIT:-3}"
ADMM_ITERS="${ADMM_ITERS:-60}"

RHOS=(${RHOS:-0.85 0.90 0.95})
WEIGHTS=(${WEIGHTS:-0.15 0.20 0.25})

OUT_BASE="${OUT_BASE:-$HOME/mtf-smoke/phys-fine-sweep}"
mkdir -p "${OUT_BASE}"

common_flags=(
  --div2k-root "${DIV2K_ROOT}"
  --subset train --degradation bicubic --scale X2
  --image-mode rgb --limit "${LIMIT}" --auto-download
  --patterns ${PATTERNS}
  --method admm
  --denoiser-type drunet_color
  --denoiser-device cuda
  --admm-iters "${ADMM_ITERS}"
  --admm-mtf-weighting-mode combined
  --admm-mtf-sigma-adapt
  --use-physics-scheduler
  --collect-only --enable-ssim
)

for rho in "${RHOS[@]}"; do
  for w in "${WEIGHTS[@]}"; do
    out_dir="${OUT_BASE}/rho${rho}-w${w}"
    echo "=== Running rho=${rho}, weight=${w} -> ${out_dir}"
    python -m mtf_aware_deblurring.pipelines.reconstruct \
      "${common_flags[@]}" \
      --admm-rho "${rho}" \
      --admm-denoiser-weight "${w}" \
      --output-dir "${out_dir}"
  done
done

echo "Sweep complete. Metrics CSVs are under ${OUT_BASE}/rho*-w*/admm_metrics.csv"

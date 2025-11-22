#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/admm_sweep_modification.sh OUT_ROOT DIV2K_ROOT IMG_LIMIT
# Example:
#   bash scripts/admm_sweep_modification.sh "$HOME/mtf-smoke/admm-sweeps-mod50" data 50

OUT_ROOT="${1:-"$HOME/mtf-smoke/admm-sweeps-mod"}"
DIV2K_ROOT="${2:-"data"}"
IMG_LIMIT="${3:-5}"

mkdir -p "$OUT_ROOT"

# Hyperparameter grid
RHO_VALUES=(0.75 0.85 0.95 1.05 1.15)
WEIGHT_VALUES=(0.16 0.18 0.20 0.22)
# We keep only gamma here; combined was removed to avoid duplicate behavior.
MTF_MODES=(gamma)

run_once() {
  local rho="$1"
  local weight="$2"
  local mode="$3"

  local tag="rho${rho}_w${weight}_${mode}"
  local out_dir="${OUT_ROOT}/${tag}"

  echo "==> Running ${tag}"
  mkdir -p "$out_dir"

  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root "$DIV2K_ROOT" \
    --subset train \
    --degradation bicubic \
    --scale X2 \
    --image-mode rgb \
    --limit "$IMG_LIMIT" \
    --patterns box random legendre \
    --method admm \
    --denoiser-type drunet_color \
    --denoiser-device cuda \
    --admm-iters 60 \
    --admm-rho "$rho" \
    --admm-denoiser-weight "$weight" \
    --admm-mtf-weighting-mode "$mode" \
    --admm-mtf-sigma-adapt \
    --enable-ssim \
    --use-physics-scheduler \
    --collect-only \
    --output-dir "$out_dir"
}

for rho in "${RHO_VALUES[@]}"; do
  for weight in "${WEIGHT_VALUES[@]}"; do
    for mode in "${MTF_MODES[@]}"; do
      run_once "$rho" "$weight" "$mode"
    done
  done
done

# Aggregate per-run metrics into a single CSV.
OUT_ROOT="$OUT_ROOT" python - << 'EOF'
import csv
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])

rows = []
for run_dir in sorted(out_root.glob("rho*_w*_*")):
    metrics_path = run_dir / "admm_metrics.csv"
    if not metrics_path.is_file():
        continue
    with metrics_path.open() as f:
        run_rows = list(csv.DictReader(f))
    if not run_rows:
        continue

    by_pattern = {}
    for r in run_rows:
        by_pattern.setdefault(r["pattern"], []).append(r)

    def avg(pattern, key):
        vals = [float(rr[key]) for rr in by_pattern.get(pattern, [])]
        return sum(vals) / len(vals) if vals else float("nan")

    rows.append(
        {
            "tag": run_dir.name,
            "box_psnr": avg("box", "psnr"),
            "random_psnr": avg("random", "psnr"),
            "legendre_psnr": avg("legendre", "psnr"),
            "box_ssim": avg("box", "ssim"),
            "random_ssim": avg("random", "ssim"),
            "legendre_ssim": avg("legendre", "ssim"),
        }
    )

if rows:
    out_path = out_root / "sweep_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
else:
    print(f"No metrics found under {out_root}")
EOF

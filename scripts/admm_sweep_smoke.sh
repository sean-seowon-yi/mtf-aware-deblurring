#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/mtf-aware-deblurring}"
DIV2K_ROOT="${DIV2K_ROOT:-$HOME/datasets/DIV2K}"
OUT_ROOT="${OUT_ROOT:-$HOME/mtf-smoke/admm-sweeps}"
LIMIT="${LIMIT:-1}"
VENV="${VENV:-/tmp/$USER/envs/CSC2529}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/$USER/pip-cache}"

mkdir -p "$OUT_ROOT" "$PIP_CACHE_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
export PIP_CACHE_DIR

python - <<'PY' || pip install --no-cache-dir -e "$REPO" scipy >/dev/null
import importlib, sys
for pkg in ["numpy", "torch", "scipy"]:
    try:
        importlib.import_module(pkg)
    except Exception as e:
        print(f"MISSING {pkg}: {e}")
        sys.exit(1)
print("deps_ok")
PY

cd "$REPO"

COMBOS=(
  "0.40 0.90 1.50 0.10 60 on"
  "0.40 0.90 1.50 0.10 60 off"
)

run_one() {
  local rho="$1"; local w="$2"; local mtf_s="$3"; local mtf_f="$4"; local iters="$5"; local sched="$6"
  local tag="rho${rho}_w${w}_mtfs${mtf_s}_mtff${mtf_f}_it${iters}_sched${sched}"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"
  local extra_sched=""
  if [ "$sched" = "on" ]; then
    extra_sched="--use-physics-scheduler"
  fi
  echo "[RUN] $tag"
  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root "$DIV2K_ROOT" \
    --subset train \
    --degradation bicubic \
    --scale X2 \
    --image-mode rgb \
    --limit "$LIMIT" \
    --auto-download \
    --method admm \
    --admm-iters "$iters" \
    --admm-rho "$rho" \
    --admm-denoiser-weight "$w" \
    --admm-mtf-scale "$mtf_s" \
    --admm-mtf-floor "$mtf_f" \
    --denoiser-type drunet_color \
    --denoiser-device cuda \
    $extra_sched \
    --collect-only \
    --output-dir "$out_dir"
}

for combo in "${COMBOS[@]}"; do
  run_one $combo
done
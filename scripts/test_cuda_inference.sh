#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=test-cuda-infer
#SBATCH --output=logs/%x-%j.out

# Usage:
#   VENV_PATH=/path/to/venv sbatch scripts/test_cuda_inference.sh
# or set PYTHON to a specific interpreter before submission.

set -euo pipefail

mkdir -p logs

VENV_PATH="${VENV_PATH:-}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ -n "$VENV_PATH" ]]; then
    if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
        echo "Virtualenv not found at $VENV_PATH; set VENV_PATH to an existing env." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    PYTHON_BIN="python"
fi

$PYTHON_BIN - <<'PYCODE'
import torch

msg = [
    f"PyTorch version: {torch.__version__}",
    f"CUDA available: {torch.cuda.is_available()}",
]

if torch.cuda.is_available():
    device = torch.device("cuda")
    t = torch.randn(1, 3, 64, 64, device=device)
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
    with torch.inference_mode():
        out = conv(t)
    msg.append(f"Conv output shape: {tuple(out.shape)}")
    msg.append(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    msg.append("No CUDA device detected.")

print("\n".join(msg))
PYCODE

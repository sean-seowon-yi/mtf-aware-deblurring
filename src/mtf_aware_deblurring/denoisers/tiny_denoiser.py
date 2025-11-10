from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency path
    import torch
    from torch import nn
except ImportError:  # torch is optional unless denoiser baselines are used
    torch = None
    nn = None

from ..torch_utils import resolve_device


class TinyDenoiserNet(nn.Module if nn is not None else object):
    """Lightweight residual CNN denoiser trained on DIV2K patches."""

    def __init__(self, channels: int = 3, features: int = 64, depth: int = 8) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to instantiate TinyDenoiserNet.")
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch is required to run the TinyDenoiser.")
        return torch.clamp(x - self.body(x), 0.0, 1.0)


class TinyDenoiserWrapper:
    """NumPy-friendly wrapper for the PyTorch denoiser."""

    def __init__(self, model: TinyDenoiserNet, *, device: Optional[str] = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use TinyDenoiserWrapper.")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.noise_sigma: Optional[float] = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if torch is None:
            raise ImportError("PyTorch is required to use TinyDenoiserWrapper.")
        arr = np.asarray(image, dtype=np.float32)
        squeeze_channel = False
        if arr.ndim == 2:
            arr = arr[..., None]
            squeeze_channel = True
        reduce_to_one = False
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
            reduce_to_one = True
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            denoised = self.model(tensor)
        out = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        if reduce_to_one:
            out = out[..., :1]
        if squeeze_channel:
            out = out[..., 0]
        return np.clip(out.astype(np.float32), 0.0, 1.0)


def default_denoiser_weights() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "tiny_denoiser_sigma15.pth"


def build_tiny_denoiser(
    *,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> TinyDenoiserWrapper:
    if torch is None:
        raise ImportError("PyTorch is required to build the TinyDenoiser. Install torch>=2.0.")
    path = Path(weights_path) if weights_path is not None else default_denoiser_weights()
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find denoiser weights at {path}. Re-run scripts/train_tiny_denoiser.py to regenerate them."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = TinyDenoiserNet()
    model.load_state_dict(state_dict)
    wrapper = TinyDenoiserWrapper(model, device=device)
    if isinstance(checkpoint, dict) and "noise_sigma" in checkpoint:
        wrapper.noise_sigma = float(checkpoint["noise_sigma"])
    return wrapper


__all__ = [
    "TinyDenoiserNet",
    "TinyDenoiserWrapper",
    "build_tiny_denoiser",
    "default_denoiser_weights",
]

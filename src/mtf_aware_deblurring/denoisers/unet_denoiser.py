from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency path
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from ..torch_utils import resolve_device


def default_unet_denoiser_weights() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "unet_denoiser_sigma15.pth"


class ConvBlock(nn.Module if nn is not None else object):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for ConvBlock.")
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):  # type: ignore[override]
        return self.block(x)


class UpBlock(nn.Module if nn is not None else object):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for UpBlock.")
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.GELU(),
        )

    def forward(self, x):  # type: ignore[override]
        return self.up(x)


class UNetDenoiserNet(nn.Module if nn is not None else object):
    """Shallow UNet denoiser tailored for motion-blur priors."""

    def __init__(self, channels: int = 1, base_features: int = 64) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to instantiate UNetDenoiserNet.")
        super().__init__()
        feats = base_features
        self.enc1 = ConvBlock(channels, feats)
        self.down1 = nn.Conv2d(feats, feats * 2, kernel_size=2, stride=2)
        self.enc2 = ConvBlock(feats * 2, feats * 2)
        self.down2 = nn.Conv2d(feats * 2, feats * 4, kernel_size=2, stride=2)
        self.enc3 = ConvBlock(feats * 4, feats * 4)
        self.up1 = UpBlock(feats * 4, feats * 2)
        self.dec1 = ConvBlock(feats * 4, feats * 2)
        self.up2 = UpBlock(feats * 2, feats)
        self.dec2 = ConvBlock(feats * 2, feats)
        self.out_conv = nn.Conv2d(feats, channels, kernel_size=1)

    def forward(self, x):  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch is required to run UNetDenoiserNet.")
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d1 = self.up1(e3)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return torch.clamp(x - self.out_conv(d2), 0.0, 1.0)


class UNetDenoiserWrapper:
    """NumPy-friendly wrapper."""

    def __init__(self, model: UNetDenoiserNet, *, device: Optional[str] = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use UNetDenoiserWrapper.")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if torch is None:
            raise ImportError("PyTorch is required to use UNetDenoiserWrapper.")
        arr = np.asarray(image, dtype=np.float32)
        squeeze = False
        if arr.ndim == 2:
            arr = arr[..., None]
            squeeze = True
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            denoised = self.model(tensor)
        out = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        if squeeze:
            out = out[..., 0]
        return np.clip(out.astype(np.float32), 0.0, 1.0)


def build_unet_denoiser(
    *,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> UNetDenoiserWrapper:
    if torch is None:
        raise ImportError("PyTorch is required to build the UNet denoiser. Install torch>=2.0.")
    path = Path(weights_path) if weights_path is not None else default_unet_denoiser_weights()
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find UNet denoiser weights at {path}. Run scripts/train_unet_denoiser.py first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = UNetDenoiserNet(channels=1)
    model.load_state_dict(state_dict, strict=True)
    wrapper = UNetDenoiserWrapper(model, device=device)
    return wrapper


__all__ = [
    "UNetDenoiserNet",
    "UNetDenoiserWrapper",
    "build_unet_denoiser",
    "default_unet_denoiser_weights",
]

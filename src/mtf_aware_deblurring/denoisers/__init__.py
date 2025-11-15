from pathlib import Path
from typing import Literal, Optional

from .tiny_denoiser import (
    TinyDenoiserNet,
    TinyDenoiserWrapper,
    build_tiny_denoiser,
    default_denoiser_weights,
)
from .dncnn import (
    DnCNNDenoiserNet,
    DnCNNWrapper,
    build_dncnn_denoiser,
    default_dncnn_weights,
)
from .unet_denoiser import (
    UNetDenoiserNet,
    UNetDenoiserWrapper,
    build_unet_denoiser,
    default_unet_denoiser_weights,
)

# ✅ import the DRUNet adapter
from .drunet_adapter import build_drunet_denoiser

__all__ = [
    "TinyDenoiserNet",
    "TinyDenoiserWrapper",
    "build_tiny_denoiser",
    "default_denoiser_weights",
    "DnCNNDenoiserNet",
    "DnCNNWrapper",
    "build_dncnn_denoiser",
    "default_dncnn_weights",
    "UNetDenoiserNet",
    "UNetDenoiserWrapper",
    "build_unet_denoiser",
    "default_unet_denoiser_weights",
    "build_denoiser",
    "build_drunet_denoiser",
]


def build_denoiser(
    denoiser_type: Literal[
        "tiny",
        "dncnn",
        "unet",
        "drunet_color",
        "drunet_gray",
    ] = "tiny",
    *,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
):
    # Existing ones
    if denoiser_type == "tiny":
        return build_tiny_denoiser(weights_path=weights_path, device=device)
    if denoiser_type == "dncnn":
        return build_dncnn_denoiser(weights_path=weights_path, device=device)
    if denoiser_type == "unet":
        return build_unet_denoiser(weights_path=weights_path, device=device)

    # New: DRUNet color / gray
    if denoiser_type in ("drunet_color", "drunet_gray"):
        mode = "color" if denoiser_type == "drunet_color" else "gray"

        # weights_path is optional override; if None, adapter auto-downloads from HF
        return build_drunet_denoiser(
            mode=mode,
            device=device or "cuda",  # default to CUDA if available
            sigma=25.0,               # you can expose this later if you want
        )

    raise ValueError(
        f"Unsupported denoiser_type '{denoiser_type}'. "
        "Expected one of: 'tiny', 'dncnn', 'unet', 'drunet_color', 'drunet_gray'."
    )

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
]


def build_denoiser(
    denoiser_type: Literal["tiny", "dncnn", "unet"] = "tiny",
    *,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
):
    if denoiser_type == "tiny":
        return build_tiny_denoiser(weights_path=weights_path, device=device)
    if denoiser_type == "dncnn":
        return build_dncnn_denoiser(weights_path=weights_path, device=device)
    if denoiser_type == "unet":
        return build_unet_denoiser(weights_path=weights_path, device=device)
    raise ValueError(f"Unsupported denoiser_type '{denoiser_type}'. Expected 'tiny' or 'dncnn'.")

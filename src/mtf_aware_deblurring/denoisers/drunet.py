# ----------------------------------------------------------------------
# DRUNet wrapper for mtf_aware_deblurring
#
# Architecture backbone:
#   UNetRes (Residual U-Net) adapted from
#   “Plug-and-Play Image Restoration with Deep Denoiser Prior” (Zhang et al., 2021)
#   Original DPIR repository: https://github.com/cszn/DPIR
#   License: MIT License (see MIT-LICENSE file in DPIR)
#
# This wrapper:
#   - adds noise-level conditioning (extra input channel)
#   - loads pretrained weights (DPIR model_zoo)
#   - exposes a clean (x, sigma) → denoised interface
# ----------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Union

import torch
import torch.nn as nn

from .drunet_backbone import UNetRes


Number = Union[float, int]
TensorOrNumber = Union[torch.Tensor, Number]


class DRUNet(nn.Module):
    """
    DRUNet denoiser with noise-level conditioning.

    Expected usage:
        model = DRUNet(
            n_channels=3,
            default_sigma=25.0,
            weight_path="path/to/drunet_color.pth",
            device="cuda",
        )

        x_denoised = model(x_noisy, sigma=15.0)  # sigma in [0, 255] convention

    Args
    ----
    n_channels: int
        Number of image channels (1 = grayscale, 3 = RGB).
    default_sigma: float
        Default noise level (in [0, 255]) to use when sigma is not provided at forward().
    nc, nb, act_mode, downsample_mode, upsample_mode:
        Passed directly to UNetRes backbone. Should match the config used
        when the pretrained weights were trained (DPIR uses the defaults here).
    weight_path: optional str
        Path to a DPIR DRUNet checkpoint (.pth). If provided, it will be loaded.
    device: str or torch.device
        Device to put the model on ("cpu", "cuda", etc.).
    strict: bool
        Passed to load_state_dict. Set to False if you expect minor key mismatches.
    """

    def __init__(
        self,
        n_channels: int = 3,
        default_sigma: float = 25.0,
        nc = [64, 128, 256, 512],
        nb: int = 4,
        act_mode: str = "R",
        downsample_mode: str = "strideconv",
        upsample_mode: str = "convtranspose",
        weight_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.default_sigma = float(default_sigma)
        self.device = torch.device(device)

        # +1 channel for noise map
        self.backbone = UNetRes(
            in_nc=n_channels + 1,
            out_nc=n_channels,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )

        # Optionally load pretrained weights
        if weight_path is not None:
            self._load_pretrained_weights(weight_path, strict=strict)

        # Move to device and eval by default (denoiser)
        self.to(self.device)
        self.eval()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_pretrained_weights(self, weight_path: str, strict: bool = True) -> None:
        ckpt = torch.load(weight_path, map_location=self.device)

        # Handle possible nesting (e.g., {"state_dict": ..., "epoch": ...})
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt

        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        if missing:
            print(f"[DRUNet] Warning: missing keys when loading weights: {missing}")
        if unexpected:
            print(f"[DRUNet] Warning: unexpected keys when loading weights: {unexpected}")

    # ------------------------------------------------------------------
    # Noise map handling
    # ------------------------------------------------------------------
    def _build_sigma_map(self, x: torch.Tensor, sigma: Optional[TensorOrNumber]) -> torch.Tensor:
        """
        Build a (B, 1, H, W) noise map tensor given x and sigma.

        sigma can be:
          - None         → use self.default_sigma
          - scalar (int/float)
          - tensor of shape (B,), (B,1), or (B,1,1,1)
          - full map of shape (B,1,H,W)
        All values are in [0, 255]; internally normalized to [0,1] like DPIR.
        """
        B, _, H, W = x.shape

        if sigma is None:
            sigma_val = self.default_sigma
            sigma_tensor = torch.full((B, 1, 1, 1), float(sigma_val), device=x.device, dtype=x.dtype)
        elif isinstance(sigma, (int, float)):
            sigma_tensor = torch.full((B, 1, 1, 1), float(sigma), device=x.device, dtype=x.dtype)
        elif isinstance(sigma, torch.Tensor):
            # try to broadcast/fix shapes
            if sigma.ndim == 0:
                sigma_tensor = sigma.view(1, 1, 1, 1).expand(B, 1, 1, 1)
            elif sigma.ndim == 1:
                # (B,) → (B,1,1,1)
                sigma_tensor = sigma.view(B, 1, 1, 1)
            elif sigma.ndim == 2:
                # (B,1) → (B,1,1,1)
                sigma_tensor = sigma.view(B, 1, 1, 1)
            elif sigma.ndim == 4:
                # Assume already (B,1,H,W)
                sigma_tensor = sigma
            else:
                raise ValueError(f"Unsupported sigma shape: {tuple(sigma.shape)}")
        else:
            raise TypeError(f"Unsupported sigma type: {type(sigma)}")

        # Normalize by 255, as in DPIR (noise level in [0, 255])
        sigma_map = sigma_tensor / 255.0

        # Broadcast to spatial size if needed
        if sigma_map.shape[-2:] != (H, W):
            sigma_map = sigma_map.expand(B, 1, H, W)

        return sigma_map

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor, sigma: Optional[TensorOrNumber] = None) -> torch.Tensor:
        """
        Denoise x using DRUNet.

        Args
        ----
        x : (B, C, H, W) tensor
            Noisy input image(s) in [0, 1] or [0, 255] (you decide; be consistent
            with how DRUNet was trained).
        sigma : optional scalar or tensor
            Noise level in [0, 255]. If None, self.default_sigma is used.

        Returns
        -------
        (B, C, H, W) tensor
            Denoised image(s).
        """
        x = x.to(self.device)
        sigma_map = self._build_sigma_map(x, sigma)
        x_in = torch.cat([x, sigma_map], dim=1)  # (B, C+1, H, W)
        return self.backbone(x_in)


# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------
def load_pretrained_drunet_color(
    weight_path: str,
    default_sigma: float = 25.0,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> DRUNet:
    """
    Convenience function for RGB DRUNet (3-channel) using pretrained color weights.
    """
    return DRUNet(
        n_channels=3,
        default_sigma=default_sigma,
        weight_path=weight_path,
        device=device,
        **kwargs,
    )


def load_pretrained_drunet_gray(
    weight_path: str,
    default_sigma: float = 25.0,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> DRUNet:
    """
    Convenience function for grayscale DRUNet (1-channel) using pretrained gray weights.
    """
    return DRUNet(
        n_channels=1,
        default_sigma=default_sigma,
        weight_path=weight_path,
        device=device,
        **kwargs,
    )


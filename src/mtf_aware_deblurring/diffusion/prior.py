from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .score_model import DEFAULT_TINY_SCORE_WEIGHTS, TinyScoreUNet, build_tiny_score_model
from ..torch_utils import resolve_device

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch optional
    torch = None
    nn = None


@dataclass
class DiffusionPriorConfig:
    """Configuration describing the diffusion-prior proximal operator."""

    steps: int = 12
    sigma_min: float = 0.01
    sigma_max: float = 0.5
    schedule: Literal["geom", "linear"] = "geom"
    guidance: float = 1.0
    noise_scale: float = 1.0
    clamp_min: float = 0.0
    clamp_max: float = 1.0


def _build_sigma_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    mode: Literal["geom", "linear"],
) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive for the diffusion schedule.")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min/max must be positive.")
    if sigma_min > sigma_max:
        sigma_min, sigma_max = sigma_max, sigma_min
    if mode == "geom":
        schedule = np.geomspace(sigma_max, sigma_min, steps, dtype=np.float32)
    elif mode == "linear":
        schedule = np.linspace(sigma_max, sigma_min, steps, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported schedule '{mode}'.")
    return schedule


def _numpy_to_tensor(image: np.ndarray, device: torch.device) -> tuple[torch.Tensor, bool]:
    arr = np.asarray(image, dtype=np.float32)
    squeeze_channel = False
    if arr.ndim == 2:
        arr = arr[..., None]
        squeeze_channel = True
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, squeeze_channel


def _tensor_to_numpy(tensor: torch.Tensor, squeeze_channel: bool) -> np.ndarray:
    arr = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    if squeeze_channel:
        arr = arr[..., 0]
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


class DiffusionPrior:
    """Implements the diffusion-based proximal operator for ADMM-PnP."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DiffusionPriorConfig] = None,
        *,
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DiffusionPrior.")
        if config is None:
            config = DiffusionPriorConfig()
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        self.schedule = torch.from_numpy(
            _build_sigma_schedule(config.steps, config.sigma_min, config.sigma_max, config.schedule)
        ).to(self.device)

    def update_schedule(self, steps: Optional[int] = None) -> None:
        cfg = self.config
        if steps is not None:
            cfg.steps = int(steps)
        self.schedule = torch.from_numpy(
            _build_sigma_schedule(cfg.steps, cfg.sigma_min, cfg.sigma_max, cfg.schedule)
        ).to(self.device)

    @torch.inference_mode()
    def proximal(
        self,
        reference: np.ndarray,
        rho: float,
        *,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> np.ndarray:
        if torch is None:
            raise ImportError("PyTorch is required to run DiffusionPrior.")
        if rho <= 0:
            raise ValueError("rho must be positive.")
        if steps is not None and steps != self.config.steps:
            self.update_schedule(steps)
        cfg = self.config
        guidance = float(guidance if guidance is not None else cfg.guidance)
        noise_scale = float(noise_scale if noise_scale is not None else cfg.noise_scale)
        sigmas = self.schedule
        tensor, squeeze_channel = _numpy_to_tensor(reference, self.device)
        z = tensor.clone()
        if noise_scale > 0:
            z = z + noise_scale * sigmas[0] * torch.randn_like(z)

        for sigma in sigmas:
            sigma_batch = torch.full((z.shape[0],), sigma.item(), device=self.device)
            score = self.model(z, sigma_batch)
            alpha = guidance * sigma.item() ** 2
            grad = score - rho * (z - tensor)
            z = z + alpha * grad
            z = torch.clamp(z, cfg.clamp_min, cfg.clamp_max)
        return _tensor_to_numpy(z, squeeze_channel)


def build_diffusion_prior(
    prior_type: Literal["tiny_score"] = "tiny_score",
    *,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
    config: Optional[DiffusionPriorConfig] = None,
    in_channels: int = 3,
) -> DiffusionPrior:
    if prior_type == "tiny_score":
        model = build_tiny_score_model(weights_path=weights_path, in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported diffusion prior '{prior_type}'.")
    return DiffusionPrior(model=model, config=config, device=device)


__all__ = [
    "DiffusionPrior",
    "DiffusionPriorConfig",
    "build_diffusion_prior",
]

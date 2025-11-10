from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional
    torch = None
    nn = None
    F = None

DEFAULT_TINY_SCORE_WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "tiny_score_unet.pth"


def _default_weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SinusoidalTimeEmbedding(nn.Module if nn is not None else object):
    """Standard sinusoidal embedding used for noise levels."""

    def __init__(self, embedding_dim: int = 64) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for SinusoidalTimeEmbedding.")
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")
        self.embedding_dim = embedding_dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch is required to run TinyScoreUNet.")
        sigma = sigma.float().view(-1, 1)
        half = self.embedding_dim // 2
        device = sigma.device
        freq = torch.exp(torch.linspace(0, math.log(10_000.0), half, device=device))
        angles = sigma * freq
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module if nn is not None else object):
    """Residual block with FiLM-style time conditioning."""

    def __init__(self, channels: int, embedding_dim: int) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for ResidualBlock.")
        super().__init__()
        groups = max(1, channels // 8)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(embedding_dim, channels)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch is required to run TinyScoreUNet.")
        h = self.conv1(self.act(self.norm1(x)))
        shift = self.time_proj(embedding).unsqueeze(-1).unsqueeze(-1)
        h = h + shift
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class TinyScoreUNet(nn.Module if nn is not None else object):
    """Compact UNet-like score model for spatially small diffusion priors."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 6,
        embedding_dim: int = 64,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to instantiate TinyScoreUNet.")
        super().__init__()
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.embedding = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(base_channels, embedding_dim) for _ in range(depth)
        )
        groups = max(1, base_channels // 8)
        self.out_norm = nn.GroupNorm(groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.apply(_default_weight_init)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch is required to run TinyScoreUNet.")
        if sigma.ndim == 0:
            sigma = sigma[None]
        if sigma.ndim == 2 and sigma.shape[1] != 1:
            sigma = sigma[:, :1]
        sigma = sigma.view(x.shape[0], -1)
        emb = self.embedding(sigma.squeeze(-1))
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h, emb)
        h = self.conv_out(F.silu(self.out_norm(h)))
        return h


def build_tiny_score_model(
    *,
    weights_path: Optional[Path] = None,
    in_channels: int = 3,
) -> TinyScoreUNet:
    if torch is None:
        raise ImportError("PyTorch is required to build the TinyScoreUNet.")
    model = TinyScoreUNet(in_channels=in_channels)
    path = Path(weights_path) if weights_path is not None else DEFAULT_TINY_SCORE_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find TinyScoreUNet weights at {path}. "
            "Train a diffusion score model (see scripts/train_tiny_score_model.py) or provide --diffusion-prior-weights."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


__all__ = [
    "TinyScoreUNet",
    "build_tiny_score_model",
    "DEFAULT_TINY_SCORE_WEIGHTS",
]

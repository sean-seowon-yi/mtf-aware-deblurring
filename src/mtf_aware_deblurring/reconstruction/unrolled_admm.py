from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..torch_utils import resolve_device


def _logit(x: float, eps: float = 1e-6) -> float:
    clipped = min(max(x, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _pad_to_shape_2d(kernel: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Center-pad a 2D kernel (or batch of kernels) to a target shape."""
    k_h, k_w = kernel.shape[-2:]
    t_h, t_w = shape
    if k_h > t_h or k_w > t_w:
        raise ValueError(f"Kernel {kernel.shape[-2:]} larger than target {shape}.")
    pad_h = t_h - k_h
    pad_w = t_w - k_w
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(kernel, pad)


@dataclass
class UnrolledADMMConfig:
    """Configuration for the learnable unrolled ADMM module."""

    steps: int = 8
    rho_init: float = 0.4
    denoiser_weight_init: float = 1.0
    sigma_multiplier_init: float = 1.0
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    denoise_every: int = 1
    learn_mtf_mask: bool = True
    mtf_gamma_init: float = 1.5
    mtf_floor_init: float = 0.1
    mtf_cutoff_init: float = 0.01
    mtf_cutoff_max: float = 0.2
    eps: float = 1e-6


class ContextParamHead(nn.Module):
    """
    Small MLP that maps physics/context features to per-step parameter deltas.

    Outputs shape: (batch, steps, channels).
    """

    def __init__(self, *, in_dim: int, steps: int, out_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.steps = steps
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, steps * out_channels),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        x = self.net(ctx)
        return x.view(-1, self.steps, self.out_channels)


class MaskParamHead(nn.Module):
    """Maps context features to mask parameter deltas (gamma, floor, cutoff)."""

    def __init__(self, in_dim: int, hidden: int = 64, out_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


class UnrolledADMM(nn.Module):
    """
    Torch-native unrolled ADMM/PnP module with learnable physics-aware parameters.

    - Learns per-iteration rho, denoiser weight, and sigma multiplier (base params + context deltas).
    - Optionally learns MTF trust-mask parameters (gamma, floor, cutoff) with context conditioning.
    - Uses a provided torch denoiser (expects signature denoiser(x, sigma=None) or denoiser(x)).
    """

    def __init__(
        self,
        denoiser: nn.Module,
        config: Optional[UnrolledADMMConfig] = None,
        *,
        channels: int = 1,
        context_dim: int = 0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = UnrolledADMMConfig()
        if config.steps < 1:
            raise ValueError("steps must be positive.")
        self.cfg = config
        self.channels = channels
        self.device = resolve_device(device) if device is not None else None
        self.denoiser = denoiser
        if self.device is not None:
            self.denoiser.to(self.device)

        steps = config.steps
        self.log_rho = nn.Parameter(torch.full((steps,), math.log(config.rho_init)))
        self.weight_logits = nn.Parameter(
            torch.full((steps,), _logit(config.denoiser_weight_init))
        )
        self.log_sigma = nn.Parameter(torch.full((steps,), math.log(config.sigma_multiplier_init)))
        self.denoise_logits = nn.Parameter(torch.full((steps,), 4.0))  # ~0.982 prob

        if config.learn_mtf_mask:
            self.mask_gamma = nn.Parameter(torch.tensor(config.mtf_gamma_init))
            self.mask_floor = nn.Parameter(torch.tensor(_logit(config.mtf_floor_init)))
            self.mask_cutoff = nn.Parameter(torch.tensor(_logit(config.mtf_cutoff_init)))
        else:
            self.register_parameter("mask_gamma", None)
            self.register_parameter("mask_floor", None)
            self.register_parameter("mask_cutoff", None)

        self.step_head: Optional[ContextParamHead] = None
        self.mask_head: Optional[MaskParamHead] = None
        if context_dim > 0:
            self.step_head = ContextParamHead(in_dim=context_dim, steps=steps, out_channels=3)
            if config.learn_mtf_mask:
                self.mask_head = MaskParamHead(in_dim=context_dim)

    def _prepare_otf(self, kernel: torch.Tensor, shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = _pad_to_shape_2d(kernel, shape)
        shifted = torch.fft.ifftshift(padded, dim=(-2, -1))
        otf = torch.fft.fft2(shifted)
        return otf, torch.conj(otf)

    def _normalize_mtf(self, mtf: torch.Tensor, eps: float) -> torch.Tensor:
        max_val = mtf.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (mtf / max_val).clamp(0.0, 1.0)

    def _build_trust_mask(
        self,
        mtf: torch.Tensor,
        gamma: torch.Tensor,
        floor: torch.Tensor,
        cutoff: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mtf_norm = self._normalize_mtf(mtf, eps)
        gamma_expanded = gamma.view(-1, 1, 1, 1)
        floor_expanded = floor.view(-1, 1, 1, 1)
        cutoff_expanded = cutoff.view(-1, 1, 1, 1)

        w = mtf_norm.pow(gamma_expanded)
        if cutoff_expanded.gt(0).any():
            w = torch.where(mtf_norm < cutoff_expanded, torch.zeros_like(w), w)
        mask_pos = w > 0
        w = torch.where(
            mask_pos,
            torch.clamp(w, min=floor_expanded, max=torch.ones_like(w)),
            w,
        )
        return w

    def _call_denoiser(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if hasattr(self.denoiser, "__call__"):
            try:
                return self.denoiser(x, sigma=sigma)
            except TypeError:
                return self.denoiser(x)
        raise TypeError("Provided denoiser is not callable.")

    def _resolve_params(
        self,
        batch_size: int,
        *,
        context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Combine base parameters with optional context deltas."""
        cfg = self.cfg
        if context is not None and self.step_head is not None:
            deltas = self.step_head(context)  # (B, steps, 3)
            deltas = torch.clamp(deltas, min=-5.0, max=5.0)
        else:
            deltas = None

        base_log_rho = self.log_rho.view(1, -1).expand(batch_size, -1)
        base_weight = self.weight_logits.view(1, -1).expand(batch_size, -1)
        base_log_sigma = self.log_sigma.view(1, -1).expand(batch_size, -1)
        base_denoise = self.denoise_logits.view(1, -1).expand(batch_size, -1)

        if deltas is not None:
            log_rho = base_log_rho + deltas[:, :, 0]
            weight_logits = base_weight + deltas[:, :, 1]
            log_sigma = base_log_sigma + deltas[:, :, 2]
        else:
            log_rho = base_log_rho
            weight_logits = base_weight
            log_sigma = base_log_sigma

        # Clamp to sane ranges to avoid overflow/NaNs from aggressive context.
        log_rho = torch.clamp(log_rho, min=math.log(1e-4), max=math.log(5.0))
        log_sigma = torch.clamp(log_sigma, min=math.log(1e-3), max=math.log(3.0))
        weight_logits = torch.clamp(weight_logits, min=-6.0, max=6.0)

        rho = log_rho.exp().clamp(min=cfg.eps)
        weight = torch.sigmoid(weight_logits)
        sigma_mult = log_sigma.exp().clamp(min=cfg.eps)
        denoise_prob = torch.sigmoid(base_denoise)

        mask_gamma: Optional[torch.Tensor] = None
        mask_floor: Optional[torch.Tensor] = None
        mask_cutoff: Optional[torch.Tensor] = None
        if cfg.learn_mtf_mask:
            if context is not None and self.mask_head is not None:
                mask_delta = self.mask_head(context)
            else:
                mask_delta = None
            gamma_base = self.mask_gamma
            floor_base = self.mask_floor
            cutoff_base = self.mask_cutoff
            if mask_delta is not None:
                gamma = F.softplus(gamma_base + mask_delta[:, 0]) + cfg.eps
                floor = torch.sigmoid(floor_base + mask_delta[:, 1])
                cutoff = torch.sigmoid(cutoff_base + mask_delta[:, 2]) * cfg.mtf_cutoff_max
            else:
                gamma = F.softplus(gamma_base) + cfg.eps
                floor = torch.sigmoid(floor_base)
                cutoff = torch.sigmoid(cutoff_base) * cfg.mtf_cutoff_max
            # Clamp mask params to stable ranges
            gamma = torch.clamp(gamma, min=0.1, max=6.0)
            floor = torch.clamp(floor, min=0.0, max=0.8)
            cutoff = torch.clamp(cutoff, min=0.0, max=cfg.mtf_cutoff_max)
            mask_gamma = gamma
            mask_floor = floor
            mask_cutoff = cutoff

        return rho, weight, sigma_mult, mask_gamma, mask_floor, mask_cutoff

    def forward(
        self,
        observation: torch.Tensor,
        kernel: torch.Tensor,
        *,
        mtf: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list[Dict[str, Any]]]]:
        """
        Args:
            observation: (B, C, H, W) noisy/blurred image.
            kernel: (B, 1, kH, kW) or (B, kH, kW) motion PSF kernel.
            mtf: optional precomputed MTF map (B, 1, H, W) or (B, H, W).
            context: optional physics feature tensor (B, context_dim).
            return_trace: whether to return per-iteration diagnostics.
        """
        cfg = self.cfg
        eps = cfg.eps

        y = observation
        if y.dim() == 3:
            y = y.unsqueeze(0)
        if y.dim() != 4:
            raise ValueError("observation must have shape (B, C, H, W).")

        if kernel.dim() == 3:
            kernel = kernel.unsqueeze(1)
        if kernel.dim() != 4:
            raise ValueError("kernel must have shape (B, 1, kH, kW) or (B, kH, kW).")

        if mtf is not None and mtf.dim() == 3:
            mtf = mtf.unsqueeze(1)

        b, c, h, w = y.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}.")

        if self.device is not None:
            y = y.to(self.device)
            kernel = kernel.to(self.device)
            if mtf is not None:
                mtf = mtf.to(self.device)
            if context is not None:
                context = context.to(self.device)

        rho, weight, sigma_mult, mask_gamma, mask_floor, mask_cutoff = self._resolve_params(
            b, context=context
        )

        otf_list = []
        otf_conj_list = []
        for i in range(b):
            otf_i, otf_conj_i = self._prepare_otf(kernel[i, 0], (h, w))
            otf_list.append(otf_i)
            otf_conj_list.append(otf_conj_i)
        otf = torch.stack(otf_list, dim=0)
        otf_conj = torch.stack(otf_conj_list, dim=0)
        otf_abs2 = (otf * otf_conj).real

        otf = otf.unsqueeze(1)  # (B, 1, H, W)
        otf_conj = otf_conj.unsqueeze(1)
        otf_abs2 = otf_abs2.unsqueeze(1)

        weight_map: Optional[torch.Tensor] = None
        if cfg.learn_mtf_mask and mtf is not None and mask_gamma is not None:
            mtf = mtf if mtf.dim() == 4 else mtf.unsqueeze(1)
            weight_map = self._build_trust_mask(mtf, mask_gamma, mask_floor, mask_cutoff, eps)

        y_fft = torch.fft.fft2(y)
        x = y.clone()
        z = y.clone()
        u = torch.zeros_like(y)
        trace: list[Dict[str, Any]] = []

        denoise_every = max(int(cfg.denoise_every), 1)

        for t in range(cfg.steps):
            rho_t = rho[:, t].view(b, 1, 1, 1)
            weight_t = weight[:, t].view(b, 1, 1, 1)
            sigma_t = sigma_mult[:, t].view(b, 1, 1, 1)
            denoise_p = torch.sigmoid(self.denoise_logits[t]).view(1, 1, 1, 1)

            rhs_prior = rho_t * torch.fft.fft2(z - u)
            if weight_map is not None:
                numerator = (otf_conj * weight_map) * y_fft + rhs_prior
                denom = (otf_abs2 * weight_map) + rho_t
            else:
                numerator = otf_conj * y_fft + rhs_prior
                denom = otf_abs2 + rho_t
            x = torch.fft.ifft2(numerator / denom).real

            perform_denoise = ((t % denoise_every) == 0) or (t == cfg.steps - 1)
            v = x + u
            if perform_denoise:
                theoretical_sigma = torch.rsqrt(rho_t.clamp_min(eps))
                effective_sigma = theoretical_sigma * sigma_t
                z_denoised = self._call_denoiser(v, sigma=effective_sigma)
                z_hat = (1.0 - weight_t) * v + weight_t * z_denoised
                if denoise_p < 1.0:
                    z_hat = denoise_p * z_hat + (1.0 - denoise_p) * v
            else:
                effective_sigma = torch.zeros_like(rho_t)
                z_hat = v

            z = torch.clamp(z_hat, cfg.clamp_min, cfg.clamp_max)
            u = u + x - z

            if return_trace:
                x_flat = x.view(b, -1)
                z_flat = z.view(b, -1)
                primal = (x_flat - z_flat).norm(dim=1) / (x_flat.norm(dim=1) + eps)
                dual = (z_flat - z_hat.view(b, -1)).norm(dim=1) / (
                    z_hat.view(b, -1).norm(dim=1) + eps
                )
                trace.append(
                    {
                        "iteration": t + 1,
                        "rho": rho_t.mean().item(),
                        "weight": weight_t.mean().item(),
                        "sigma_mult": sigma_t.mean().item(),
                        "primal_res": primal.mean().item(),
                        "dual_res": dual.mean().item(),
                        "denoised": perform_denoise,
                    }
                )

        return z, trace if return_trace else None


__all__ = ["UnrolledADMMConfig", "UnrolledADMM"]

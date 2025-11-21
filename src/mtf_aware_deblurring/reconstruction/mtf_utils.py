from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.fft import fft2, ifftshift

from ..optics import pad_to_shape


def compute_mtf2(kernel: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Compute |H|^2 for a kernel on the given image grid."""
    kernel = np.asarray(kernel, dtype=np.float32)
    padded = pad_to_shape(kernel, image_shape)
    otf = fft2(ifftshift(padded))
    return (np.abs(otf) ** 2).astype(np.float32)


def infer_tau_from_observation(
    obs: np.ndarray,
    tau_min: float = 1e-4,
    tau_max: float = 1e-1,
) -> float:
    """Estimate a noise-to-signal power ratio tau from the observation via MAD."""
    obs_f = np.asarray(obs, dtype=np.float32)
    if obs_f.size == 0:
        return tau_min
    if obs_f.ndim == 3:
        obs_f = np.mean(obs_f, axis=-1)
    dx = obs_f[:, 1:] - obs_f[:, :-1]
    dy = obs_f[1:, :] - obs_f[:-1, :]
    diffs = np.concatenate([dx.ravel(), dy.ravel()])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return tau_min
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med))
    if mad <= 0 or not np.isfinite(mad):
        return tau_min
    sigma_n = mad / 0.6745
    var_y = float(np.var(obs_f))
    if not np.isfinite(var_y) or var_y <= 0.0:
        return tau_min
    var_n = sigma_n**2
    var_x = max(var_y - var_n, 1e-6)
    tau = var_n / var_x
    return float(np.clip(tau, tau_min, tau_max))


def build_mtf_weights(
    mtf2: np.ndarray,
    tau: float = 1e-4,
    alpha: float = 0.5,
    floor: float = 0.05,
) -> np.ndarray:
    """
    Build per-frequency weights from MTF^2 using a Wiener-like formulation.

    W0 = |H|^2 / (|H|^2 + tau), optionally compressed by alpha and floored.
    """
    mtf2 = np.asarray(mtf2, dtype=np.float32)
    if mtf2.size == 0:
        return mtf2
    mtf2_clipped = np.clip(mtf2, 0.0, None)
    mtf2_max = float(mtf2_clipped.max())
    if not np.isfinite(mtf2_max) or mtf2_max <= 0.0:
        return np.ones_like(mtf2_clipped, dtype=np.float32)
    tau_eff = max(float(tau), 1e-12)
    W0 = mtf2_clipped / (mtf2_clipped + tau_eff)
    if alpha != 1.0:
        weights = np.power(W0, alpha, dtype=np.float32)
    else:
        weights = W0.astype(np.float32)
    return np.clip(weights, floor, 1.0, out=weights)


def mtf_quality_from_kernel(
    kernel: np.ndarray,
    image_shape: tuple[int, int],
    tau: float = 1e-3,
) -> float:
    """
    Compute a scalar MTF quality score ~[0,1], focusing on mid-high frequencies.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    if H <= 0 or W <= 0:
        return 0.0
    mtf2 = compute_mtf2(kernel, (H, W))
    mtf2 = np.clip(mtf2, 0.0, None)
    mtf2_max = float(mtf2.max())
    if not np.isfinite(mtf2_max) or mtf2_max <= 0.0:
        return 0.0
    mtf2_norm = mtf2 / (mtf2_max + 1e-8)
    fy = np.fft.fftfreq(H, d=1.0)
    fx = np.fft.fftfreq(W, d=1.0)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    rad = np.sqrt(FX**2 + FY**2)
    rad_max = float(rad.max())
    if not np.isfinite(rad_max) or rad_max <= 0.0:
        return 0.0
    rad_norm = rad / rad_max
    band_mask = (rad_norm >= 0.2) & (rad_norm <= 0.8)
    mtf_mask = mtf2_norm > tau
    mask = band_mask & mtf_mask
    if mask.any():
        q = float(np.mean(mtf2_norm[mask]))
    elif mtf_mask.any():
        q = float(np.mean(mtf2_norm[mtf_mask]))
    else:
        q = 0.0
    if not np.isfinite(q):
        return 0.0
    return float(np.clip(q, 0.0, 1.0))


def mtf_adaptive_sigma_bounds(Q: float) -> Tuple[float, float]:
    """
    Map MTF quality Q to (sigma_max, sigma_min) for DRUNet / score priors.
    """
    SIGMA_MAX_LO, SIGMA_MAX_HI = 25.0, 40.0
    SIGMA_MIN_LO, SIGMA_MIN_HI = 8.0, 20.0
    Q = float(np.clip(Q, 0.0, 1.0))
    beta = 0.7
    Q_eff = Q**beta
    t = 1.0 - Q_eff
    sigma_max = SIGMA_MAX_LO + t * (SIGMA_MAX_HI - SIGMA_MAX_LO)
    sigma_min = SIGMA_MIN_LO + t * (SIGMA_MIN_HI - SIGMA_MIN_LO)
    return float(sigma_max), float(sigma_min)


__all__ = [
    "compute_mtf2",
    "infer_tau_from_observation",
    "build_mtf_weights",
    "mtf_quality_from_kernel",
    "mtf_adaptive_sigma_bounds",
]

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter

from ..metrics import psnr, ssim
from ..optics import fft_convolve2d
from .results import ReconstructionResult


def richardson_lucy(
    image: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 25,
    clip: bool = True,
    eps: float = 1e-8,
    damping: float = 0.9,
    tv_weight: float = 0.0,
    smooth_weight: float = 0.1,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Richardson-Lucy with optional damping, TV, and Gaussian smoothing."""
    kernel = np.clip(kernel, 0, None)
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        raise ValueError("Kernel must have positive sum for Richardson-Lucy.")
    kernel = kernel / kernel_sum
    kernel_mirror = np.flip(np.flip(kernel, axis=0), axis=1)

    def _total_variation(channel: np.ndarray) -> np.ndarray:
        grad_y = np.gradient(channel, axis=0)
        grad_x = np.gradient(channel, axis=1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + eps
        div_y = np.gradient(grad_y / magnitude, axis=0)
        div_x = np.gradient(grad_x / magnitude, axis=1)
        return div_y + div_x

    def _postprocess(channel: np.ndarray) -> np.ndarray:
        if tv_weight > 0:
            channel -= tv_weight * _total_variation(channel)
        if smooth_weight > 0:
            smoothed = gaussian_filter(channel, sigma=smooth_sigma)
            channel = (1.0 - smooth_weight) * channel + smooth_weight * smoothed
        if clip:
            channel = np.clip(channel, 0.0, 1.0)
        return channel

    def _deconv(channel: np.ndarray) -> np.ndarray:
        observed = np.clip(channel, eps, 1.0).astype(np.float32)
        estimate = observed.copy()
        for _ in range(iterations):
            conv_est = fft_convolve2d(estimate, kernel) + eps
            relative_blur = observed / conv_est
            correction = fft_convolve2d(relative_blur, kernel_mirror)
            estimate *= correction**damping
            estimate = _postprocess(estimate)
        return estimate

    if image.ndim == 2:
        return _deconv(image)
    channels = [_deconv(image[..., c]) for c in range(image.shape[2])]
    result = np.stack(channels, axis=-1)
    if clip:
        result = np.clip(result, 0.0, 1.0)
    return result


def run_richardson_lucy_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    iterations: int = 25,
    damping: float = 0.9,
    tv_weight: float = 0.0,
    smooth_weight: float = 0.1,
    smooth_sigma: float = 1.0,
) -> Dict[str, ReconstructionResult]:
    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        recon = richardson_lucy(
            data["noisy"],
            data["kernel"],
            iterations=iterations,
            damping=damping,
            tv_weight=tv_weight,
            smooth_weight=smooth_weight,
            smooth_sigma=smooth_sigma,
        )
        value = psnr(scene, recon)
        ssim_val = ssim(scene, recon)
        outputs[pattern] = ReconstructionResult(reconstruction=recon, psnr=value, ssim=ssim_val)
    return outputs


__all__ = [
    "richardson_lucy",
    "run_richardson_lucy_baseline",
]

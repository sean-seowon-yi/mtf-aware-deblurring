from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from ..metrics import psnr
from ..optics import pad_to_shape
from .results import ReconstructionResult


def _wiener_filter_kernel(kernel: np.ndarray, image_shape: tuple[int, int], k: float) -> np.ndarray:
    padded = pad_to_shape(kernel, image_shape)
    otf = fft2(ifftshift(padded))
    denom = (np.abs(otf) ** 2) + k
    return np.conj(otf) / denom


def wiener_deconvolution(image: np.ndarray, kernel: np.ndarray, k: float = 1e-3) -> np.ndarray:
    h, w = image.shape[:2]
    wf = _wiener_filter_kernel(kernel, (h, w), k)

    def _deconv(channel: np.ndarray) -> np.ndarray:
        Y = fft2(channel)
        X_hat = np.real(ifft2(Y * wf))
        return np.clip(X_hat, 0.0, 1.0)

    if image.ndim == 2:
        return _deconv(image)
    channels = [_deconv(image[..., c]) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def run_wiener_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    k: float = 1e-3,
) -> Dict[str, ReconstructionResult]:
    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        recon = wiener_deconvolution(data["noisy"], data["kernel"], k=k)
        value = psnr(scene, recon)
        outputs[pattern] = ReconstructionResult(reconstruction=recon, psnr=value)
    return outputs


__all__ = ["wiener_deconvolution", "run_wiener_baseline"]

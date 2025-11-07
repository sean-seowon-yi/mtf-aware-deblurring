from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift


def _spectral_snr_single(img_clean: np.ndarray, img_noisy: np.ndarray, eps: float) -> np.ndarray:
    F_clean = fftshift(fft2(img_clean))
    F_noisy = fftshift(fft2(img_noisy))
    P_signal = np.abs(F_clean) ** 2
    P_noise = np.abs(F_noisy - F_clean) ** 2
    ssnr = (P_signal + eps) / (P_noise + eps)
    return np.log10(ssnr + 1.0)


def spectral_snr(img_clean: np.ndarray, img_noisy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if img_clean.ndim == 2:
        return _spectral_snr_single(img_clean, img_noisy, eps)
    if img_clean.ndim == 3:
        channels = [
            _spectral_snr_single(img_clean[..., c], img_noisy[..., c], eps) for c in range(img_clean.shape[2])
        ]
        return np.stack(channels, axis=-1)
    raise ValueError("spectral_snr expects 2D or 3D inputs.")


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0, eps: float = 1e-12) -> float:
    mse = np.mean((reference - estimate) ** 2)
    if mse <= eps:
        return float("inf")
    return 10 * np.log10((data_range ** 2) / mse)


__all__ = ["spectral_snr", "psnr"]

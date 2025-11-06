from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift


def spectral_snr(img_clean: np.ndarray, img_noisy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    F_clean = fftshift(fft2(img_clean))
    F_noisy = fftshift(fft2(img_noisy))
    P_signal = np.abs(F_clean) ** 2
    P_noise = np.abs(F_noisy - F_clean) ** 2
    ssnr = (P_signal + eps) / (P_noise + eps)
    return np.log10(ssnr + 1.0)


__all__ = ["spectral_snr"]

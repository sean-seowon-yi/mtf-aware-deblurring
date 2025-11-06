from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def pad_to_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    H, W = shape
    h, w = arr.shape
    out = np.zeros((H, W), dtype=arr.dtype)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = arr
    return out


def motion_psf_from_code(code: np.ndarray, length_px: float) -> np.ndarray:
    T = len(code)
    positions = np.linspace(-length_px / 2, length_px / 2, T, endpoint=False) + length_px / (2 * T)
    L = int(np.ceil(length_px))
    L = max(L, 1)
    psf = np.zeros(L, dtype=np.float64)
    for c, pos in zip(code, positions):
        if c <= 0:
            continue
        idx = int(np.floor((pos + length_px / 2) / length_px * L))
        idx = np.clip(idx, 0, L - 1)
        psf[idx] += 1.0
    s = psf.sum()
    if s > 0:
        psf /= s
    else:
        psf[L // 2] = 1.0
    return psf


def kernel2d_from_psf1d(psf1d: np.ndarray) -> np.ndarray:
    return psf1d.reshape(1, -1)


def fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = img.shape
    K = pad_to_shape(kernel, (H, W))
    IMG = fft2(img)
    KER = fft2(ifftshift(K))
    Y = IMG * KER
    return np.real(ifft2(Y))


def otf2d(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    K = pad_to_shape(kernel, shape)
    return fftshift(fft2(ifftshift(K)))


def mtf_from_kernel(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    O = otf2d(kernel, shape)
    M = np.abs(O)
    return M / (M.max() + 1e-12)


__all__ = [
    "pad_to_shape",
    "motion_psf_from_code",
    "kernel2d_from_psf1d",
    "fft_convolve2d",
    "otf2d",
    "mtf_from_kernel",
]

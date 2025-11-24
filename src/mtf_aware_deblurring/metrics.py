from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

_LPIPS_CACHE = {}


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


def _ssim_single(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    data_range: float,
    win_size: int,
    sigma: float,
    K1: float,
    K2: float,
) -> float:
    # Basic SSIM implementation using a Gaussian window via scipy.ndimage.gaussian_filter.
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    if ref.shape != est.shape:
        raise ValueError("Reference and estimate must have the same shape for SSIM.")

    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2

    # Gaussian smoothing approximates local means and variances.
    mu_x = gaussian_filter(ref, sigma=sigma, truncate=win_size / (2 * sigma))
    mu_y = gaussian_filter(est, sigma=sigma, truncate=win_size / (2 * sigma))

    sigma_x2 = gaussian_filter(ref * ref, sigma=sigma, truncate=win_size / (2 * sigma)) - mu_x**2
    sigma_y2 = gaussian_filter(est * est, sigma=sigma, truncate=win_size / (2 * sigma)) - mu_y**2
    sigma_xy = gaussian_filter(ref * est, sigma=sigma, truncate=win_size / (2 * sigma)) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean())


def ssim(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """
    Structural Similarity Index (SSIM).

    This implementation mirrors the standard formulation with a Gaussian
    window. It supports grayscale (H,W) or channel-first/last images; channels
    are averaged.
    """
    if reference.ndim == 2:
        return _ssim_single(reference, estimate, data_range=data_range, win_size=win_size, sigma=sigma, K1=K1, K2=K2)
    if reference.ndim == 3:
        if estimate.shape != reference.shape:
            raise ValueError("Reference and estimate must have the same shape for SSIM.")
        scores = [
            _ssim_single(reference[..., c], estimate[..., c], data_range=data_range, win_size=win_size, sigma=sigma, K1=K1, K2=K2)
            for c in range(reference.shape[2])
        ]
        return float(np.mean(scores))
    raise ValueError("ssim expects 2D or 3D inputs.")

def lpips_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    device: str = "cpu",
    net: str = "vgg",
) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    Uses the lpips package; caches the model per (device, net) to avoid reloads.
    Expects inputs in [0, 1]; converts to [-1, 1] and RGB 3-channel for the metric.
    """
    try:
        import lpips  # type: ignore
    except ImportError as e:
        raise RuntimeError("lpips is not installed. Please install lpips>=0.1.4.") from e

    dev = torch.device(device)
    key = (str(dev), net)
    if key not in _LPIPS_CACHE:
        model = lpips.LPIPS(net=net).to(dev)
        model.eval()
        _LPIPS_CACHE[key] = model
    model = _LPIPS_CACHE[key]

    def _prep(x: np.ndarray) -> torch.Tensor:
        if x.ndim == 2:
            x = x[..., None]
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        if x.shape[2] != 3:
            raise ValueError("LPIPS expects 1-channel or 3-channel inputs.")
        tensor = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(dev)
        tensor = tensor * 2.0 - 1.0  # normalize to [-1, 1]
        return tensor

    with torch.no_grad():
        ref_t = _prep(reference)
        est_t = _prep(estimate)
        # Avoid extremely small inputs
        min_size = 64
        if ref_t.shape[-1] < min_size or ref_t.shape[-2] < min_size:
            ref_t = F.interpolate(ref_t, size=(max(min_size, ref_t.shape[-2]), max(min_size, ref_t.shape[-1])), mode="bilinear", align_corners=False)
            est_t = F.interpolate(est_t, size=ref_t.shape[-2:], mode="bilinear", align_corners=False)
        score = model(ref_t, est_t).item()
    return float(score)

__all__ = ["spectral_snr", "psnr", "ssim", "lpips_distance"]

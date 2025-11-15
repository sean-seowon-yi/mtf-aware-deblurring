from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from ..denoisers import build_denoiser
from ..metrics import psnr
from ..optics import pad_to_shape
from .results import ReconstructionResult
from ..denoisers.drunet_adapter import build_drunet_denoiser


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    kernel = np.clip(kernel, 0.0, None)
    s = float(kernel.sum())
    if s <= 0:
        raise ValueError("Kernel must have positive sum.")
    return kernel / s


def _prepare_otf(kernel: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    padded = pad_to_shape(kernel, shape)
    otf = fft2(ifftshift(padded))
    return otf, np.conj(otf)


def _apply_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    axes = (0, 1)
    spectrum = fft2(image, axes=axes)
    if image.ndim == 2:
        prod = spectrum * otf
    else:
        prod = spectrum * otf[..., None]
    return np.real(ifft2(prod, axes=axes))


def adam_denoiser_deconvolution(
    observation: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 80,
    lr: float = 0.04,
    beta1: float = 0.9,
    beta2: float = 0.995,
    eps: float = 1e-8,
    denoiser_weight: float = 0.2,
    denoiser_interval: int = 2,
    denoiser: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    denoiser_weights: Optional[Path] = None,
    denoiser_device: Optional[str] = None,
    denoiser_type: str = "tiny",
    callback: Optional[Callable[[int, float], None]] = None,
) -> np.ndarray:
    """
    ADAM-based deconvolution with a plug-and-play CNN denoiser regularizer.
    """
    image = np.asarray(observation, dtype=np.float32)
    x = image.copy()
    kernel_norm = _normalize_kernel(kernel)
    otf, otf_conj = _prepare_otf(kernel_norm, image.shape[:2])

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    if denoiser is not None:
        denoiser_obj = denoiser
    else:
        if denoiser_type in ("drunet_color", "drunet_gray"):
            mode = "color" if denoiser_type == "drunet_color" else "gray"
            denoiser_obj = build_drunet_denoiser(
                mode=mode,
                device=denoiser_device or "cuda",
                sigma=20.0,  # or expose as argument if you like
            )
        else:
            denoiser_obj = build_denoiser(
                denoiser_type,
                weights_path=denoiser_weights,
                device=denoiser_device,
            )
    denoiser_weight = float(np.clip(denoiser_weight, 0.0, 1.0))
    denoiser_interval = max(int(denoiser_interval), 1)

    for t in range(1, iterations + 1):
        forward = _apply_otf(x, otf)
        residual = forward - image
        grad = _apply_otf(residual, otf_conj)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        x = np.clip(x - step, 0.0, 1.0)

        if denoiser_weight > 0.0 and (t % denoiser_interval == 0):
            denoised = denoiser_obj(x)
            x = np.clip((1.0 - denoiser_weight) * x + denoiser_weight * denoised, 0.0, 1.0)

        if callback is not None:
            loss = float(np.mean((forward - image) ** 2))
            callback(t, loss)

    return x


def run_adam_denoiser_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    *,
    iterations: int = 80,
    lr: float = 0.04,
    beta1: float = 0.9,
    beta2: float = 0.995,
    eps: float = 1e-8,
    denoiser_weight: float = 0.2,
    denoiser_interval: int = 2,
    denoiser_weights: Optional[Path] = None,
    denoiser_device: Optional[str] = None,
    denoiser_type: str = "tiny",
) -> Dict[str, ReconstructionResult]:
    denoiser = build_denoiser(
        denoiser_type,
        weights_path=denoiser_weights,
        device=denoiser_device,
    )
    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        recon = adam_denoiser_deconvolution(
            data["noisy"],
            data["kernel"],
            iterations=iterations,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            denoiser_weight=denoiser_weight,
            denoiser_interval=denoiser_interval,
            denoiser_type=denoiser_type,
            denoiser=denoiser,
        )
        value = psnr(scene, recon)
        outputs[pattern] = ReconstructionResult(reconstruction=recon, psnr=value)
    return outputs


__all__ = [
    "adam_denoiser_deconvolution",
    "run_adam_denoiser_baseline",
]

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from ..denoisers import build_denoiser
from ..diffusion import DiffusionPrior, DiffusionPriorConfig, build_diffusion_prior
from ..metrics import psnr, ssim
from ..optics import pad_to_shape
from .results import ReconstructionResult
from ..denoisers.drunet_adapter import build_drunet_denoiser
from .prior_scheduler import PhysicsAwareScheduler, PhysicsContext, SchedulerDecision
from .mtf_utils import (
    build_mtf_weights,
    compute_mtf2,
    infer_tau_from_observation,
    mtf_adaptive_sigma_bounds,
    mtf_quality_from_kernel,
)

_DEFAULT_DIFFUSION_CONFIG = DiffusionPriorConfig()


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError("Kernel must have positive sum.")
    return kernel / total


def _prepare_otf(kernel: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    padded = pad_to_shape(kernel, shape)
    otf = fft2(ifftshift(padded))
    return otf, np.conj(otf)


def _fft2_image(image: np.ndarray) -> np.ndarray:
    return fft2(image, axes=(0, 1))


def _ifft2_image(freq: np.ndarray) -> np.ndarray:
    return np.real(ifft2(freq, axes=(0, 1)))


def _apply_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    spectrum = fft2(image, axes=(0, 1))
    if image.ndim == 2:
        prod = spectrum * otf
    else:
        prod = spectrum * otf[..., None]
    return np.real(ifft2(prod, axes=(0, 1)))


def _normalize_mtf(mtf: np.ndarray) -> np.ndarray:
    mtf = np.asarray(mtf, dtype=np.float32)
    max_val = float(mtf.max())
    if max_val > 0:
        mtf = mtf / max_val
    return np.clip(mtf, 0.0, 1.0)


def _build_trust_mask(mtf: np.ndarray, pattern: str, scale: float, floor: float) -> np.ndarray:
    if scale <= 0 or floor <= 0:
        return None  # type: ignore[return-value]
    mtf_norm = _normalize_mtf(mtf)
    gamma = max(scale, 0.1)
    mask = np.clip(mtf_norm**gamma, floor, 1.0)
    return mask.astype(np.float32)


def _x_update(
    z: np.ndarray,
    u: np.ndarray,
    *,
    otf_conj: np.ndarray,
    denom: np.ndarray,
    Y_fft: np.ndarray,
    rho: float,
) -> np.ndarray:
    z_minus_u_fft = _fft2_image(z - u)
    numerator = (otf_conj * Y_fft) + rho * z_minus_u_fft
    X_fft = numerator / denom
    return np.clip(_ifft2_image(X_fft), 0.0, 1.0)


def admm_denoiser_deconvolution(
    observation: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 60,
    rho: float = 0.4,
    denoiser_weight: float = 1.0,
    denoiser: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    denoiser_weights: Optional[Path] = None,
    denoiser_device: Optional[str] = None,
    denoiser_type: str = "tiny",
    callback: Optional[Callable[[int, float], None]] = None,
    scheduler: Optional[PhysicsAwareScheduler] = None,
    pattern_name: Optional[str] = None,
    mtf_weights: Optional[np.ndarray] = None,
    mtf_weights_ready: bool = False,
    mtf_scale: float = 1.5,
    mtf_floor: float = 0.2,
    sigma_scale: Optional[float] = None,
    trace: Optional[list[Dict[str, Any]]] = None,
    denoiser_state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Plug-and-play ADMM solver that alternates between frequency-domain deblurring and
    TinyDenoiser-based proximal updates.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive.")

    obs = np.asarray(observation, dtype=np.float32)
    kernel_norm = _normalize_kernel(kernel)
    otf, otf_conj = _prepare_otf(kernel_norm, obs.shape[:2])
    otf_abs2 = np.abs(otf) ** 2
    weight_map: Optional[np.ndarray] = None
    weight_stats: Dict[str, float] = {}
    if mtf_weights is not None:
        w = np.asarray(mtf_weights, dtype=np.float32)
        if not mtf_weights_ready:
            max_val = float(w.max())
            if max_val > 0:
                w = w / max_val
            w = np.clip(w ** float(mtf_scale), float(mtf_floor), 1.0)
        weight_map = w
        weight_stats = {
            "mtf_weight_min": float(w.min()),
            "mtf_weight_max": float(w.max()),
            "mtf_weight_mean": float(w.mean()),
        }
    denom = otf_abs2 + float(rho)
    if obs.ndim == 3:
        denom = denom[..., None]
        otf_conj = otf_conj[..., None]
        otf_abs2 = otf_abs2[..., None]
        if weight_map is not None:
            weight_map = weight_map[..., None]

    Y_fft = _fft2_image(obs)
    x = obs.copy()
    z = obs.copy()
    u = np.zeros_like(obs)

    if denoiser is None:
        denoiser_obj = build_denoiser(
            denoiser_type,
            weights_path=denoiser_weights,
            device=denoiser_device,
        )
    else:
        denoiser_obj = denoiser
    if hasattr(denoiser_obj, "reset"):
        denoiser_obj.reset()
    denoiser_weight = float(np.clip(denoiser_weight, 0.0, 1.0))

    scheduler_active = scheduler is not None and pattern_name is not None

    for t in range(1, iterations + 1):
        current_weight = denoiser_weight
        current_rho = rho
        current_sigma_scale = sigma_scale
        decision: Optional[SchedulerDecision] = None
        if scheduler_active and scheduler is not None:
            decision = scheduler.plan_iteration(pattern_name, t)
            current_weight = float(np.clip(decision.denoiser_weight, 0.0, 1.0))
            current_rho = float(max(decision.rho, 1e-6))
            if decision.sigma_scale is not None:
                current_sigma_scale = decision.sigma_scale

        if weight_map is not None:
            weighted_abs2 = otf_abs2 * weight_map
            weighted_conj = otf_conj * weight_map
        else:
            weighted_abs2 = otf_abs2
            weighted_conj = otf_conj

        denom = weighted_abs2 + current_rho
        z_minus_u_fft = _fft2_image(z - u)
        numerator = (weighted_conj * Y_fft) + current_rho * z_minus_u_fft
        X_fft = numerator / denom
        x = np.clip(_ifft2_image(X_fft), 0.0, 1.0)

        v = x + u
        if current_weight > 0.0:
            denoised = denoiser_obj(v)
            if current_sigma_scale is not None and hasattr(denoiser_obj, "sigma_scale"):
                try:
                    denoiser_obj.sigma_scale = float(current_sigma_scale)
                except Exception:
                    pass
            z = np.clip((1.0 - current_weight) * v + current_weight * denoised, 0.0, 1.0)
        else:
            z = np.clip(v, 0.0, 1.0)

        u = u + x - z

        if callback is not None:
            forward = _apply_otf(x, otf)
            loss = float(np.mean((forward - obs) ** 2))
            callback(t, loss)

        if trace is not None:
            entry: Dict[str, Any] = {
                "iteration": t,
                "rho": float(current_rho),
                "denoiser_weight": float(current_weight),
                "sigma_scale": None if current_sigma_scale is None else float(current_sigma_scale),
            }
            if weight_stats:
                entry.update(weight_stats)
            if decision is not None:
                entry["scheduler_extras"] = decision.extras
            if denoiser_state is not None:
                for key in ("last_sigma", "sigma"):
                    if key in denoiser_state:
                        entry["denoiser_sigma"] = float(denoiser_state[key])
                        break
            trace.append(entry)

    return x


def admm_diffusion_deconvolution(
    observation: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 40,
    rho: float = 0.6,
    diffusion_prior: Optional[DiffusionPrior] = None,
    diffusion_prior_type: Literal["tiny_score"] = "tiny_score",
    diffusion_prior_weights: Optional[Path] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_guidance: Optional[float] = None,
    diffusion_noise_scale: Optional[float] = None,
    diffusion_sigma_min: float = 0.01,
    diffusion_sigma_max: float = 0.5,
    diffusion_schedule: Literal["geom", "linear"] = "geom",
    diffusion_device: Optional[str] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> np.ndarray:
    """
    ADMM solver that replaces the proximal denoiser with a diffusion prior.

    The implementation follows recent plug-and-play diffusion literature such as
    Diffusion Posterior Sampling (Chung et al., CVPR 2023) and ADMM-Score (Wang et al.,
    NeurIPS 2023) where the score-model delivers gradients of the log-prior that guide the
    proximal step.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive.")

    obs = np.asarray(observation, dtype=np.float32)
    kernel_norm = _normalize_kernel(kernel)
    otf, otf_conj = _prepare_otf(kernel_norm, obs.shape[:2])
    denom = np.abs(otf) ** 2 + float(rho)
    if obs.ndim == 3:
        denom = denom[..., None]

    Y_fft = _fft2_image(obs)
    x = obs.copy()
    z = obs.copy()
    u = np.zeros_like(obs)

    if diffusion_prior is None:
        base = _DEFAULT_DIFFUSION_CONFIG
        cfg = DiffusionPriorConfig(
            steps=diffusion_steps if diffusion_steps is not None else base.steps,
            sigma_min=diffusion_sigma_min,
            sigma_max=diffusion_sigma_max,
            schedule=diffusion_schedule,
            guidance=diffusion_guidance if diffusion_guidance is not None else base.guidance,
            noise_scale=diffusion_noise_scale if diffusion_noise_scale is not None else base.noise_scale,
        )
        channels = 1 if obs.ndim == 2 else obs.shape[2]
        diffusion_prior = build_diffusion_prior(
            diffusion_prior_type,
            weights_path=diffusion_prior_weights,
            device=diffusion_device,
            config=cfg,
            in_channels=channels,
        )
    else:
        if diffusion_steps is not None:
            diffusion_prior.update_schedule(diffusion_steps)
    if diffusion_prior is None:  # mypy guard
        raise RuntimeError("Diffusion prior construction failed.")

    for t in range(1, iterations + 1):
        x = _x_update(z, u, otf_conj=otf_conj, denom=denom, Y_fft=Y_fft, rho=rho)
        v = x + u
        z = diffusion_prior.proximal(
            v,
            rho=rho,
            guidance=diffusion_guidance,
            noise_scale=diffusion_noise_scale,
        )
        u = u + x - z

        if callback is not None:
            forward = _apply_otf(x, otf)
            loss = float(np.mean((forward - obs) ** 2))
            callback(t, loss)

    return x

def run_admm_denoiser_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    *,
    iterations: int = 60,
    rho: float = 0.4,
    denoiser_weight: float = 1.0,
    denoiser_weights: Optional[Path] = None,
    denoiser_device: Optional[str] = None,
    denoiser_type: str = "tiny",
    scheduler: Optional[PhysicsAwareScheduler] = None,
    pattern_contexts: Optional[Dict[str, PhysicsContext]] = None,
    mtf_scale: float = 1.5,
    mtf_floor: float = 0.2,
    mtf_weighting_mode: Literal["gamma", "wiener", "combined", "none"] = "none",
    mtf_wiener_alpha: float = 0.5,
    mtf_wiener_floor: float = 0.05,
    mtf_wiener_tau_min: float = 1e-4,
    mtf_wiener_tau_max: float = 1e-1,
    mtf_sigma_adapt: bool = False,
) -> Dict[str, ReconstructionResult]:

    denoiser: Callable[[np.ndarray], np.ndarray]
    is_drunet = denoiser_type in ("drunet_color", "drunet_gray")

    if is_drunet:
        mode = "color" if denoiser_type == "drunet_color" else "gray"
        active_pattern: Dict[str, Any] = {"name": None}
        sigma_tracker: Dict[str, Any] = {"last_sigma": None}

        # Exponential schedule; modulated by the physics-aware scheduler if available.
        def sigma_schedule(t: int, T: int) -> float:
            pattern_name = active_pattern["name"]
            sigma_max, sigma_min = 25.0, 8.0
            if mtf_sigma_adapt and pattern_name is not None:
                kernel = active_pattern.get("kernel")
                image_shape = active_pattern.get("image_shape")
                if kernel is not None and image_shape is not None:
                    Q = mtf_quality_from_kernel(kernel, image_shape)
                    # Only adapt for weak/medium MTF cases; leave high quality cases at defaults.
                    if Q < 0.7:
                        sigma_max, sigma_min = mtf_adaptive_sigma_bounds(Q)
            elif pattern_name == "legendre":
                sigma_max, sigma_min = 40.0, 20.0
            if T <= 1:
                base_sigma = sigma_min
            else:
                r = (sigma_min / sigma_max) ** (1.0 / max(T - 1, 1))
                base_sigma = sigma_max * (r ** (t - 1))

            if scheduler is not None and pattern_name is not None:
                decision = scheduler.plan_iteration(pattern_name, t)
                if decision.sigma_scale is not None:
                    base_sigma = float(np.clip(base_sigma * decision.sigma_scale, sigma_min, sigma_max * 1.5))
            sigma_tracker["last_sigma"] = base_sigma
            return base_sigma

        denoiser = build_drunet_denoiser(
            mode=mode,
            device=denoiser_device or "cuda",
            sigma_init=25.0,
            sigma_schedule=sigma_schedule,
            iterations=iterations,
        )
    else:
        denoiser = build_denoiser(
            denoiser_type,
            weights_path=denoiser_weights,
            device=denoiser_device,
        )

    if scheduler is not None:
        scheduler_contexts = pattern_contexts or {}
        scheduler.set_contexts(scheduler_contexts)

    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        if is_drunet:
            active_pattern["name"] = pattern
            active_pattern["kernel"] = data.get("kernel")
            active_pattern["image_shape"] = data.get("noisy", scene).shape[:2]
        if scheduler is not None:
            scheduler.reset()
        ctx = None
        if pattern_contexts is not None:
            ctx = pattern_contexts.get(pattern)
        noisy = np.asarray(data["noisy"], dtype=np.float32)
        kernel = np.asarray(data["kernel"], dtype=np.float32)

        mtf_weights: Optional[np.ndarray] = None
        mtf_weights_ready = False
        gamma_mask: Optional[np.ndarray] = None
        wiener_mask: Optional[np.ndarray] = None
        if ctx is not None and mtf_weighting_mode in ("gamma", "combined"):
            gamma_mask = _build_trust_mask(ctx.mtf, pattern, scale=mtf_scale, floor=mtf_floor)
        if mtf_weighting_mode in ("wiener", "combined"):
            mtf2 = compute_mtf2(kernel, noisy.shape[:2])
            tau_est = infer_tau_from_observation(
                noisy, tau_min=mtf_wiener_tau_min, tau_max=mtf_wiener_tau_max
            )
            wiener_mask = build_mtf_weights(
                mtf2,
                tau=tau_est,
                alpha=mtf_wiener_alpha,
                floor=mtf_wiener_floor,
            )
        if mtf_weighting_mode == "gamma":
            mtf_weights = gamma_mask
            mtf_weights_ready = False
        elif mtf_weighting_mode == "wiener":
            mtf_weights = wiener_mask
            mtf_weights_ready = True
        elif mtf_weighting_mode == "combined":
            if gamma_mask is not None and wiener_mask is not None:
                combined = gamma_mask * wiener_mask
                max_val = float(combined.max())
                if max_val > 0:
                    combined = combined / max_val
                combined = np.clip(combined, min(mtf_floor, mtf_wiener_floor), 1.0)
                mtf_weights = combined.astype(np.float32)
                mtf_weights_ready = True
            elif gamma_mask is not None:
                mtf_weights = gamma_mask
                mtf_weights_ready = False
            else:
                mtf_weights = wiener_mask
                mtf_weights_ready = True
        else:
            mtf_weights = None
            mtf_weights_ready = False
        sigma_scale = None
        if scheduler is not None:
            # sigma scaling will be provided per iteration; initialize hints
            sigma_scale = 1.0
        trace: list[Dict[str, Any]] = []
        recon = admm_denoiser_deconvolution(
            noisy,
            kernel,
            iterations=iterations,
            rho=rho,
            denoiser_weight=denoiser_weight,
            denoiser_type=denoiser_type,
            denoiser=denoiser,
            scheduler=scheduler,
            pattern_name=pattern,
            mtf_weights=mtf_weights,
            mtf_scale=mtf_scale,
            mtf_floor=mtf_floor,
            sigma_scale=sigma_scale,
            trace=trace,
            denoiser_state=sigma_tracker if is_drunet else None,
            mtf_weights_ready=mtf_weights_ready,
        )
        value = psnr(scene, recon)
        ssim_val = ssim(scene, recon)
        outputs[pattern] = ReconstructionResult(
            reconstruction=recon,
            psnr=value,
            ssim=ssim_val,
            trace=trace if trace else None,
        )
    return outputs


def run_admm_diffusion_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    *,
    iterations: int = 40,
    rho: float = 0.6,
    diffusion_prior: Optional[DiffusionPrior] = None,
    diffusion_prior_type: Literal["tiny_score"] = "tiny_score",
    diffusion_prior_weights: Optional[Path] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_guidance: Optional[float] = None,
    diffusion_noise_scale: Optional[float] = None,
    diffusion_sigma_min: float = 0.01,
    diffusion_sigma_max: float = 0.5,
    diffusion_schedule: Literal["geom", "linear"] = "geom",
    diffusion_device: Optional[str] = None,
) -> Dict[str, ReconstructionResult]:
    prior = diffusion_prior
    if prior is None:
        base = _DEFAULT_DIFFUSION_CONFIG
        cfg = DiffusionPriorConfig(
            steps=diffusion_steps if diffusion_steps is not None else base.steps,
            sigma_min=diffusion_sigma_min,
            sigma_max=diffusion_sigma_max,
            schedule=diffusion_schedule,
            guidance=diffusion_guidance if diffusion_guidance is not None else base.guidance,
            noise_scale=diffusion_noise_scale if diffusion_noise_scale is not None else base.noise_scale,
        )
        channels = 1 if scene.ndim == 2 else scene.shape[2]
        prior = build_diffusion_prior(
            diffusion_prior_type,
            weights_path=diffusion_prior_weights,
            device=diffusion_device,
            config=cfg,
            in_channels=channels,
        )
    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        recon = admm_diffusion_deconvolution(
            data["noisy"],
            data["kernel"],
            iterations=iterations,
            rho=rho,
            diffusion_prior=prior,
            diffusion_steps=diffusion_steps,
            diffusion_guidance=diffusion_guidance,
            diffusion_noise_scale=diffusion_noise_scale,
        )
        value = psnr(scene, recon)
        ssim_val = ssim(scene, recon)
        outputs[pattern] = ReconstructionResult(reconstruction=recon, psnr=value, ssim=ssim_val, trace=None)
    return outputs


__all__ = [
    "admm_denoiser_deconvolution",
    "admm_diffusion_deconvolution",
    "run_admm_denoiser_baseline",
    "run_admm_diffusion_baseline",
]

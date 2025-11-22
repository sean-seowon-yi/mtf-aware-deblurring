from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Protocol, TypedDict

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

# --- Internal Imports (Adjust paths as per your project structure) ---
from ..denoisers import build_denoiser
from ..diffusion import DiffusionPrior, DiffusionPriorConfig, build_diffusion_prior
from ..metrics import psnr, ssim
from ..optics import pad_to_shape
from .results import ReconstructionResult
from .prior_scheduler import PhysicsAwareScheduler, PhysicsContext

# MTF Utils
from .mtf_utils import (
    build_mtf_weights,
    compute_mtf2,
    infer_tau_from_observation,
    mtf_quality_from_kernel,
)

_DEFAULT_DIFFUSION_CONFIG = DiffusionPriorConfig()

class ProximalCallable(Protocol):
    def __call__(self, image: np.ndarray, sigma: float) -> np.ndarray: ...

class OpticsConfig(TypedDict):
    mtf_gamma: float
    mtf_floor: float

# --- Helper Functions ---

def get_optics_heuristics(pattern_name: str) -> OpticsConfig:
    name = str(pattern_name).lower()
    if any(x in name for x in ["box", "rect", "diffuser", "sinc"]):
        return {"mtf_gamma": 2.0, "mtf_floor": 0.1}
    elif any(x in name for x in ["rand", "uniform", "noise"]):
        return {"mtf_gamma": 1.0, "mtf_floor": 0.3}
    else:
        return {"mtf_gamma": 1.5, "mtf_floor": 0.2}

def _fft2_image(image: np.ndarray) -> np.ndarray:
    return fft2(image, axes=(0, 1))

def _ifft2_image(freq: np.ndarray) -> np.ndarray:
    return np.real(ifft2(freq, axes=(0, 1)))

def _normalize_mtf(mtf: np.ndarray) -> np.ndarray:
    mtf = np.asarray(mtf, dtype=np.float32)
    max_val = float(mtf.max())
    if max_val > 0: mtf /= max_val
    return np.clip(mtf, 0.0, 1.0)

def _build_trust_mask(mtf: np.ndarray, *, scale: float, floor: float) -> np.ndarray:
    mtf_norm = _normalize_mtf(mtf)
    gamma = max(scale, 0.1)
    return np.clip(mtf_norm ** gamma, floor, 1.0).astype(np.float32)

def _prepare_otf(kernel: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.asarray(kernel, dtype=np.float32)
    k_sum = kernel.sum()
    if k_sum > 0: kernel /= k_sum
    padded = pad_to_shape(kernel, shape)
    otf = fft2(ifftshift(padded))
    return otf, np.conj(otf)

def _apply_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    spec = fft2(image, axes=(0, 1))
    if image.ndim == 2: prod = spec * otf
    else: prod = spec * otf[..., None]
    return np.real(ifft2(prod, axes=(0, 1)))

def _solve_x_update(z, u, otf_conj, otf_abs2, Y_fft, weight_map, rho):
    """
    Weighted Frequency Domain Least Squares Update.
    Solves: (W*|H|^2 + rho*I)x = W*H'*Y + rho*(z - u)
    """
    rhs_prior = rho * _fft2_image(z - u)
    if weight_map is not None:
        numerator = (otf_conj * weight_map * Y_fft) + rhs_prior
        denom = (otf_abs2 * weight_map) + rho
    else:
        numerator = (otf_conj * Y_fft) + rhs_prior
        denom = otf_abs2 + rho
    return _ifft2_image(numerator / denom)

# --- Core Logic ---

def _core_pnp_admm(
    observation: np.ndarray, 
    kernel: np.ndarray, 
    prox_operator: ProximalCallable,
    *, 
    iterations: int, 
    rho: float, 
    prior_weight: float, 
    mtf_weights: Optional[np.ndarray],
    sigma_multiplier: float, 
    scheduler: Optional[PhysicsAwareScheduler], 
    pattern_name: Optional[str],
    callback: Optional[Callable], 
    trace: Optional[list], 
    use_internal_ramp: bool = True
) -> np.ndarray:
    
    obs = np.asarray(observation, dtype=np.float32)
    otf, otf_conj = _prepare_otf(kernel, obs.shape[:2])
    otf_abs2 = np.abs(otf) ** 2
    Y_fft = _fft2_image(obs)

    # Handle Multi-channel (e.g., RGB) vs Grayscale
    if obs.ndim == 3:
        otf_conj = otf_conj[..., None]
        otf_abs2 = otf_abs2[..., None]
        if mtf_weights is not None: mtf_weights = mtf_weights[..., None]

    x = obs.copy()
    z = obs.copy()
    u = np.zeros_like(obs)
    if scheduler: scheduler.reset()

    # Determine Rho Strategy
    effective_use_ramp = use_internal_ramp and (scheduler is None)
    if effective_use_ramp:
        rho_start = max(rho * 0.02, 1e-4)
        rho_end = rho * 1.2
        rho_step = (rho_end / rho_start) ** (1.0 / (iterations - 1)) if iterations > 1 else 1.0
    else:
        rho_start, rho_step = rho, 1.0

    for t in range(1, iterations + 1):
        # 1. Determine Parameters
        if effective_use_ramp:
            current_rho = rho_start * (rho_step ** (t - 1))
        else:
            current_rho = rho
        current_weight = prior_weight

        # Scheduler Hook
        scheduler_extras = {}
        if scheduler and pattern_name:
            decision = scheduler.plan_iteration(pattern_name, t)
            if decision.rho > 0: current_rho = float(decision.rho)
            if decision.denoiser_weight is not None: current_weight = float(decision.denoiser_weight)
            scheduler_extras = decision.extras or {}

        # 2. X-Update (Data Consistency)
        x = _solve_x_update(z, u, otf_conj, otf_abs2, Y_fft, mtf_weights, current_rho)

        # 3. Interleaved Denoising Logic
        # Optimization: Only run heavy CNN on ODD iterations or LAST iteration
        perform_denoising = (t % 2 != 0) or (t == iterations)

        if perform_denoising:
            # PnP Theory: sigma = sqrt(lambda / rho). Assuming lambda=1.
            theoretical_sigma = np.sqrt(1.0 / current_rho) if current_rho > 0 else 0.0
            
            # Apply multipliers for perceptual tuning
            effective_sigma = theoretical_sigma * 0.65 * sigma_multiplier
            
            v = x + u
            z_denoised = prox_operator(v, sigma=effective_sigma)
            
            # Apply Denoiser Weight (Relaxation)
            if current_weight < 1.0:
                z_hat = (1.0 - current_weight) * v + current_weight * z_denoised
            else:
                z_hat = z_denoised
        else:
            # Skip denoising step to save compute (reuse previous state implied)
            v = x + u
            z_hat = v 
            effective_sigma = 0.0

        # 4. Dual Update
        z = np.clip(z_hat, 0.0, 1.0)
        u = u + x - z

        # 5. Logging & Callbacks
        if callback:
            est_blur = _apply_otf(x, otf)
            callback(t, float(np.mean((est_blur - obs) ** 2)))
            
        if trace is not None:
            step_log = {
                "iteration": t, "rho": current_rho, "sigma_pnp": effective_sigma,
                "weight": current_weight, "skipped": not perform_denoising
            }
            # Inject scheduler decision rationale into trace for debugging
            step_log.update(scheduler_extras)
            trace.append(step_log)

    return z

def admm_denoiser_deconvolution(
    observation: np.ndarray, 
    kernel: np.ndarray, 
    *, 
    iterations: int, 
    rho: float,
    denoiser_weight: float, 
    denoiser_obj: Any, 
    mtf_weights: Optional[np.ndarray] = None,
    sigma_multiplier: float = 1.0, 
    denoiser_sigma_scale: float = 8.0,  # IMPROVEMENT: Parameterized scaling
    scheduler: Optional[PhysicsAwareScheduler] = None,
    pattern_name: Optional[str] = None, 
    trace: Optional[list] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> np.ndarray:

    def denoiser_prox(image: np.ndarray, sigma: float) -> np.ndarray:
        # Scale sigma input for the specific CNN (DRUNet/Tiny) preferences
        # Lower scale = network thinks noise is lower = preserves more grain
        sigma_input = float(np.clip(sigma * denoiser_sigma_scale, 0.0, 50.0))

        if hasattr(denoiser_obj, "sigma_scale"): denoiser_obj.sigma_scale = 1.0
        try:
            return denoiser_obj(image, sigma=sigma_input)
        except TypeError as e:
            # Fallback if denoiser doesn't accept sigma arg
            if "sigma" in str(e) or "unexpected keyword" in str(e):
                return denoiser_obj(image)
            raise e

    return _core_pnp_admm(
        observation, kernel, prox_operator=denoiser_prox, iterations=iterations,
        rho=rho, prior_weight=denoiser_weight, mtf_weights=mtf_weights,
        sigma_multiplier=sigma_multiplier, scheduler=scheduler, pattern_name=pattern_name,
        callback=callback, trace=trace, use_internal_ramp=True
    )

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
    denoiser_sigma_scale: float = 8.0, # IMPROVEMENT: Exposed to runner
    scheduler: Optional[PhysicsAwareScheduler] = None,
    pattern_contexts: Optional[Dict[str, PhysicsContext]] = None,
    mtf_scale: float = 0.0, 
    mtf_floor: float = 0.0, 
    mtf_weighting_mode: str = "none",
    mtf_wiener_alpha: float = 0.5, 
    mtf_wiener_floor: float = 0.05,
    mtf_wiener_tau_min: float = 1e-4, 
    mtf_wiener_tau_max: float = 1e-1,
    mtf_sigma_adapt: bool = False,
) -> Dict[str, ReconstructionResult]:

    denoiser = build_denoiser(denoiser_type, weights_path=denoiser_weights, device=denoiser_device or "cpu")
    if hasattr(denoiser, "reset"): denoiser.reset()
    if scheduler: scheduler.set_contexts(pattern_contexts or {})
        
    outputs = {}
    for pattern, data in forward_results.items():
        noisy, kernel = data["noisy"], data["kernel"]
        
        # MTF Weighting Logic
        final_mask = None
        if mtf_weighting_mode != "none":
            defaults = get_optics_heuristics(pattern)
            use_gamma = mtf_scale if mtf_scale > 0 else defaults["mtf_gamma"]
            use_floor = mtf_floor if mtf_floor > 0 else defaults["mtf_floor"]
            mtf2 = compute_mtf2(kernel, noisy.shape[:2])
            
            if mtf_weighting_mode == "gamma":
                final_mask = _build_trust_mask(mtf2, scale=use_gamma, floor=use_floor)
            elif mtf_weighting_mode == "wiener":
                tau = infer_tau_from_observation(noisy, mtf_wiener_tau_min, mtf_wiener_tau_max)
                final_mask = build_mtf_weights(mtf2, tau=tau, alpha=mtf_wiener_alpha, floor=mtf_wiener_floor)
            elif mtf_weighting_mode == "combined":
                tau = infer_tau_from_observation(noisy, mtf_wiener_tau_min, mtf_wiener_tau_max)
                w_mask = build_mtf_weights(mtf2, tau=tau, alpha=mtf_wiener_alpha, floor=mtf_wiener_floor)
                g_mask = _build_trust_mask(mtf2, scale=use_gamma, floor=use_floor)
                combined = w_mask * g_mask
                if combined.max() > 0: combined /= combined.max()
                final_mask = np.clip(combined, use_floor, 1.0).astype(np.float32)

        # Sigma Adaptation Logic
        sigma_mult = 1.0
        if mtf_sigma_adapt:
            Q = mtf_quality_from_kernel(kernel, noisy.shape[:2])
            sigma_mult = 1.0 + 0.4 * np.clip(1.0 - Q, 0.0, 1.0)
            print(f"[{pattern}] MTF Quality Q={Q:.2f} -> Sigma Multiplier={sigma_mult:.2f}x")

        trace_log = []
        recon = admm_denoiser_deconvolution(
            noisy, kernel, iterations=iterations, rho=rho,
            denoiser_weight=denoiser_weight, denoiser_obj=denoiser, mtf_weights=final_mask,
            sigma_multiplier=sigma_mult, denoiser_sigma_scale=denoiser_sigma_scale,
            scheduler=scheduler, pattern_name=pattern,
            trace=trace_log
        )
        outputs[pattern] = ReconstructionResult(
            reconstruction=recon, psnr=psnr(scene, recon), ssim=ssim(scene, recon), trace=trace_log
        )
    return outputs

# --- Diffusion Wrappers (Standard) ---

def admm_diffusion_deconvolution(
    observation: np.ndarray, kernel: np.ndarray, *, iterations: int = 40, rho: float = 0.6,
    diffusion_prior: Optional[DiffusionPrior] = None, diffusion_prior_type: str = "tiny_score",
    diffusion_prior_weights: Optional[Path] = None, diffusion_steps: Optional[int] = None,
    diffusion_guidance: Optional[float] = None, diffusion_noise_scale: Optional[float] = None,
    diffusion_sigma_min: float = 0.01, diffusion_sigma_max: float = 0.5,
    diffusion_schedule: Literal["geom", "linear"] = "geom", diffusion_device: Optional[str] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if diffusion_prior is None:
        base = _DEFAULT_DIFFUSION_CONFIG
        cfg = DiffusionPriorConfig(
            steps=diffusion_steps or base.steps, sigma_min=diffusion_sigma_min,
            sigma_max=diffusion_sigma_max, schedule=diffusion_schedule,
            guidance=diffusion_guidance or base.guidance, noise_scale=diffusion_noise_scale or base.noise_scale,
        )
        channels = 1 if obs.ndim == 2 else obs.shape[2]
        diffusion_prior = build_diffusion_prior(
            diffusion_prior_type, weights_path=diffusion_prior_weights,
            device=diffusion_device or "cpu", config=cfg, in_channels=channels
        )
    else:
        if diffusion_steps is not None: diffusion_prior.update_schedule(diffusion_steps)
    diffusion_prior.update_schedule(iterations)

    def diff_prox(image: np.ndarray, sigma: float) -> np.ndarray:
        return diffusion_prior.proximal(image, rho=rho, guidance=diffusion_guidance, noise_scale=diffusion_noise_scale)

    return _core_pnp_admm(
        obs, kernel, prox_operator=diff_prox, iterations=iterations, rho=rho, prior_weight=1.0,
        mtf_weights=None, sigma_multiplier=1.0, scheduler=None, pattern_name=None,
        callback=callback, trace=None, use_internal_ramp=False
    )

def run_admm_diffusion_baseline(
    scene: np.ndarray, forward_results: Dict[str, Dict[str, np.ndarray]], *, iterations: int = 40,
    rho: float = 0.6, diffusion_prior_type: str = "tiny_score", diffusion_prior_weights: Optional[Path] = None,
    diffusion_steps: Optional[int] = None, diffusion_guidance: Optional[float] = None,
    diffusion_noise_scale: Optional[float] = None, diffusion_sigma_min: float = 0.01,
    diffusion_sigma_max: float = 0.5, diffusion_schedule: Literal["geom", "linear"] = "geom",
    diffusion_device: Optional[str] = None,
) -> Dict[str, ReconstructionResult]:
    base = _DEFAULT_DIFFUSION_CONFIG
    cfg = DiffusionPriorConfig(
        steps=diffusion_steps or base.steps, sigma_min=diffusion_sigma_min,
        sigma_max=diffusion_sigma_max, schedule=diffusion_schedule,
        guidance=diffusion_guidance or base.guidance, noise_scale=diffusion_noise_scale or base.noise_scale,
    )
    obs_shape = next(iter(forward_results.values()))["noisy"].shape
    prior = build_diffusion_prior(
        diffusion_prior_type, weights_path=diffusion_prior_weights, device=diffusion_device or "cpu",
        config=cfg, in_channels=1 if len(obs_shape) == 2 else obs_shape[2],
    )
    prior.update_schedule(iterations)
    outputs = {}
    for pattern, data in forward_results.items():
        recon = admm_diffusion_deconvolution(
            data["noisy"], data["kernel"], iterations=iterations, rho=rho, diffusion_prior=prior,
            diffusion_prior_type=diffusion_prior_type, diffusion_prior_weights=diffusion_prior_weights,
            diffusion_steps=diffusion_steps, diffusion_guidance=diffusion_guidance,
            diffusion_noise_scale=diffusion_noise_scale, diffusion_sigma_min=diffusion_sigma_min,
            diffusion_sigma_max=diffusion_sigma_max, diffusion_schedule=diffusion_schedule,
            diffusion_device=diffusion_device,
        )
        outputs[pattern] = ReconstructionResult(
            reconstruction=recon, psnr=psnr(scene, recon), ssim=ssim(scene, recon), trace=None
        )
    return outputs

__all__ = [
    "admm_denoiser_deconvolution", "admm_diffusion_deconvolution",
    "run_admm_denoiser_baseline", "run_admm_diffusion_baseline",
]
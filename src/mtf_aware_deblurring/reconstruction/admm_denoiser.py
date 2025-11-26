from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Protocol, TypedDict

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

# --- Internal Imports ---
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

# --- Heuristics / thresholds for optics quality and MTF cutoff ---
# These are deliberately mild; you can tune them per dataset.
_Q_FALLBACK_THRESHOLD = 0.0        # if >0, very poor optics fall back to pure denoising
_MTF_CUTOFF_BASE = 0.01            # soft cutoff when optics are decent
_MTF_CUTOFF_HARD = 0.02            # stronger cutoff when optics are bad


class ProximalCallable(Protocol):
    def __call__(self, image: np.ndarray, sigma: float) -> np.ndarray:
        ...


class OpticsConfig(TypedDict):
    mtf_gamma: float
    mtf_floor: float


# --- Helper Functions ---


def get_optics_heuristics(pattern_name: str) -> OpticsConfig:
    """
    Simple pattern-name based heuristics for MTF gamma / floor.
    """
    name = str(pattern_name).lower()
    if any(x in name for x in ["box", "rect", "diffuser", "sinc"]):
        # box-like codes: stronger gamma, modest floor
        return {"mtf_gamma": 2.0, "mtf_floor": 0.1}
    elif any(x in name for x in ["rand", "uniform", "noise"]):
        # random codes: smoother MTF, higher floor so we trust more frequencies
        return {"mtf_gamma": 1.0, "mtf_floor": 0.3}
    else:
        # legendre, etc.
        return {"mtf_gamma": 1.5, "mtf_floor": 0.2}


def _fft2_image(image: np.ndarray) -> np.ndarray:
    return fft2(image, axes=(0, 1))


def _ifft2_image(freq: np.ndarray) -> np.ndarray:
    return np.real(ifft2(freq, axes=(0, 1)))


def _normalize_mtf(mtf: np.ndarray) -> np.ndarray:
    mtf = np.asarray(mtf, dtype=np.float32)
    max_val = float(mtf.max())
    if max_val > 0:
        mtf /= max_val
    return np.clip(mtf, 0.0, 1.0)


def _build_trust_mask(
    mtf: np.ndarray,
    *,
    scale: float,
    floor: float,
    cutoff: float = _MTF_CUTOFF_BASE,
) -> np.ndarray:
    """
    Build a trust mask from the MTF.

    - Normalize to [0, 1].
    - Raise to gamma (scale).
    - Optionally zero out very low-MTF regions (hard cutoff).
    - For the remaining positive entries, enforce a floor (like v2),
      BUT keep true zeros where we intentionally killed frequencies.

    This combines the robustness of the floor-style weighting from v2
    with the ability to explicitly kill hopeless frequencies.
    """
    mtf_norm = _normalize_mtf(mtf)
    gamma = max(scale, 0.1)
    w = mtf_norm ** gamma

    # Hard zero out very low MTF regions if requested
    if cutoff > 0.0:
        w[mtf_norm < cutoff] = 0.0

    # Enforce a floor only on entries that are non-zero (i.e., we still trust them)
    if floor > 0.0:
        mask_pos = w > 0.0
        w[mask_pos] = np.clip(w[mask_pos], floor, 1.0)

    # Allow genuine zeros where optics destroyed information
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def _prepare_otf(kernel: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.asarray(kernel, dtype=np.float32)
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    padded = pad_to_shape(kernel, shape)
    otf = fft2(ifftshift(padded))
    return otf, np.conj(otf)


def _apply_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    spec = fft2(image, axes=(0, 1))
    if image.ndim == 2:
        prod = spec * otf
    else:
        prod = spec * otf[..., None]
    return np.real(ifft2(prod, axes=(0, 1)))


def _solve_x_update(
    z: np.ndarray,
    u: np.ndarray,
    otf_conj: np.ndarray,
    otf_abs2: np.ndarray,
    Y_fft: np.ndarray,
    weight_map: Optional[np.ndarray],
    rho: float,
) -> np.ndarray:
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
    denoiser_interval: int = 2,
    scheduler: Optional[PhysicsAwareScheduler] = None,
    pattern_name: Optional[str] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    trace: Optional[list] = None,
    use_internal_ramp: bool = True,
    # Conservative stopping thresholds (can be tuned/disabled)
    tol_primal: float = 1e-4,
    tol_dual: float = 1e-4,
    min_check_iter: int = 10,
) -> np.ndarray:
    """
    Core PnP-ADMM loop with:

    - Optional internal rho ramp (when no scheduler is present).
    - Physics-aware scheduler hooks.
    - Interleaved denoising via denoiser_interval.
    - Optional early stopping using normalized primal/dual residuals.
    """

    obs = np.asarray(observation, dtype=np.float32)
    otf, otf_conj = _prepare_otf(kernel, obs.shape[:2])
    otf_abs2 = np.abs(otf) ** 2
    Y_fft = _fft2_image(obs)

    # Handle multi-channel vs grayscale
    if obs.ndim == 3:
        otf_conj = otf_conj[..., None]
        otf_abs2 = otf_abs2[..., None]
        if mtf_weights is not None:
            mtf_weights = mtf_weights[..., None]

    x = obs.copy()
    z = obs.copy()
    u = np.zeros_like(obs)

    if scheduler:
        scheduler.reset()

    denoiser_interval = max(int(denoiser_interval), 1)

    # Rho ramp if no external scheduler is controlling rho
    effective_use_ramp = use_internal_ramp and (scheduler is None)
    if effective_use_ramp:
        rho_start = max(rho * 0.02, 1e-4)
        rho_end = rho * 1.2
        rho_step = (rho_end / rho_start) ** (1.0 / (iterations - 1)) if iterations > 1 else 1.0
    else:
        rho_start, rho_step = rho, 1.0

    # For residual computation
    prev_z = z.copy()

    for t in range(1, iterations + 1):
        # 1. Determine parameters
        if effective_use_ramp:
            current_rho = rho_start * (rho_step ** (t - 1))
        else:
            current_rho = rho
        current_weight = prior_weight

        # Scheduler hook
        scheduler_extras: Dict[str, Any] = {}
        if scheduler and pattern_name:
            decision = scheduler.plan_iteration(pattern_name, t)
            if decision.rho > 0:
                current_rho = float(decision.rho)
            if decision.denoiser_weight is not None:
                current_weight = float(decision.denoiser_weight)
            scheduler_extras = decision.extras or {}

        # 2. X update (data consistency)
        x = _solve_x_update(z, u, otf_conj, otf_abs2, Y_fft, mtf_weights, current_rho)

        # 3. Interleaved denoising
        perform_denoising = (((t - 1) % denoiser_interval) == 0) or (t == iterations)

        if perform_denoising:
            theoretical_sigma = np.sqrt(1.0 / current_rho) if current_rho > 0 else 0.0
            effective_sigma = theoretical_sigma * 0.65 * sigma_multiplier
            v = x + u
            z_denoised = prox_operator(v, sigma=effective_sigma)
            if current_weight < 1.0:
                z_hat = (1.0 - current_weight) * v + current_weight * z_denoised
            else:
                z_hat = z_denoised
        else:
            v = x + u
            z_hat = v
            effective_sigma = 0.0

        # 4. Dual update
        z = np.clip(z_hat, 0.0, 1.0)
        u = u + x - z

        # 5. Residuals & logging
        x_flat = x.ravel()
        z_flat = z.ravel()
        prev_z_flat = prev_z.ravel()

        # Avoid division by zero by adding a tiny epsilon
        primal_res = float(
            np.linalg.norm((x_flat - z_flat), ord=2)
            / (np.linalg.norm(x_flat, ord=2) + 1e-8)
        )
        dual_res = float(
            np.linalg.norm((z_flat - prev_z_flat), ord=2)
            / (np.linalg.norm(prev_z_flat, ord=2) + 1e-8)
        )
        prev_z = z.copy()

        if callback:
            est_blur = _apply_otf(x, otf)
            mse_val = float(np.mean((est_blur - obs) ** 2))
            callback(t, mse_val)

        if trace is not None:
            step_log: Dict[str, Any] = {
                "iteration": t,
                "rho": float(current_rho),
                "sigma_pnp": float(effective_sigma),
                "weight": float(current_weight),
                "skipped": not perform_denoising,
                "primal_res": primal_res,
                "dual_res": dual_res,
            }
            step_log.update(scheduler_extras)
            trace.append(step_log)

        # 6. Early stopping (more conservative: only after min_check_iter)
        if (t >= min_check_iter) and (tol_primal > 0.0 or tol_dual > 0.0):
            if (primal_res < tol_primal) and (dual_res < tol_dual):
                print(
                    f"[ADMM] Early stop at iter {t} "
                    f"(primal={primal_res:.2e}, dual={dual_res:.2e})"
                )
                break

    return z


# --- Denoiser-based ADMM ---


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
    denoiser_sigma_scale: float = 8.0,
    denoiser_interval: int = 2,
    scheduler: Optional[PhysicsAwareScheduler] = None,
    pattern_name: Optional[str] = None,
    trace: Optional[list] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> np.ndarray:
    """
    ADMM with a learned denoiser prior (DRUNet/tiny).
    """

    def denoiser_prox(image: np.ndarray, sigma: float) -> np.ndarray:
        # Scale sigma input for the specific CNN (DRUNet or tiny).
        # Clip to a sane range as in the second version.
        sigma_input = float(np.clip(sigma * denoiser_sigma_scale, 0.0, 50.0))

        if hasattr(denoiser_obj, "sigma_scale"):
            denoiser_obj.sigma_scale = 1.0
        try:
            return denoiser_obj(image, sigma=sigma_input)
        except TypeError as e:
            if "sigma" in str(e) or "unexpected keyword" in str(e):
                return denoiser_obj(image)
            raise e

    return _core_pnp_admm(
        observation,
        kernel,
        prox_operator=denoiser_prox,
        iterations=iterations,
        rho=rho,
        prior_weight=denoiser_weight,
        mtf_weights=mtf_weights,
        sigma_multiplier=sigma_multiplier,
        denoiser_interval=denoiser_interval,
        scheduler=scheduler,
        pattern_name=pattern_name,
        callback=callback,
        trace=trace,
        use_internal_ramp=True,
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
    denoiser_sigma_scale: float = 8.0,
    denoiser_interval: int = 2,
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
    """
    Wrapper to run ADMM+denoiser across patterns with MTF-aware weighting.
    """

    denoiser = build_denoiser(
        denoiser_type,
        weights_path=denoiser_weights,
        device=denoiser_device or "cpu",
    )
    if hasattr(denoiser, "reset"):
        denoiser.reset()
    if scheduler:
        scheduler.set_contexts(pattern_contexts or {})

    outputs: Dict[str, ReconstructionResult] = {}

    for pattern, data in forward_results.items():
        noisy = data["noisy"]
        kernel = data["kernel"]

        noisy_psnr = psnr(scene, noisy)
        noisy_ssim = ssim(scene, noisy)
        print(
            f"[{pattern}] Forward noisy PSNR: {noisy_psnr:.2f} dB | "
            f"SSIM: {noisy_ssim:.4f}"
        )

        # Compute MTF and quality score once
        mtf2 = compute_mtf2(kernel, noisy.shape[:2])
        Q = mtf_quality_from_kernel(kernel, noisy.shape[:2])
        optics_hopeless = Q < _Q_FALLBACK_THRESHOLD

        # Sigma adaptation (PnP sigma multiplier)
        sigma_mult = 1.0
        if mtf_sigma_adapt:
            sigma_mult = 1.0 + 0.4 * np.clip(1.0 - Q, 0.0, 1.0)
            print(f"[{pattern}] MTF Quality Q={Q:.2f} -> Sigma Multiplier={sigma_mult:.2f}x")

        # Optional pure-denoising fallback for extremely bad optics
        if optics_hopeless:
            print(
                f"[{pattern}] Q={Q:.2f} below threshold {_Q_FALLBACK_THRESHOLD:.2f}, "
                f"using pure denoising fallback instead of deconvolution."
            )
            sigma_fallback = 25.0  # strong but reasonable training-scale sigma
            try:
                pure = denoiser(noisy, sigma=sigma_fallback)
            except TypeError as e:
                if "sigma" in str(e) or "unexpected keyword" in str(e):
                    pure = denoiser(noisy)
                else:
                    raise e

            outputs[pattern] = ReconstructionResult(
                reconstruction=pure,
                psnr=psnr(scene, pure),
                ssim=ssim(scene, pure),
                trace=[],
            )
            continue

        # --- MTF weighting logic ---
        final_mask: Optional[np.ndarray] = None
        if mtf_weighting_mode != "none":
            defaults = get_optics_heuristics(pattern)
            use_gamma = mtf_scale if mtf_scale > 0 else defaults["mtf_gamma"]
            use_floor = mtf_floor if mtf_floor > 0 else defaults["mtf_floor"]

            # For poor optics, you *could* increase cutoff to kill more HF.
            cutoff = _MTF_CUTOFF_BASE if Q >= 0.15 else _MTF_CUTOFF_HARD

            if mtf_weighting_mode == "gamma":
                final_mask = _build_trust_mask(
                    mtf2,
                    scale=use_gamma,
                    floor=use_floor,
                    cutoff=cutoff,
                )
            elif mtf_weighting_mode == "wiener":
                tau = infer_tau_from_observation(
                    noisy,
                    mtf_wiener_tau_min,
                    mtf_wiener_tau_max,
                )
                final_mask = build_mtf_weights(
                    mtf2,
                    tau=tau,
                    alpha=mtf_wiener_alpha,
                    floor=mtf_wiener_floor,
                )
            elif mtf_weighting_mode == "combined":
                tau = infer_tau_from_observation(
                    noisy,
                    mtf_wiener_tau_min,
                    mtf_wiener_tau_max,
                )
                w_mask = build_mtf_weights(
                    mtf2,
                    tau=tau,
                    alpha=mtf_wiener_alpha,
                    floor=mtf_wiener_floor,
                )
                g_mask = _build_trust_mask(
                    mtf2,
                    scale=use_gamma,
                    floor=use_floor,
                    cutoff=cutoff,
                )
                combined = w_mask * g_mask
                max_val = combined.max()
                if max_val > 0:
                    combined = combined / max_val
                # Enforce a floor on non-zero entries, like in _build_trust_mask
                if use_floor > 0.0:
                    mask_pos = combined > 0.0
                    combined[mask_pos] = np.clip(combined[mask_pos], use_floor, 1.0)
                final_mask = np.clip(combined, 0.0, 1.0).astype(np.float32)

        trace_log: list = []
        recon = admm_denoiser_deconvolution(
            noisy,
            kernel,
            iterations=iterations,
            rho=rho,
            denoiser_weight=denoiser_weight,
            denoiser_obj=denoiser,
            mtf_weights=final_mask,
            sigma_multiplier=sigma_mult,
            denoiser_sigma_scale=denoiser_sigma_scale,
            denoiser_interval=denoiser_interval,
            scheduler=scheduler,
            pattern_name=pattern,
            trace=trace_log,
        )
        outputs[pattern] = ReconstructionResult(
            reconstruction=recon,
            psnr=psnr(scene, recon),
            ssim=ssim(scene, recon),
            trace=trace_log,
        )

    return outputs


# --- Diffusion Wrappers (unchanged behaviour, just formatted) ---


def admm_diffusion_deconvolution(
    observation: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 40,
    rho: float = 0.6,
    diffusion_prior: Optional[DiffusionPrior] = None,
    diffusion_prior_type: str = "tiny_score",
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
    obs = np.asarray(observation, dtype=np.float32)
    if diffusion_prior is None:
        base = _DEFAULT_DIFFUSION_CONFIG
        cfg = DiffusionPriorConfig(
            steps=diffusion_steps or base.steps,
            sigma_min=diffusion_sigma_min,
            sigma_max=diffusion_sigma_max,
            schedule=diffusion_schedule,
            guidance=diffusion_guidance or base.guidance,
            noise_scale=diffusion_noise_scale or base.noise_scale,
        )
        channels = 1 if obs.ndim == 2 else obs.shape[2]
        diffusion_prior = build_diffusion_prior(
            diffusion_prior_type,
            weights_path=diffusion_prior_weights,
            device=diffusion_device or "cpu",
            config=cfg,
            in_channels=channels,
        )
    else:
        if diffusion_steps is not None:
            diffusion_prior.update_schedule(diffusion_steps)
    diffusion_prior.update_schedule(iterations)

    def diff_prox(image: np.ndarray, sigma: float) -> np.ndarray:
        return diffusion_prior.proximal(
            image,
            rho=rho,
            guidance=diffusion_guidance,
            noise_scale=diffusion_noise_scale,
        )

    return _core_pnp_admm(
        obs,
        kernel,
        prox_operator=diff_prox,
        iterations=iterations,
        rho=rho,
        prior_weight=1.0,
        mtf_weights=None,
        sigma_multiplier=1.0,
        denoiser_interval=1,
        scheduler=None,
        pattern_name=None,
        callback=callback,
        trace=None,
        use_internal_ramp=False,
        # diffusion prior is already strong; turn off early-stop here by default
        tol_primal=0.0,
        tol_dual=0.0,
    )


def run_admm_diffusion_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    *,
    iterations: int = 40,
    rho: float = 0.6,
    diffusion_prior_type: str = "tiny_score",
    diffusion_prior_weights: Optional[Path] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_guidance: Optional[float] = None,
    diffusion_noise_scale: Optional[float] = None,
    diffusion_sigma_min: float = 0.01,
    diffusion_sigma_max: float = 0.5,
    diffusion_schedule: Literal["geom", "linear"] = "geom",
    diffusion_device: Optional[str] = None,
) -> Dict[str, ReconstructionResult]:
    base = _DEFAULT_DIFFUSION_CONFIG
    cfg = DiffusionPriorConfig(
        steps=diffusion_steps or base.steps,
        sigma_min=diffusion_sigma_min,
        sigma_max=diffusion_sigma_max,
        schedule=diffusion_schedule,
        guidance=diffusion_guidance or base.guidance,
        noise_scale=diffusion_noise_scale or base.noise_scale,
    )
    obs_shape = next(iter(forward_results.values()))["noisy"].shape
    prior = build_diffusion_prior(
        diffusion_prior_type,
        weights_path=diffusion_prior_weights,
        device=diffusion_device or "cpu",
        config=cfg,
        in_channels=1 if len(obs_shape) == 2 else obs_shape[2],
    )
    prior.update_schedule(iterations)

    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        recon = admm_diffusion_deconvolution(
            data["noisy"],
            data["kernel"],
            iterations=iterations,
            rho=rho,
            diffusion_prior=prior,
            diffusion_prior_type=diffusion_prior_type,
            diffusion_prior_weights=diffusion_prior_weights,
            diffusion_steps=diffusion_steps,
            diffusion_guidance=diffusion_guidance,
            diffusion_noise_scale=diffusion_noise_scale,
            diffusion_sigma_min=diffusion_sigma_min,
            diffusion_sigma_max=diffusion_sigma_max,
            diffusion_schedule=diffusion_schedule,
            diffusion_device=diffusion_device,
        )
        outputs[pattern] = ReconstructionResult(
            reconstruction=recon,
            psnr=psnr(scene, recon),
            ssim=ssim(scene, recon),
            trace=None,
        )

    return outputs


__all__ = [
    "admm_denoiser_deconvolution",
    "admm_diffusion_deconvolution",
    "run_admm_denoiser_baseline",
    "run_admm_diffusion_baseline",
]

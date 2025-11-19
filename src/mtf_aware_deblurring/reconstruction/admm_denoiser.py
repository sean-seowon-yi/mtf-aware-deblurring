from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Literal, Optional

import numpy as np
import torch
from numpy.fft import fft2, ifft2, ifftshift

from ..denoisers import build_denoiser
from ..diffusion import DiffusionPrior, DiffusionPriorConfig, build_diffusion_prior
from ..metrics import psnr
from ..optics import pad_to_shape
from .results import ReconstructionResult
from ..denoisers.drunet_adapter import build_drunet_denoiser
from ..diffusion.guided_uncond_prior import GuidedDiffusionUncondPrior

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


# ------------------------------------------------------------------------------
# Guided diffusion prior (patch-wise, level-based)
# ------------------------------------------------------------------------------

def _apply_guided_prior_patchwise(
    v: np.ndarray,
    prior: GuidedDiffusionUncondPrior,
    level: float,
    tile_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Apply a GuidedDiffusionUncondPrior to an image of arbitrary size using
    overlapping 2D patches, driven by a normalized prior-strength level.

    Parameters
    ----------
    v : np.ndarray
        Image in [0,1], shape (H, W) or (H, W, C).
    prior : GuidedDiffusionUncondPrior
        Must expose `denoise_level_01(x_01: torch.Tensor, level: float)`.
    level : float
        Prior strength in [0,1].
    tile_size : int
        Patch size (guided-diffusion uncond is 256x256).
    overlap : int
        Overlap between patches for smooth blending.

    Returns
    -------
    np.ndarray
        Denoised image in [0,1], same shape as v.
    """
    assert v.ndim in (2, 3), "v must be HxW or HxWxC"
    v_np = np.asarray(v, dtype=np.float32)
    H, W = v_np.shape[:2]
    has_channels = (v_np.ndim == 3)
    C = v_np.shape[2] if has_channels else 1

    device = next(prior.parameters()).device  # type: ignore[arg-type]

    # Small images: pad once and run a single patch
    if H <= tile_size and W <= tile_size:
        pad_bottom = tile_size - H
        pad_right = tile_size - W
        if has_channels:
            pad_config = ((0, pad_bottom), (0, pad_right), (0, 0))
        else:
            pad_config = ((0, pad_bottom), (0, pad_right))
        v_padded = np.pad(v_np, pad_config, mode="reflect")

        # [N,C,H,W] in [0,1]
        if has_channels:
            v_t = torch.from_numpy(v_padded).permute(2, 0, 1).unsqueeze(0)
        else:
            v_t = torch.from_numpy(v_padded)[None, None, ...]
        v_t = v_t.float().to(device)

        with torch.no_grad():
            out_t = prior.denoise_level_01(v_t, level)

        out_np = out_t.detach().cpu().numpy()
        if has_channels:
            out_np = out_np[0].transpose(1, 2, 0)
        else:
            out_np = out_np[0, 0]

        return out_np[:H, :W].astype(np.float32)

    # General case: overlapping tiles
    stride = tile_size - overlap
    out = np.zeros_like(v_np, dtype=np.float32)
    weight = np.zeros((H, W, 1), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            patch = v_np[y:y_end, x:x_end, ...] if has_channels else v_np[y:y_end, x:x_end]

            ph, pw = patch.shape[:2]
            pad_bottom = tile_size - ph
            pad_right = tile_size - pw
            if has_channels:
                pad_config = ((0, pad_bottom), (0, pad_right), (0, 0))
            else:
                pad_config = ((0, pad_bottom), (0, pad_right))
            patch_padded = np.pad(patch, pad_config, mode="reflect")

            if has_channels:
                patch_t = torch.from_numpy(patch_padded).permute(2, 0, 1).unsqueeze(0)
            else:
                patch_t = torch.from_numpy(patch_padded)[None, None, ...]
            patch_t = patch_t.float().to(device)

            with torch.no_grad():
                out_patch_t = prior.denoise_level_01(patch_t, level)

            out_patch = out_patch_t.detach().cpu().numpy()
            if has_channels:
                out_patch = out_patch[0].transpose(1, 2, 0)
            else:
                out_patch = out_patch[0, 0]

            out_patch = out_patch[:ph, :pw]

            if has_channels:
                out[y:y_end, x:x_end, :] += out_patch
            else:
                out[y:y_end, x:x_end] += out_patch
            weight[y:y_end, x:x_end, 0] += 1.0

    weight = np.clip(weight, 1e-6, None)
    if has_channels:
        out /= weight
    else:
        out /= weight[..., 0]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------
# MTF tools
# ------------------------------------------------------------------------------

def _compute_mtf2(kernel: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    kernel_norm = _normalize_kernel(kernel)
    otf, _ = _prepare_otf(kernel_norm, image_shape)
    mtf2 = np.abs(otf) ** 2
    return mtf2


def _build_mtf_weights(
    mtf2: np.ndarray,
    tau: float = 1e-4,
    alpha: float = 0.5,
    floor: float = 0.05,
) -> np.ndarray:
    """
    Build per-frequency weights from MTF^2:

    - Normalize |H|^2 to [0, 1].
    - Apply exponent alpha (< 1) to compress dynamic range.
    - Clamp to [floor, 1] so we never fully ignore the data term.
    """
    eps = 1e-8
    mtf2_max = float(mtf2.max())
    if not np.isfinite(mtf2_max) or mtf2_max <= 0.0:
        return np.ones_like(mtf2, dtype=np.float32)

    mtf2_norm = (mtf2 / (mtf2_max + eps)).astype(np.float32)
    W = np.power(mtf2_norm, alpha)
    W = np.clip(W, floor, 1.0)
    return W


def _mtf_quality_from_kernel(kernel: np.ndarray, image_shape: tuple[int, int], tau: float = 1e-3) -> float:
    mtf2 = _compute_mtf2(kernel, image_shape)
    mtf2_max = float(mtf2.max())
    if not np.isfinite(mtf2_max) or mtf2_max <= 0.0:
        return 0.0

    mtf2_max = mtf2_max + 1e-8
    mtf2_norm = mtf2 / mtf2_max
    mask = mtf2_norm > tau
    if not mask.any():
        return 0.0
    return float(mtf2_norm[mask].mean())


def _mtf_adaptive_sigma_bounds(Q: float) -> tuple[float, float]:
    """
    Map an MTF quality scalar Q (~[0, 1]) to (sigma_max, sigma_min).

    Lower Q (worse MTF) -> larger sigmas (stronger prior).

    Intended for DRUNet / score-based priors that expect pixel-noise-like sigma.
    Not used directly for guided_uncond (which uses normalized levels).
    """
    SIGMA_MAX_LO, SIGMA_MAX_HI = 25.0, 40.0
    SIGMA_MIN_LO, SIGMA_MIN_HI = 8.0, 20.0

    Q_clamped = float(np.clip(Q, 0.0, 1.0))
    sigma_max = SIGMA_MAX_LO + (1.0 - Q_clamped) * (SIGMA_MAX_HI - SIGMA_MAX_LO)
    sigma_min = SIGMA_MIN_LO + (1.0 - Q_clamped) * (SIGMA_MIN_HI - SIGMA_MIN_LO)
    return sigma_max, sigma_min


def _build_sigma_schedule(
    T: int,
    sigma_min: float,
    sigma_max: float,
    schedule: Literal["geom", "linear"] = "geom",
) -> np.ndarray:
    """
    Build a length-T sigma schedule from sigma_max -> sigma_min.

    Used for score-based diffusion priors (tiny_score), not for the
    guided_uncond prior (which uses normalized levels).
    """
    if T <= 1:
        return np.array([sigma_min], dtype=np.float32)

    if schedule == "linear":
        return np.linspace(sigma_max, sigma_min, T, dtype=np.float32)

    # geometric
    r = (sigma_min / sigma_max) ** (1.0 / max(T - 1, 1))
    return np.array([sigma_max * (r ** t) for t in range(T)], dtype=np.float32)


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
    return _ifft2_image(X_fft)


# ------------------------------------------------------------------------------
# ADMM + DRUNet / classical denoiser (unchanged)
# ------------------------------------------------------------------------------

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
    use_mtf_weighting: bool = False,
    rho_min: Optional[float] = None,
    denoiser_interval: int = 1,
) -> np.ndarray:
    """
    Plug-and-play ADMM solver that alternates between frequency-domain deblurring and
    TinyDenoiser / DRUNet-based proximal updates.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive.")

    obs = np.asarray(observation, dtype=np.float32)
    kernel_norm = _normalize_kernel(kernel)
    otf, otf_conj = _prepare_otf(kernel_norm, obs.shape[:2])

    mtf2 = np.abs(otf) ** 2
    if use_mtf_weighting:
        W_2d = _build_mtf_weights(mtf2)
        denom_2d = mtf2 * W_2d + float(rho)
    else:
        W_2d = None
        denom_2d = mtf2 + float(rho)

    Y_fft = _fft2_image(obs)

    if obs.ndim == 3:
        if W_2d is not None:
            W = W_2d[..., None]
            Y_fft_weighted = Y_fft * W
            denom = denom_2d[..., None]
        else:
            Y_fft_weighted = Y_fft
            denom = denom_2d[..., None]
        otf_conj_use = otf_conj[..., None]
    else:
        if W_2d is not None:
            Y_fft_weighted = Y_fft * W_2d
        else:
            Y_fft_weighted = Y_fft
        denom = denom_2d
        otf_conj_use = otf_conj

    if rho_min is None or rho_min >= rho:
        def rho_schedule(t: int) -> float:
            return float(rho)
    else:
        rho0 = float(rho_min)
        rhoT = float(rho)

        def rho_schedule(t: int) -> float:
            if iterations <= 1:
                return rhoT
            r = (rhoT / rho0) ** (1.0 / max(iterations - 1, 1))
            return rho0 * (r ** (t - 1))

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
    denoiser_interval = max(int(denoiser_interval), 1)

    for t in range(1, iterations + 1):
        rho_t = rho_schedule(t)

        if use_mtf_weighting and W_2d is not None:
            denom_2d = mtf2 * W_2d + rho_t
        else:
            denom_2d = mtf2 + rho_t

        if obs.ndim == 3:
            denom = denom_2d[..., None]
        else:
            denom = denom_2d

        x = _x_update(
            z,
            u,
            otf_conj=otf_conj_use,
            denom=denom,
            Y_fft=Y_fft_weighted,
            rho=rho_t,
        )

        v = x + u
        apply_prior = (denoiser_weight > 0.0) and ((t % denoiser_interval) == 0)

        if apply_prior:
            v_for_denoiser = np.clip(v, 0.0, 1.0)
            denoised = denoiser_obj(v_for_denoiser)
            z = (1.0 - denoiser_weight) * v + denoiser_weight * denoised
        else:
            z = v

        z = np.clip(z, 0.0, 1.0)
        u = u + x - z

        if callback is not None:
            forward = _apply_otf(x, otf)
            loss = float(np.mean((forward - obs) ** 2))
            callback(t, loss)

    x = np.clip(x, 0.0, 1.0)
    return x


# ------------------------------------------------------------------------------
# ADMM + diffusion prior (tiny_score or guided_uncond) with prior_weight
# ------------------------------------------------------------------------------

def admm_diffusion_deconvolution(
    observation: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 40,
    rho: float = 0.6,
    diffusion_prior: Optional[DiffusionPrior] = None,
    diffusion_prior_type: Literal["tiny_score", "guided_uncond"] = "tiny_score",
    diffusion_prior_weights: Optional[Path] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_guidance: Optional[float] = None,
    diffusion_noise_scale: Optional[float] = None,
    diffusion_sigma_min: float = 0.01,
    diffusion_sigma_max: float = 0.5,
    diffusion_schedule: Literal["geom", "linear"] = "geom",
    diffusion_device: Optional[str] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    use_mtf_weighting: bool = False,
    prior_weight: float = 0.5,   # <<< NEW: blend strength for guided_uncond
) -> np.ndarray:
    """
    ADMM solver that replaces the proximal denoiser with either:

    - a score-based diffusion prior (DiffusionPrior.proximal), or
    - a pretrained unconditional guided-diffusion prior, applied patch-wise.

    For `tiny_score`, diffusion_sigma_* are actual sigma values for the
    score-based prior. For `guided_uncond`, diffusion_sigma_* are interpreted
    as normalized prior-strength levels in [0,1] and are clamped.

    `prior_weight` controls how strongly the guided prior overwrites the
    ADMM iterate v:

        z_t = (1 - prior_weight) * v_t + prior_weight * prior_out_t
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive.")

    prior_weight = float(np.clip(prior_weight, 0.0, 1.0))

    obs = np.asarray(observation, dtype=np.float32)
    kernel_norm = _normalize_kernel(kernel)
    otf, otf_conj = _prepare_otf(kernel_norm, obs.shape[:2])

    mtf2 = np.abs(otf) ** 2
    if use_mtf_weighting:
        W_2d = _build_mtf_weights(mtf2)
        denom_2d = mtf2 * W_2d + float(rho)
    else:
        W_2d = None
        denom_2d = mtf2 + float(rho)

    Y_fft = _fft2_image(obs)

    if obs.ndim == 3:
        if W_2d is not None:
            W = W_2d[..., None]
            Y_fft_weighted = Y_fft * W
            denom = denom_2d[..., None]
        else:
            Y_fft_weighted = Y_fft
            denom = denom_2d[..., None]
        otf_conj_use = otf_conj[..., None]
    else:
        if W_2d is not None:
            Y_fft_weighted = Y_fft * W_2d
        else:
            Y_fft_weighted = Y_fft
        denom = denom_2d
        otf_conj_use = otf_conj

    x = obs.copy()
    z = obs.copy()
    u = np.zeros_like(obs)

    # Build / reuse prior
    prior = diffusion_prior
    if prior is None:
        if diffusion_prior_type == "guided_uncond":
            prior = GuidedDiffusionUncondPrior(
                ckpt_path=diffusion_prior_weights,
                device=diffusion_device or "cuda",
                image_size=256,
                use_fp16=(diffusion_device or "cuda") != "cpu",
            )
        else:
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
            prior = build_diffusion_prior(
                diffusion_prior_type,
                weights_path=diffusion_prior_weights,
                device=diffusion_device,
                config=cfg,
                in_channels=channels,
            )
    else:
        if isinstance(prior, DiffusionPrior) and diffusion_steps is not None:
            prior.update_schedule(diffusion_steps)

    if prior is None:
        raise RuntimeError("Diffusion prior construction failed.")

    is_guided_prior = isinstance(prior, GuidedDiffusionUncondPrior)

    if is_guided_prior:
        # Treat diffusion_sigma_* as normalized levels in [0,1]
        level_max = float(np.clip(diffusion_sigma_max, 0.0, 1.0))
        level_min = float(np.clip(diffusion_sigma_min, 0.0, level_max))
        if iterations <= 1:
            level_schedule = np.array([level_min], dtype=np.float32)
        else:
            if diffusion_schedule == "linear":
                level_schedule = np.linspace(level_max, level_min, iterations, dtype=np.float32)
            else:
                # geometric-ish in level space: not super meaningful, but keep option
                r = (max(level_min, 1e-4) / max(level_max, 1e-4)) ** (1.0 / max(iterations - 1, 1))
                level_schedule = np.array(
                    [level_max * (r ** t) for t in range(iterations)],
                    dtype=np.float32,
                )
        sigma_schedule = None
    else:
        sigma_schedule = _build_sigma_schedule(
            T=iterations,
            sigma_min=diffusion_sigma_min,
            sigma_max=diffusion_sigma_max,
            schedule=diffusion_schedule,
        )
        level_schedule = None

    for t in range(1, iterations + 1):
        x = _x_update(
            z,
            u,
            otf_conj=otf_conj_use,
            denom=denom,
            Y_fft=Y_fft_weighted,
            rho=rho,
        )

        v = x + u

        if is_guided_prior:
            level_t = float(level_schedule[t - 1])
            v_clipped = np.clip(v, 0.0, 1.0)

            prior_out = _apply_guided_prior_patchwise(
                v_clipped,
                prior,
                level=level_t,
                tile_size=256,
                overlap=64,
            )

            # <<< HERE: convex combination of v and prior output >>>
            z = (1.0 - prior_weight) * v + prior_weight * prior_out
            z = np.clip(z, 0.0, 1.0)
        else:
            # tiny_score / other DiffusionPrior (uses its own internal schedule)
            z = prior.proximal(
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

    x = np.clip(x, 0.0, 1.0)
    return x


# ------------------------------------------------------------------------------
# Baseline runners
# ------------------------------------------------------------------------------

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
    use_mtf_weighting: bool = False,
    rho_min: float = 0.02,
    denoiser_interval: int = 1
) -> Dict[str, ReconstructionResult]:

    denoiser: Callable[[np.ndarray], np.ndarray]
    is_drunet = denoiser_type in ("drunet_color", "drunet_gray")

    if use_mtf_weighting is True:
        use_mtf_sigma_adapt = True
    else:
        use_mtf_sigma_adapt = False

    if is_drunet:
        mode = "color" if denoiser_type == "drunet_color" else "gray"
        active_pattern: Dict[str, object] = {"name": None, "sigma_bounds": None}

        def sigma_schedule(t: int, T: int) -> float:
            sigma_bounds = active_pattern.get("sigma_bounds")
            if sigma_bounds is not None:
                sigma_max, sigma_min = sigma_bounds
            else:
                pattern_name = active_pattern.get("name")
                if pattern_name == "legendre":
                    sigma_max, sigma_min = 40.0, 20.0
                else:
                    sigma_max, sigma_min = 25.0, 8.0

            if T <= 1:
                return sigma_min
            r = (sigma_min / sigma_max) ** (1.0 / max(T - 1, 1))
            return sigma_max * (r ** (t - 1))

        denoiser = build_drunet_denoiser(
            mode=mode,
            device=denoiser_device or "cuda",
            sigma_init=25.0,
            sigma_schedule=sigma_schedule,
            iterations=iterations,
        )
    else:
        active_pattern = {}  # type: ignore[assignment]
        denoiser = build_denoiser(
            denoiser_type,
            weights_path=denoiser_weights,
            device=denoiser_device,
        )

    outputs: Dict[str, ReconstructionResult] = {}
    for pattern, data in forward_results.items():
        if is_drunet:
            active_pattern["name"] = pattern
            if use_mtf_sigma_adapt:
                noisy = np.asarray(data["noisy"], dtype=np.float32)
                kernel = np.asarray(data["kernel"], dtype=np.float32)
                Q = _mtf_quality_from_kernel(kernel, noisy.shape[:2])
                active_pattern["sigma_bounds"] = _mtf_adaptive_sigma_bounds(Q)
            else:
                active_pattern["sigma_bounds"] = None

        recon = admm_denoiser_deconvolution(
            data["noisy"],
            data["kernel"],
            iterations=iterations,
            rho=rho,
            denoiser_weight=denoiser_weight,
            denoiser_type=denoiser_type,
            denoiser=denoiser,
            use_mtf_weighting=use_mtf_weighting,
            rho_min=rho_min,
            denoiser_interval=denoiser_interval
        )
        value = psnr(scene, recon)
        outputs[pattern] = ReconstructionResult(reconstruction=recon, psnr=value)
    return outputs


def run_admm_diffusion_baseline(
    scene: np.ndarray,
    forward_results: Dict[str, Dict[str, np.ndarray]],
    *,
    iterations: int = 40,
    rho: float = 0.6,
    diffusion_prior: Optional[DiffusionPrior] = None,
    diffusion_prior_type: Literal["tiny_score", "guided_uncond"] = "tiny_score",
    diffusion_prior_weights: Optional[Path] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_guidance: Optional[float] = None,
    diffusion_noise_scale: Optional[float] = None,
    diffusion_sigma_min: float = 0.01,
    diffusion_sigma_max: float = 0.5,
    diffusion_schedule: Literal["geom", "linear"] = "geom",
    diffusion_device: Optional[str] = None,
    use_mtf_weighting: bool = False,
    diffusion_prior_weight: float = 0.5,   # <<< NEW: baseline-level control
) -> Dict[str, ReconstructionResult]:
    """
    Run ADMM + diffusion prior (tiny_score or guided_uncond) for each pattern.

    - `use_mtf_weighting` controls whether the *data term* is MTF-weighted.
    - For `tiny_score`, you can additionally enable MTF-adaptive sigma bounds.
    - For `guided_uncond`, diffusion_sigma_* are interpreted as normalized
      prior-strength levels in [0,1].
    - `diffusion_prior_weight` is forwarded to `admm_diffusion_deconvolution`
      and only affects the guided_uncond branch.
    """
    if diffusion_prior_type == "guided_uncond":
        use_mtf_sigma_adapt = False
    else:
        use_mtf_sigma_adapt = bool(use_mtf_weighting)

    prior = diffusion_prior

    if prior is None and diffusion_prior_type != "guided_uncond":
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

    if prior is None and diffusion_prior_type == "guided_uncond":
        prior = GuidedDiffusionUncondPrior(
            ckpt_path=diffusion_prior_weights,
            device=diffusion_device or "cuda",
            image_size=256,
            use_fp16=(diffusion_device or "cuda") != "cpu",
        )

    outputs: Dict[str, ReconstructionResult] = {}

    for pattern, data in forward_results.items():
        noisy = np.asarray(data["noisy"], dtype=np.float32)
        kernel = np.asarray(data["kernel"], dtype=np.float32)

        if diffusion_prior_type != "guided_uncond" and use_mtf_sigma_adapt:
            Q = _mtf_quality_from_kernel(kernel, noisy.shape[:2])
            sigma_max_pat, sigma_min_pat = _mtf_adaptive_sigma_bounds(Q)
        else:
            sigma_max_pat, sigma_min_pat = diffusion_sigma_max, diffusion_sigma_min

        recon = admm_diffusion_deconvolution(
            noisy,
            kernel,
            iterations=iterations,
            rho=rho,
            diffusion_prior=prior,
            diffusion_prior_type=diffusion_prior_type,
            diffusion_steps=diffusion_steps,
            diffusion_guidance=diffusion_guidance,
            diffusion_noise_scale=diffusion_noise_scale,
            diffusion_sigma_min=sigma_min_pat,
            diffusion_sigma_max=sigma_max_pat,
            diffusion_schedule=diffusion_schedule,
            diffusion_device=diffusion_device,
            use_mtf_weighting=use_mtf_weighting,
            prior_weight=diffusion_prior_weight,
        )

        value = psnr(scene, recon)
        outputs[pattern] = ReconstructionResult(reconstruction=recon, psnr=value)
    return outputs


__all__ = [
    "admm_denoiser_deconvolution",
    "admm_diffusion_deconvolution",
    "run_admm_denoiser_baseline",
    "run_admm_diffusion_baseline",
]

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
# tau update
# ------------------------------------------------------------------------------

def _estimate_noise_sigma_mad(image: np.ndarray) -> float:
    """
    Estimate spatial-domain noise standard deviation using a simple MAD
    (median absolute deviation) on finite differences.

    Assumes:
    - image is float32, roughly in [0, 1].
    - noise is approximately i.i.d. and dominates high-frequency differences.

    Returns
    -------
    float
        Estimated noise sigma. Returns 0.0 if estimation fails.
    """
    img = np.asarray(image, dtype=np.float32)

    # Convert color to gray by averaging channels (noise level should be similar)
    if img.ndim == 3:
        img = img.mean(axis=2)

    H, W = img.shape[:2]
    if H < 2 or W < 2:
        return 0.0

    # Simple high-frequency proxy: finite differences
    dx = img[:, 1:] - img[:, :-1]
    dy = img[1:, :] - img[:-1, :]

    diffs = np.concatenate([dx.ravel(), dy.ravel()])
    if diffs.size == 0:
        return 0.0

    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0

    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med))
    if mad <= 0.0 or not np.isfinite(mad):
        return 0.0

    # For Gaussian noise, sigma ≈ MAD / 0.6745
    sigma = mad / 0.6745
    return float(sigma)

def _infer_tau_from_observation(
    obs: np.ndarray,
    tau_min: float = 1e-4,
    tau_max: float = 1e-1,
) -> float:
    """
    Infer a dimensionless noise-to-signal power ratio tau from the observation.

    We interpret tau as:

        tau ≈ Var(noise) / Var(signal)

    which matches the role it plays in the Wiener-like MTF weighting:

        W0(f) = |H(f)|^2 / (|H(f)|^2 + tau)

    Steps:
    - Estimate noise sigma via MAD on finite differences.
    - Estimate total variance of the observation.
    - Approximate signal variance as Var(y) - Var(noise), floored at small > 0.
    - Set tau = Var(noise) / Var(signal), then clamp to [tau_min, tau_max].

    Parameters
    ----------
    obs : np.ndarray
        Observed image, assumed float32 in [0, 1].
    tau_min : float
        Lower bound for tau (avoid degenerate zero).
    tau_max : float
        Upper bound for tau (avoid absurdly large NSR).

    Returns
    -------
    float
        Inferred tau to be passed into _build_mtf_weights.
    """
    obs_f = np.asarray(obs, dtype=np.float32)
    if obs_f.size == 0:
        return tau_min

    sigma_n = _estimate_noise_sigma_mad(obs_f)
    if sigma_n <= 0.0 or not np.isfinite(sigma_n):
        return tau_min

    var_y = float(np.var(obs_f))
    if not np.isfinite(var_y) or var_y <= 0.0:
        return tau_min

    var_n = sigma_n**2
    var_x = max(var_y - var_n, 1e-6)  # approximate signal variance

    tau = var_n / var_x  # noise-to-signal power ratio
    tau = float(np.clip(tau, tau_min, tau_max))
    return tau


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
    Build per-frequency weights from MTF^2 in a statistically motivated way.

    We interpret `tau` as an approximate noise-to-signal power ratio in the
    frequency domain and use a Wiener-like base weight

        W0(f) = |H(f)|^2 / (|H(f)|^2 + tau)

    so that frequencies with very low MTF receive less weight in the data term.
    We then optionally compress the dynamic range with `alpha` and clamp to
    [floor, 1] so that the data term is never completely ignored.

    Parameters
    ----------
    mtf2 : np.ndarray
        Array of |H(f)|^2 for the current kernel and image size.
    tau : float, optional
        Approximate noise-to-signal power ratio in the frequency domain.
        Larger values make the weighting more conservative (more downweighting
        of low-MTF frequencies).
    alpha : float, optional
        Exponent for dynamic-range compression. alpha < 1 flattens the range,
        alpha = 1 keeps the Wiener weights as-is.
    floor : float, optional
        Minimum allowed weight. Ensures no frequency is completely discarded.

    Returns
    -------
    np.ndarray
        Frequency weights W(f) with the same shape as mtf2, dtype float32.
    """
    # Ensure numeric stability and correct dtype
    mtf2 = np.asarray(mtf2, dtype=np.float32)
    if mtf2.size == 0:
        return np.ones_like(mtf2, dtype=np.float32)

    # Clip negative numerical noise
    mtf2_clipped = np.clip(mtf2, 0.0, None)
    mtf2_max = float(mtf2_clipped.max())
    if not np.isfinite(mtf2_max) or mtf2_max <= 0.0:
        # Degenerate kernel: fall back to uniform weighting
        return np.ones_like(mtf2_clipped, dtype=np.float32)

    # --- NEW: Wiener-like base weight ---------------------------------------
    # tau approximates noise power: when |H|^2 << tau, W0 ~ 0; when |H|^2 >> tau, W0 ~ 1.
    tau_eff = max(float(tau), 1e-12)
    W0 = mtf2_clipped / (mtf2_clipped + tau_eff)

    # --- NEW: Optional dynamic-range compression using alpha ----------------
    if alpha != 1.0:
        W = np.power(W0, alpha, dtype=np.float32)
    else:
        W = W0.astype(np.float32)

    # --- Same as before: enforce a floor to never completely ignore data ----
    W = np.clip(W, floor, 1.0, out=W)
    return W



def _mtf_quality_from_kernel(
    kernel: np.ndarray,
    image_shape: tuple[int, int],
    tau: float = 1e-3,
) -> float:
    """
    Compute a scalar MTF 'quality' score in [0, 1] from the kernel and image shape.

    Compared to a plain global average, this version:
    - Computes MTF^2 over the image's frequency grid.
    - Normalizes to [0, 1].
    - Focuses on a mid-high frequency band (in normalized radial frequency),
      where blur is most damaging.
    - Ignores frequencies where the normalized MTF^2 is below `tau`.

    The result is a more informative measure of how well the optics preserve
    useful detail, especially for motion / coded blur patterns.

    Parameters
    ----------
    kernel : np.ndarray
        Blur kernel (PSF).
    image_shape : tuple[int, int]
        (H, W) of the corresponding image domain.
    tau : float, optional
        Threshold on normalized MTF^2. Frequencies with MTF^2 <= tau are ignored.

    Returns
    -------
    float
        MTF quality score Q in [0, 1]. Lower means worse mid–high-frequency MTF.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    if H <= 0 or W <= 0:
        return 0.0

    mtf2 = _compute_mtf2(kernel, (H, W)).astype(np.float32)
    mtf2 = np.clip(mtf2, 0.0, None)

    mtf2_max = float(mtf2.max())
    if not np.isfinite(mtf2_max) or mtf2_max <= 0.0:
        # Degenerate kernel
        return 0.0

    # Normalize MTF^2 to [0, 1]
    mtf2_norm = mtf2 / (mtf2_max + 1e-8)

    # --- Build a normalized radial frequency grid ---------------------------
    # Frequencies in cycles per pixel along each axis
    fy = np.fft.fftfreq(H, d=1.0)  # shape (H,)
    fx = np.fft.fftfreq(W, d=1.0)  # shape (W,)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    rad = np.sqrt(FX**2 + FY**2)

    # Normalize radius to [0, 1]
    rad_max = float(rad.max())
    if not np.isfinite(rad_max) or rad_max <= 0.0:
        return 0.0
    rad_norm = rad / rad_max

    # --- Focus on a mid–high frequency band --------------------------------
    # E.g. 0.2–0.8 of the radial range: ignore DC and extreme Nyquist edge.
    band_lo = 0.2
    band_hi = 0.8
    band_mask = (rad_norm >= band_lo) & (rad_norm <= band_hi)

    # --- Combine band selection with MTF threshold --------------------------
    mtf_mask = (mtf2_norm > tau)
    mask = band_mask & mtf_mask

    if not mask.any():
        # Fallback: if mid–high band is empty, try the global mask
        if mtf_mask.any():
            return float(mtf2_norm[mtf_mask].mean())
        return 0.0

    Q = float(mtf2_norm[mask].mean())
    # Ensure Q in [0, 1]
    if not np.isfinite(Q):
        return 0.0
    return float(np.clip(Q, 0.0, 1.0))



def _mtf_adaptive_sigma_bounds(Q: float) -> tuple[float, float]:
    """
    Map an MTF quality scalar Q (~[0, 1]) to (sigma_max, sigma_min) for DRUNet /
    score-based priors.

    Lower Q (worse mid-high-frequency MTF) -> larger sigmas (stronger prior).
    Higher Q (better MTF) -> smaller sigmas (we trust the data more).

    Compared to a purely linear mapping, this version uses a mild nonlinearity
    to avoid over-penalizing moderate-quality kernels while still giving a
    clear boost in prior strength for very poor MTF.

    Returns (sigma_max, sigma_min) in the same units as your DRUNet / diffusion
    sigma convention (e.g. [0, 50] for pixel-noise-like standard deviation).
    """
    # Base ranges (you can tune these to your dataset)
    SIGMA_MAX_LO, SIGMA_MAX_HI = 25.0, 40.0
    SIGMA_MIN_LO, SIGMA_MIN_HI = 8.0, 20.0

    # Clamp Q to [0, 1]
    Q_clamped = float(np.clip(Q, 0.0, 1.0))

    # --- Mild nonlinearity: emphasize differences near Q ~ 0 ----------------
    # Q_eff = Q^beta with beta < 1 stretches lower values and compresses higher ones.
    beta = 0.7
    Q_eff = Q_clamped**beta

    # Interpolate between "worst" and "best" according to 1 - Q_eff
    # Q_eff = 1   -> (SIGMA_MAX_LO, SIGMA_MIN_LO)
    # Q_eff = 0   -> (SIGMA_MAX_HI, SIGMA_MIN_HI)
    t = 1.0 - Q_eff

    sigma_max = SIGMA_MAX_LO + t * (SIGMA_MAX_HI - SIGMA_MAX_LO)
    sigma_min = SIGMA_MIN_LO + t * (SIGMA_MIN_HI - SIGMA_MIN_LO)

    return float(sigma_max), float(sigma_min)



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
        # NEW: infer tau from the observation (noise-to-signal ratio)
        tau_est = _infer_tau_from_observation(obs)
        W_2d = _build_mtf_weights(mtf2, tau=tau_est)
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
        # NEW: infer tau from the observation for diffusion ADMM as well
        tau_est = _infer_tau_from_observation(obs)
        W_2d = _build_mtf_weights(mtf2, tau=tau_est)
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

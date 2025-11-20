from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Literal

import numpy as np


@dataclass
class PhysicsContext:
    """Structured metadata describing the physics of a forward pass for one pattern."""

    pattern: str
    taps: int
    blur_length_px: float
    duty_cycle: float
    photon_budget: float
    read_noise_sigma: float
    random_seed: int
    code: np.ndarray
    psf: np.ndarray
    kernel: np.ndarray
    mtf: np.ndarray
    ssnr: np.ndarray
    scene_stats: Mapping[str, float] = field(default_factory=dict)
    blurred_stats: Mapping[str, float] = field(default_factory=dict)
    noisy_stats: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def mtf_band_averages(self, num_bands: int = 8) -> np.ndarray:
        """Compute radially averaged MTF magnitudes in num_bands frequency bins."""
        if num_bands <= 0:
            raise ValueError("num_bands must be positive.")
        mtf = np.asarray(self.mtf, dtype=np.float64)
        h, w = mtf.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        bins = np.linspace(0.0, float(radius.max()) + 1e-6, num_bands + 1)
        band_means = np.zeros(num_bands, dtype=np.float64)
        for idx in range(num_bands):
            mask = (radius >= bins[idx]) & (radius < bins[idx + 1])
            if np.any(mask):
                band_means[idx] = float(mtf[mask].mean())
        return band_means

    def ssnr_band_averages(self, num_bands: int = 8) -> np.ndarray:
        """Compute radially averaged spectral SNR in num_bands frequency bins."""
        ssnr = np.asarray(self.ssnr, dtype=np.float64)
        if ssnr.ndim == 3:
            ssnr = ssnr.mean(axis=-1)
        h, w = ssnr.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        bins = np.linspace(0.0, float(radius.max()) + 1e-6, num_bands + 1)
        band_means = np.zeros(num_bands, dtype=np.float64)
        for idx in range(num_bands):
            mask = (radius >= bins[idx]) & (radius < bins[idx + 1])
            if np.any(mask):
                band_means[idx] = float(ssnr[mask].mean())
        return band_means

    def summary_metrics(self) -> Dict[str, float]:
        """Return quick-look summary statistics used by schedulers."""
        mtf = np.asarray(self.mtf, dtype=np.float64)
        ssnr = np.asarray(self.ssnr, dtype=np.float64)
        if ssnr.ndim == 3:
            ssnr = ssnr.mean(axis=-1)
        stats = {
            "mean_mtf": float(mtf.mean()),
            "min_mtf": float(mtf.min()),
            "max_mtf": float(mtf.max()),
            "mean_ssnr": float(ssnr.mean()),
            "median_ssnr": float(np.median(ssnr)),
            "scene_variance": float(self.scene_stats.get("variance", 0.0)),
            "noisy_variance": float(self.noisy_stats.get("variance", 0.0)),
        }
        return stats


@dataclass
class SchedulerDecision:
    """Result for a single ADMM iteration."""

    denoiser_type: str
    denoiser_weight: float
    rho: float
    prior_choice: Literal["denoiser", "diffusion"] = "denoiser"
    sigma: Optional[float] = None
    sigma_scale: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class PhysicsAwareScheduler:
    """Base class for physics-aware schedulers."""

    def __init__(self, total_iterations: int, *, default_denoiser: str = "tiny") -> None:
        self.total_iterations = max(1, int(total_iterations))
        self.default_denoiser = default_denoiser
        self._contexts: Dict[str, PhysicsContext] = {}

    def set_contexts(self, contexts: Mapping[str, PhysicsContext]) -> None:
        """Provide the per-pattern contexts before running ADMM."""
        self._contexts = dict(contexts)

    def get_context(self, pattern: str) -> Optional[PhysicsContext]:
        return self._contexts.get(pattern)

    def reset(self) -> None:
        """Hook for subclasses that maintain per-run state."""

    def plan_iteration(self, pattern: str, iteration: int) -> SchedulerDecision:
        """Return scheduling parameters for the requested iteration."""
        raise NotImplementedError

    def _normalized_iteration(self, iteration: int) -> float:
        idx = np.clip(iteration - 1, 0, self.total_iterations - 1)
        if self.total_iterations == 1:
            return 1.0
        return idx / float(self.total_iterations - 1)


class HeuristicPhysicsAwareScheduler(PhysicsAwareScheduler):
    """Simple deterministic scheduler based on MTF/SNR heuristics."""

    def __init__(
        self,
        total_iterations: int,
        *,
        weak_mtf_threshold: float = 0.25,
        low_photon_budget: float = 600.0,
        aggressive_weight: float = 0.9,
        conservative_weight: float = 0.55,
        default_rho: float = 0.4,
        sigma_base: float = 0.18,
        sigma_min: float = 0.02,
        pattern_rho_schedule: Optional[Mapping[str, tuple[float, float]]] = None,
        pattern_weight_schedule: Optional[Mapping[str, tuple[float, float]]] = None,
        pattern_sigma_scale_schedule: Optional[Mapping[str, tuple[float, float]]] = None,
    ) -> None:
        super().__init__(total_iterations)
        self.weak_mtf_threshold = weak_mtf_threshold
        self.low_photon_budget = low_photon_budget
        self.aggressive_weight = aggressive_weight
        self.conservative_weight = conservative_weight
        self.default_rho = default_rho
        self.sigma_base = sigma_base
        self.sigma_min = sigma_min
        # Optional per-pattern curves (start -> end multipliers) applied across iterations.
        default_rho_sched = {"legendre": (1.1, 0.9)}
        default_weight_sched = {"legendre": (1.05, 0.9)}
        default_sigma_sched = {
            "legendre": (0.9, 0.6),
            "box": (1.0, 0.7),
            "random": (1.0, 0.7),
        }
        self.pattern_rho_schedule = dict(default_rho_sched if pattern_rho_schedule is None else pattern_rho_schedule)
        self.pattern_weight_schedule = dict(
            default_weight_sched if pattern_weight_schedule is None else pattern_weight_schedule
        )
        self.pattern_sigma_scale_schedule = dict(
            default_sigma_sched if pattern_sigma_scale_schedule is None else pattern_sigma_scale_schedule
        )

    def plan_iteration(self, pattern: str, iteration: int) -> SchedulerDecision:
        ctx = self.get_context(pattern)
        if ctx is None:
            sigma, sigma_scale = self._sigma_schedule(iteration, aggressive=False)
            return SchedulerDecision(
                denoiser_type=self.default_denoiser,
                denoiser_weight=self._apply_pattern_schedule(
                    pattern, self.conservative_weight, self.pattern_weight_schedule, iteration
                ),
                rho=self._apply_pattern_schedule(pattern, self.default_rho, self.pattern_rho_schedule, iteration),
                sigma=sigma,
                sigma_scale=self._apply_sigma_scale_schedule(pattern, sigma_scale, iteration),
                extras={"reason": "default"},
            )

        summary = ctx.summary_metrics()
        mtf_mean = summary["mean_mtf"]
        hard_case = mtf_mean < self.weak_mtf_threshold or ctx.photon_budget <= self.low_photon_budget

        denoiser_type = "drunet_gray" if hard_case else self.default_denoiser
        prior_choice: Literal["denoiser", "diffusion"] = "diffusion" if hard_case else "denoiser"

        weight = self.aggressive_weight if hard_case else self.conservative_weight
        rho = self.default_rho * (0.7 if hard_case else 1.0)
        weight = self._apply_pattern_schedule(pattern, weight, self.pattern_weight_schedule, iteration)
        rho = self._apply_pattern_schedule(pattern, rho, self.pattern_rho_schedule, iteration)
        sigma, sigma_scale = self._sigma_schedule(iteration, aggressive=hard_case, summary=summary, pattern=pattern)
        sigma_scale = self._apply_sigma_scale_schedule(pattern, sigma_scale, iteration)

        extras = {
            "mean_mtf": mtf_mean,
            "min_mtf": summary["min_mtf"],
            "mean_ssnr": summary["mean_ssnr"],
            "photon_budget": ctx.photon_budget,
        }

        return SchedulerDecision(
            denoiser_type=denoiser_type,
            denoiser_weight=float(np.clip(weight, 0.05, 1.0)),
            rho=float(np.clip(rho, 1e-4, 5.0)),
            prior_choice=prior_choice,
            sigma=sigma,
            sigma_scale=sigma_scale,
            extras=extras,
        )

    def _sigma_schedule(
        self,
        iteration: int,
        *,
        aggressive: bool,
        summary: Optional[Mapping[str, float]] = None,
        pattern: Optional[str] = None,
    ) -> tuple[float, float]:
        frac = self._normalized_iteration(iteration)
        start = 0.35 if aggressive else self.sigma_base
        end = 0.05 if aggressive else self.sigma_min

        if summary is not None:
            ssnr = summary.get("mean_ssnr", 0.0)
            adjust = 1.0 + 0.3 * float(1.0 / (1.0 + max(ssnr, 1e-3)))
            start *= adjust
            end *= adjust
        if pattern == "legendre" and aggressive:
            start *= 0.9
            end *= 0.95

        sigma = (1.0 - frac) * start + frac * end
        sigma = float(np.clip(sigma, 1e-3, 1.0))
        scale = sigma / max(self.sigma_base, 1e-3)
        return sigma, float(scale)

    def _apply_pattern_schedule(
        self,
        pattern: str,
        base_value: float,
        schedule: Mapping[str, tuple[float, float]],
        iteration: int,
    ) -> float:
        if not schedule:
            return base_value
        if pattern not in schedule:
            return base_value
        start, end = schedule[pattern]
        frac = self._normalized_iteration(iteration)
        scale = (1.0 - frac) * start + frac * end
        return float(base_value * scale)

    def _apply_sigma_scale_schedule(
        self,
        pattern: str,
        sigma_scale: float,
        iteration: int,
    ) -> float:
        if not self.pattern_sigma_scale_schedule:
            return sigma_scale
        if pattern not in self.pattern_sigma_scale_schedule:
            return sigma_scale
        start, end = self.pattern_sigma_scale_schedule[pattern]
        frac = self._normalized_iteration(iteration)
        mult = (1.0 - frac) * start + frac * end
        return float(sigma_scale * mult)


__all__ = [
    "PhysicsContext",
    "SchedulerDecision",
    "PhysicsAwareScheduler",
    "HeuristicPhysicsAwareScheduler",
]

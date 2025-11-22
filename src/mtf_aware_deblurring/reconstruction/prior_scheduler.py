from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Literal

import numpy as np


@dataclass
class PhysicsContext:
    pattern: str
    taps: int
    blur_length_px: float
    photon_budget: float
    mtf: np.ndarray

    duty_cycle: float = 0.5
    read_noise_sigma: float = 0.0
    random_seed: int = 0
    code: Optional[np.ndarray] = None
    psf: Optional[np.ndarray] = None
    kernel: Optional[np.ndarray] = None
    ssnr: Optional[np.ndarray] = None

    scene_stats: Mapping[str, float] = field(default_factory=dict)
    blurred_stats: Mapping[str, float] = field(default_factory=dict) 
    noisy_stats: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def quality_metrics(self) -> Dict[str, float]:
        mtf = np.asarray(self.mtf, dtype=np.float64)
        mean_mtf = float(mtf.mean())
        optics_score = np.clip(mean_mtf * 3.0, 0.0, 1.0) 
        snr_score = np.clip((self.photon_budget - 100) / 1900, 0.0, 1.0)
        return {"optics": optics_score, "snr": snr_score}


@dataclass
class SchedulerDecision:
    denoiser_type: str
    denoiser_weight: float
    rho: float
    prior_choice: Literal["denoiser", "diffusion"] = "denoiser"
    sigma: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class PhysicsAwareScheduler:
    def __init__(self, total_iterations: int, *, default_denoiser: str = "tiny") -> None:
        self.total_iterations = max(1, int(total_iterations))
        self.default_denoiser = default_denoiser
        self._contexts: Dict[str, PhysicsContext] = {}

    def set_contexts(self, contexts: Mapping[str, PhysicsContext]) -> None:
        self._contexts = dict(contexts)

    def get_context(self, pattern: str) -> Optional[PhysicsContext]:
        return self._contexts.get(pattern)

    def reset(self) -> None:
        pass

    def plan_iteration(self, pattern: str, iteration: int) -> SchedulerDecision:
        raise NotImplementedError


class AdaptivePhysicsScheduler(PhysicsAwareScheduler):
    """
    Performance-Optimized Relative Scheduler.
    Allows a 'Deep Dive' start (low rho) for texture recovery, 
    but clamps bad optics to prevent explosions.
    """

    def __init__(
        self,
        total_iterations: int,
        *,
        base_rho: float = 0.8,
        base_weight: float = 1.0,
    ) -> None:
        super().__init__(total_iterations, default_denoiser="tiny")
        self.base_rho = base_rho
        self.base_weight = base_weight

    def plan_iteration(self, pattern: str, iteration: int) -> SchedulerDecision:
        # 1. Get Physics Data
        ctx = self.get_context(pattern)
        optics_score = 0.5
        snr_score = 0.5
        
        if ctx is not None:
            metrics = ctx.quality_metrics()
            optics_score = metrics["optics"]
            snr_score = metrics["snr"]

        # 2. Calculate RELATIVE Multipliers (Fixed for Sharpness)
        # ---------------------------------------------------------
        
        # Start Multiplier: 
        # Ideally we want 0.02x (Deep Dive) to recover texture.
        # BUT, if optics are bad (score < 0.15), 0.02x causes explosions.
        # So we clamp the floor dynamically.
        
        target_start = 0.02 
        if optics_score < 0.15:
            # Box/Diffuser case: Needs a safer floor
            target_start = 0.15 
            
        physics_mult_start = target_start
        
        # End Multiplier:
        # We need to end HIGH (>1.0) to lock in the sharpness.
        physics_mult_end = 1.2    

        # 3. Geometric Ramp
        t = iteration - 1
        T = max(self.total_iterations - 1, 1)
        frac = np.clip(t / T, 0.0, 1.0)
        
        # Interpolate
        current_mult = physics_mult_start * ((physics_mult_end / physics_mult_start) ** frac)
        current_rho = self.base_rho * current_mult

        # 4. Weight Modulation
        # Keep consistent: bad optics = trust denoiser more
        weight_modifier = 1.0 + 0.2 * (1.0 - optics_score)
        current_weight = np.clip(self.base_weight * weight_modifier, 0.0, 1.0)

        return SchedulerDecision(
            denoiser_type=self.default_denoiser,
            denoiser_weight=float(current_weight),
            rho=float(current_rho),
            prior_choice="denoiser",
            sigma=None,
            extras={
                "frac": frac, 
                "optics_q": optics_score, 
                "rho_mult": current_mult
            },
        )

__all__ = [
    "PhysicsContext",
    "SchedulerDecision",
    "PhysicsAwareScheduler",
    "AdaptivePhysicsScheduler",
]

from __future__ import annotations

import numpy as np


def add_poisson_gaussian(
    img: np.ndarray,
    photon_budget: float = 2000.0,
    read_noise_sigma: float = 0.01,
    clip: bool = True,
    rng_seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    img01 = np.clip(img, 0, 1)
    lam = img01 * photon_budget
    y_counts = rng.poisson(lam=lam)
    y = y_counts / photon_budget
    y = y + rng.normal(0.0, read_noise_sigma, size=img.shape)
    if clip:
        y = np.clip(y, 0, 1)
    return y


__all__ = ["add_poisson_gaussian"]

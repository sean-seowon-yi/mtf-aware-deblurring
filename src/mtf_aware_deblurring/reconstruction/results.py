from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray
    psnr: float


__all__ = ["ReconstructionResult"]

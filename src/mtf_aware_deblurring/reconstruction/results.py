from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray
    psnr: float
    ssim: float
    lpips: Optional[float] = None
    trace: Optional[List[Dict[str, Any]]] = None


__all__ = ["ReconstructionResult"]

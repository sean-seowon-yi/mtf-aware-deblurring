from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m, M = x.min(), x.max()
    if M > m:
        return (x - m) / (M - m)
    return np.zeros_like(x)


@dataclass
class SyntheticData:
    s_type: str = "Checker Board"
    height: int = 256
    width: int = 256

    def check_pattern(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        y, x = np.indices((self.height, self.width))
        cb = ((x // 16 + y // 16) % 2).astype(float)

        edge = (x - 0.7 * y > self.width // 3).astype(float)

        dots = np.zeros((self.height, self.width), float)
        for _ in range(30):
            i = rng.integers(0, self.height)
            j = rng.integers(0, self.width)
            dots[i, j] = 1.0

        img = 0.55 * cb + 0.35 * edge + 0.10 * normalize01(rng.normal(size=(self.height, self.width)))
        img = np.clip(img + 0.6 * dots, 0, 1)
        return img

    def ring_pattern(self, seed: int = 0, freq: float = 0.03) -> np.ndarray:
        rng = np.random.default_rng(seed)
        y, x = np.indices((self.height, self.width))
        cx, cy = self.width // 2, self.height // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        rings = 0.5 + 0.5 * np.sin(2 * np.pi * freq * r)

        noise = normalize01(rng.normal(size=(self.height, self.width)))
        img = 0.9 * rings + 0.1 * noise
        return np.clip(img, 0, 1)

    def create_img(self, seed: int = 0) -> np.ndarray:
        if self.s_type == "Checker Board":
            return self.check_pattern(seed)
        if self.s_type == "Rings":
            return self.ring_pattern(seed)
        raise ValueError(f"Unknown image type: {self.s_type}")


__all__ = ["SyntheticData", "normalize01"]

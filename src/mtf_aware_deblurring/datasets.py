from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .utils import load_input_image


@dataclass
class DIV2KDataset:
    """Simple iterable over DIV2K low-resolution frames.

    Parameters
    ----------
    root:
        Path to the directory containing DIV2K assets. We expect the low-resolution
        folders (e.g. ``DIV2K_train_LR_bicubic/X2``) to live underneath this root.
    subset:
        ``"train"`` or ``"valid"`` (matches the DIV2K naming convention).
    degradation:
        Degradation label in the folder name, e.g. ``"bicubic"`` or ``"unknown"``.
    scale:
        Upscale factor folder, typically ``"X2"``, ``"X3"``, or ``"X4"``.
    limit:
        Optional maximum number of frames to iterate (defaults to all).
    target_size:
        Optional square resize applied after loading. If ``None`` the native resolution is used.
    normalize:
        Whether to scale pixel intensities to ``[0, 1]``.

    The loader presents grayscale float32 arrays by reusing ``load_input_image``.
    """

    root: Path
    subset: str = "train"
    degradation: str = "bicubic"
    scale: str = "X2"
    limit: Optional[int] = None
    target_size: Optional[int] = 256
    normalize: bool = True
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg")

    def __post_init__(self) -> None:
        base_dir = (
            Path(self.root)
            / f"DIV2K_{self.subset}_LR_{self.degradation}"
            / self.scale.upper()
        )
        if not base_dir.exists():
            raise FileNotFoundError(
                f"DIV2K directory not found: {base_dir}. "
                "Point --div2k-root to the folder that contains DIV2K_*_LR_<method>/."
            )

        files: List[Path] = []
        for ext in self.extensions:
            files.extend(sorted(base_dir.glob(f"*{ext}")))
        if not files:
            raise FileNotFoundError(
                f"No images with extensions {self.extensions} found under {base_dir}."
            )
        if self.limit is not None:
            files = files[: self.limit]
        self.files: Tuple[Path, ...] = tuple(files)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[Tuple[Path, np.ndarray]]:
        for path in self.files:
            image = load_input_image(
                path,
                target_size=self.target_size,
                normalize=self.normalize,
            )
            yield path, image


__all__ = ["DIV2KDataset"]

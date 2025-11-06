from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def default_output_dir() -> Path:
    try:
        base = Path(__file__).resolve().parent
    except NameError:  # pragma: no cover - fallback for interactive sessions
        base = Path.cwd()
    return base / "forward_model_outputs"


def configure_matplotlib_defaults() -> None:
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = False


def axes_as_list(axs):
    if isinstance(axs, np.ndarray):
        return list(np.atleast_1d(axs).ravel())
    return [axs]


def finalize_figure(fig, outfile: Optional[Path], show_plots: bool) -> None:
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def load_input_image(
    image_path: Path,
    *,
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = True,
    mode: str = "grayscale",
) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    image_path: Path to the image file.
    target_size: Optional (H, W) tuple or single integer for square resize.
    normalize: Whether to scale intensities to [0, 1].
    mode: 'grayscale' or 'rgb'.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pillow is required to load external images. Install it via `pip install pillow`."
        ) from exc

    img = Image.open(image_path)
    mode = mode.lower()
    if mode == "grayscale":
        img = img.convert("F")
    elif mode == "rgb":
        img = img.convert("RGB")
    else:
        raise ValueError("mode must be 'grayscale' or 'rgb'")

    if target_size is not None:
        if isinstance(target_size, int):
            size = (target_size, target_size)
        else:
            size = (int(target_size[0]), int(target_size[1]))
        img = img.resize(size, Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32)
    if normalize:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    return arr


__all__ = [
    "default_output_dir",
    "configure_matplotlib_defaults",
    "axes_as_list",
    "finalize_figure",
    "load_input_image",
]

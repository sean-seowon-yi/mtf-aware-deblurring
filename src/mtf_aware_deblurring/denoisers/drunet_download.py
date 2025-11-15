import os
import urllib.request
from typing import Literal

# Hugging Face DRUNet weights (deepinv/drunet)
HF_BASE_URL = "https://huggingface.co/deepinv/drunet/resolve/main"
DRUNET_COLOR_URL = f"{HF_BASE_URL}/drunet_color.pth"
DRUNET_GRAY_URL  = f"{HF_BASE_URL}/drunet_gray.pth"

# Default cache directory under user home
DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "mtf_aware_deblurring",
    "drunet",
)


def _download_file(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"[DRUNet] Downloading from {url}")
    print(f"[DRUNet] Saving to {dst_path}")
    urllib.request.urlretrieve(url, dst_path)
    print("[DRUNet] Download complete.")


def ensure_drunet_weight(
    mode: Literal["color", "gray"] = "color",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> str:
    """
    Ensure DRUNet pretrained weight is available locally.

    Parameters
    ----------
    mode : "color" or "gray"
        Which DRUNet variant to fetch.
    cache_dir : str
        Directory where weights are cached.

    Returns
    -------
    weight_path : str
        Local filesystem path to the downloaded .pth file.
    """
    if mode == "color":
        filename = "drunet_color.pth"
        url = DRUNET_COLOR_URL
    elif mode == "gray":
        filename = "drunet_gray.pth"
        url = DRUNET_GRAY_URL
    else:
        raise ValueError(f"Unknown DRUNet mode: {mode}")

    weight_path = os.path.join(cache_dir, filename)

    if not os.path.exists(weight_path):
        _download_file(url, weight_path)
    else:
        # You can remove this print if you want it silent
        print(f"[DRUNet] Using cached weights at {weight_path}")

    return weight_path


def ensure_drunet_color(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Shortcut: ensure color DRUNet weights are present locally."""
    return ensure_drunet_weight("color", cache_dir=cache_dir)


def ensure_drunet_gray(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Shortcut: ensure gray DRUNet weights are present locally."""
    return ensure_drunet_weight("gray", cache_dir=cache_dir)

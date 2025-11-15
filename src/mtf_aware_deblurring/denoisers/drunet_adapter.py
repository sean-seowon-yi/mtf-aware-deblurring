# mtf_aware_deblurring/denoisers/drunet_adapter.py

import numpy as np
import torch

from .drunet import load_pretrained_drunet_color, load_pretrained_drunet_gray
from .drunet_download import ensure_drunet_color, ensure_drunet_gray


def build_drunet_denoiser(
    mode: str = "color",
    device: str = "cuda",
    sigma_init: float = 15.0,
    sigma_schedule=None,      # Optional: (t, T) -> sigma_t
    iterations: int | None = None,
):
    """
    Returns a numpy-friendly callable: f(x_np) -> x_denoised_np.

    If sigma_schedule is provided, we compute sigma_t = sigma_schedule(t, iterations)
    where t = denoiser call index (1-based). Otherwise we use sigma_init.
    """

    if mode == "color":
        weight_path = ensure_drunet_color()
        model = load_pretrained_drunet_color(
            weight_path=weight_path,
            default_sigma=sigma_init,
            device=device,
        )
        n_channels = 3
    elif mode == "gray":
        weight_path = ensure_drunet_gray()
        model = load_pretrained_drunet_gray(
            weight_path=weight_path,
            default_sigma=sigma_init,
            device=device,
        )
        n_channels = 1
    else:
        raise ValueError(f"Unknown DRUNet mode: {mode}")

    model.eval()

    # keep track of how many times we’ve called the denoiser
    state = {"t": 0}
    T = iterations

    def denoise_np(x_np: np.ndarray) -> np.ndarray:
        state["t"] += 1
        t = state["t"]

        # decide sigma_t
        if sigma_schedule is not None and T is not None:
            sigma_t = float(sigma_schedule(t, T))
        else:
            sigma_t = float(sigma_init)

        if x_np.ndim == 2:
            x_np_ = x_np[..., None]
        elif x_np.ndim == 3:
            x_np_ = x_np
        else:
            raise ValueError(f"Unsupported input shape for DRUNet: {x_np.shape}")

        H, W, C = x_np_.shape
        if C != n_channels:
            if C == 1 and n_channels == 3:
                x_np_ = np.repeat(x_np_, 3, axis=2)
            elif C == 3 and n_channels == 1:
                x_np_ = x_np_.mean(axis=2, keepdims=True)
            else:
                raise ValueError(
                    f"Channel mismatch: input has {C} channels, DRUNet expects {n_channels}"
                )

        x_t = torch.from_numpy(x_np_.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        x_t = x_t.to(model.device)

        with torch.no_grad():
            # pass sigma_t explicitly (in [0, 255] convention)
            y_t = model(x_t, sigma=sigma_t)

        y_np = y_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if x_np.ndim == 2 or n_channels == 1:
            y_np = y_np[..., 0]

        return np.clip(y_np, 0.0, 1.0).astype(np.float32)

    return denoise_np

# guided_uncond_prior.py
# Adapter around OpenAI guided-diffusion 256x256 unconditional model
# to provide a denoiser-like prior interface for ADMM / plug-and-play.

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Use vendored guided_diffusion implementation:
# mtf_aware_deblurring/external/guided_diffusion/...
from ..external.guided_diffusion import script_util

logger = logging.getLogger(__name__)


# --- Checkpoint download / caching ------------------------------------------------

_GUIDED_UNCOND_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
)
_GUIDED_UNCOND_FILENAME = "256x256_diffusion_uncond.pt"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "mtf_aware_deblurring" / "guided_diffusion"


def ensure_guided_diffusion_uncond_weights(
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Ensure the 256x256 unconditional guided-diffusion checkpoint exists
    in a local cache directory. Download if missing.

    Returns
    -------
    ckpt_path : Path
        Path to the local .pt checkpoint file.
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / _GUIDED_UNCOND_FILENAME

    if ckpt_path.exists():
        return ckpt_path

    logger.info(
        "Downloading guided-diffusion 256x256 unconditional checkpoint "
        "to %s ... (2.2 GB, one-time download)",
        ckpt_path,
    )

    urllib.request.urlretrieve(_GUIDED_UNCOND_URL, ckpt_path)
    logger.info("Finished downloading guided-diffusion checkpoint.")

    return ckpt_path


# --- Adapter class ----------------------------------------------------------------


class GuidedDiffusionUncondPrior(nn.Module):
    """
    Wrapper around OpenAI's 256x256 unconditional guided-diffusion model.

    Exposes a denoiser-like interface for [0,1] images, driven by a
    *normalized prior-strength level* in [0, 1]:

        x_clean = prior.denoise_level_01(x_noisy, level)

    where:
        - level ~ 0.0 -> very weak prior (small t, close to identity)
        - level ~ 1.0 -> strong prior (large t, more generative)

    Internally, this maps `level` to a diffusion timestep t and uses
    the model to predict x_0 from x_t.
    """

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        device: str = "cuda",
        image_size: int = 256,
        use_fp16: bool = True,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)

        # --- USE DEFAULTS THEN OVERRIDE (this avoids missing-arg errors) ---
        defaults = script_util.model_and_diffusion_defaults()

        # Override to match the 256x256 unconditional config from README
        defaults.update(
            dict(
                image_size=image_size,
                class_cond=False,             # unconditional
                learn_sigma=True,
                diffusion_steps=1000,
                noise_schedule="linear",
                num_channels=256,
                num_res_blocks=2,
                attention_resolutions="32,16,8",
                resblock_updown=True,
                use_scale_shift_norm=True,
                use_fp16=use_fp16,
                # channel_mult, num_heads, etc. left as defaults.
            )
        )

        # Some versions include num_classes in defaults; make sure we don't pass it
        defaults.pop("num_classes", None)

        # Finally create model and diffusion using the full config dict
        self.model, self.diffusion = script_util.create_model_and_diffusion(**defaults)

        self.model.to(self.device)
        if use_fp16 and self.device.type != "cpu":
            if hasattr(self.model, "convert_to_fp16"):
                self.model.convert_to_fp16()  # type: ignore[attr-defined]

        if ckpt_path is None:
            ckpt_path = ensure_guided_diffusion_uncond_weights()

        logger.info("Loading guided-diffusion weights from %s", ckpt_path)
        state_dict = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Cache noise schedule buffers for convenience
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        self.register_buffer(
            "sqrt_alphas_cumprod",
            _to_tensor(self.diffusion.sqrt_alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            _to_tensor(self.diffusion.sqrt_one_minus_alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "alphas_cumprod",
            _to_tensor(self.diffusion.alphas_cumprod),
            persistent=False,
        )

        self.num_timesteps: int = int(self.diffusion.num_timesteps)

        # Optional: precompute sigma range for debugging / logging
        with torch.no_grad():
            ts = torch.arange(self.num_timesteps, device=self.device, dtype=torch.long)
            sigmas = self._t_to_sigma(ts)
            self.sigma_min_: float = float(sigmas.min().item())
            self.sigma_max_: float = float(sigmas.max().item())
        logger.info(
            "GuidedDiffusionUncondPrior: sigma_t in [%.4f, %.4f]",
            self.sigma_min_,
            self.sigma_max_,
        )

    # --- sigma <-> timestep helpers ------------------------------------------------

    def _t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map diffusion timestep t to an effective noise sigma_t.

        Using:
            x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
            sigma_t ≈ sqrt(1-ᾱ_t) / sqrt(ᾱ_t)
        """
        sqrt_ab = self.sqrt_alphas_cumprod[t]           # [B]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]  # [B]
        return sqrt_one_minus / (sqrt_ab + 1e-8)

    def _sigma_to_t(self, sigma: float) -> int:
        """
        Map a continuous sigma to the nearest diffusion timestep t.

        NOTE: For plug-and-play, it is easier to work with a normalized
        level in [0,1]. This function is kept for debugging / inspection.
        """
        with torch.no_grad():
            ts = torch.arange(self.num_timesteps, device=self.device, dtype=torch.long)
            sigmas = self._t_to_sigma(ts)  # [T]
            t = (sigmas - sigma).abs().argmin()
        return int(t.item())

    # --- core prediction: x0 from (x_t, t) ----------------------------------------

    def _predict_x0_from_xt(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict x0 from a noisy sample x_t at time t, using the model's epsilon
        prediction and the diffusion helper _predict_xstart_from_eps.
        """
        with torch.no_grad():
            # model_out has shape [N, 2*C, H, W] when learn_sigma=True
            model_out = self.model(x_t, t)
        eps, _ = torch.split(model_out, x_t.shape[1], dim=1)
        x0_pred = self.diffusion._predict_xstart_from_eps(x_t, t, eps)
        return x0_pred

    # --- public denoiser-style APIs ----------------------------------------------

    def denoise_at_t(self, x_m: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Use the unconditional diffusion model as a denoiser at a given timestep.

        Parameters
        ----------
        x_m : torch.Tensor
            Tensor of shape [N, C, H, W] in the model's *data space* ([-1, 1]).
        t_index : int
            Diffusion timestep (0 .. num_timesteps-1).

        Returns
        -------
        x0_denoised : torch.Tensor
            Denoised estimate of x_0 in [-1, 1].
        """
        x = x_m.to(self.device)
        assert x.dim() == 4, "Expected [N, C, H, W]"

        t = torch.full(
            (x.size(0),),
            t_index,
            device=self.device,
            dtype=torch.long,
        )

        # Sample x_t from the forward process around x_0 = x
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        x_t = sqrt_ab * x + sqrt_one_minus * eps

        x0_pred = self._predict_x0_from_xt(x_t, t)
        return x0_pred.clamp(-1.0, 1.0)

    # --- convenience: [0,1] <-> [-1,1] wrappers -----------------------------------

    def _to_model_space(self, x_01: torch.Tensor) -> torch.Tensor:
        # [0,1] -> [-1,1]
        return x_01 * 2.0 - 1.0

    def _from_model_space(self, x_m: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [0,1]
        return (x_m + 1.0) / 2.0

    # <<< CHANGED: new normalized level-based API >>>
    def denoise_level_01(self, x_01: torch.Tensor, level: float) -> torch.Tensor:
        """
        Denoise images given in [0,1] using a normalized prior-strength level.

        Parameters
        ----------
        x_01 : torch.Tensor
            [N, C, H, W] tensor in [0,1].
        level : float
            Prior strength in [0,1], mapped to timesteps:
                0.0 ~ very small t (weak prior)
                1.0 ~ large t (strong prior)

        Returns
        -------
        x_denoised_01 : torch.Tensor
            Denoised tensor in [0,1].
        """
        level = float(max(0.0, min(1.0, level)))
        t_index = int(level * (self.num_timesteps - 1))

        x_m = self._to_model_space(x_01)
        x_m_denoised = self.denoise_at_t(x_m, t_index)
        return self._from_model_space(x_m_denoised).clamp(0.0, 1.0)

    # <<< OPTIONAL: keep sigma-based denoise_01 for backwards compatibility >>>
    def denoise_01(self, x_01: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Legacy sigma-based API. `sigma` is interpreted *in diffusion units*
        (not pixel noise). For plug-and-play, use `denoise_level_01` instead.
        """
        t_index = self._sigma_to_t(float(sigma))
        x_m = self._to_model_space(x_01)
        x_m_denoised = self.denoise_at_t(x_m, t_index)
        return self._from_model_space(x_m_denoised).clamp(0.0, 1.0)

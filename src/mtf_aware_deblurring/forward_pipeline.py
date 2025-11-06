from __future__ import annotations

from .metrics import spectral_snr
from .noise import add_poisson_gaussian
from .optics import fft_convolve2d, kernel2d_from_psf1d, motion_psf_from_code, mtf_from_kernel, otf2d, pad_to_shape
from .patterns import (
    is_prime,
    legendre_base_sequence,
    legendre_symbol,
    make_exposure_code,
    modified_legendre_sequence,
    next_prime,
    previous_prime,
    resolve_legendre_prime,
)
from .runner import DEFAULT_SEED, ForwardModelRunner, main, run_forward_model
from .synthetic import SyntheticData, normalize01
from .utils import (
    axes_as_list,
    configure_matplotlib_defaults,
    default_output_dir,
    finalize_figure,
    load_input_image,
)

__all__ = [
    "DEFAULT_SEED",
    "ForwardModelRunner",
    "SyntheticData",
    "normalize01",
    "run_forward_model",
    "main",
    "configure_matplotlib_defaults",
    "default_output_dir",
    "axes_as_list",
    "finalize_figure",
    "load_input_image",
    "is_prime",
    "next_prime",
    "previous_prime",
    "legendre_symbol",
    "resolve_legendre_prime",
    "legendre_base_sequence",
    "modified_legendre_sequence",
    "make_exposure_code",
    "pad_to_shape",
    "motion_psf_from_code",
    "kernel2d_from_psf1d",
    "fft_convolve2d",
    "otf2d",
    "mtf_from_kernel",
    "add_poisson_gaussian",
    "spectral_snr",
]


if __name__ == "__main__":
    main()

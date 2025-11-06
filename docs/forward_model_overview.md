# Forward Model Overview

This module is derived from the original Colab notebook and now lives across `mtf_aware_deblurring.runner` (or
the compatibility shim `mtf_aware_deblurring.forward_pipeline`).

## Key Components
- `SyntheticData`: Generates synthetic test scenes (checkerboard, rings) with optional random seeds.
- `DIV2KDataset` (`datasets.py`): Streams low-resolution DIV2K frames for real-image simulations.
- `make_exposure_code` (`patterns.py`): Builds box, random, and Modified Legendre Sequence (MLS) flutter codes.
- `motion_psf_from_code` (`optics.py`): Converts exposure codes into motion PSFs and 2D kernels.
- `ForwardModelRunner`: High-level orchestration that produces blurred/noisy images, PSF/OTF/MTF diagnostics, and spectral SNR plots.
- `run_forward_model`: Convenience function that instantiates the runner with default settings.

## Configurable Parameters
The runner accepts numerous keyword arguments for experiment control:

| Parameter | Description |
|-----------|-------------|
| `patterns` | Sequence of exposure codes (`"box"`, `"random"`, `"legendre"`, or custom masks). |
| `T` | Number of code taps (exposure bins). |
| `blur_length_px` | Apparent motion length in pixels. |
| `duty_cycle` | Active fraction of each exposure bin for coded sequences. |
| `photon_budget` | Total photon count used for Poisson noise synthesis. |
| `read_noise_sigma` | Standard deviation of additive Gaussian read noise. |
| `save_arrays` / `save_pngs` / `save_figures` | Persist NumPy arrays, PNGs, and figure assets to disk. |
| `output_dir` | Destination directory (defaults to `forward_model_outputs/`). |
| `legendre_params` | Overrides for MLS code construction (polynomial order, randomization, etc.). |

Consult the docstrings in `runner.py` (forward runner) and the helper modules (`patterns.py`,
`optics.py`, `noise.py`, `metrics.py`) for the full list of options and default values.

## Example Usage
```python
from pathlib import Path
from mtf_aware_deblurring import SyntheticData, run_forward_model

scene = SyntheticData("Checker Board", height=256, width=256).create_img(seed=0)
results = run_forward_model(
    scene,
    patterns=["box", "random", "legendre"],
    T=31,
    blur_length_px=15.0,
    photon_budget=1000.0,
    save_figures=True,
    output_dir=Path("forward_model_outputs") / "demo",
)
```
The `results` dictionary contains PSFs, OTFs, MTFs, blurred measurements, noise realizations, and spectral SNR maps keyed by pattern name.

## Planned Extensions
- Plug the exported PSFs into classical deconvolution baselines (Wiener, Richardson-Lucy).
- Integrate the runner with PnP / HQS loops that adapt denoising strength based on the computed MTF.
- Explore additional codes (e.g., m-sequences, learned binary patterns) and longer exposure settings.
- Capture real motion data and validate the forward model against measured PSFs/MTFs.

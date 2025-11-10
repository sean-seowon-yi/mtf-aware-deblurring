# MTF-Aware Deblurring

A research-grade toolkit for coded-exposure motion-blur simulation, physics-aware reconstruction, and baseline benchmarking. The project originated from the CSC2529 course proposal and has since been refactored into a reusable Python package with CLI entry points, dataset loaders, and baseline scripts.

---

## Capabilities at a Glance

- **Forward imaging pipeline** (`pipelines/forward.py`):
  - Box, random, and Legendre (MLS) shutter codes with configurable taps `T`, blur length, and duty cycle.
  - Poisson-Gaussian noise injection under a photon budget.
  - Optional RGB processing and automatic DIV2K downloading (`--image-mode`, `--auto-download`).
  - Matplotlib visualization hooks plus structured array/PNG exports per pattern.
- **Dataset integration**:
  - `DIV2KDataset` streams low-res splits with on-the-fly resizing and color selection.
  - Helper functions ensure the required subset is downloaded before running experiments.
- **Baselines**:
  - Reusable Wiener + Richardson-Lucy algorithms live under `reconstruction/`.
  - `pipelines/reconstruct.py` automates running either method (and now the ADAM+TinyDenoiser baseline) with shared batching + CSV output.
  - Plug-and-play ADAM couples the forward physics with a bundled deep denoiser (see the baseline section below).
  - New ADMM variants support both residual CNN denoisers and score-based diffusion priors (TinyScoreUNet adapter inspired by DPS/ADMM-Score).
- **Documentation**:
  - Proposal summary, forward-model overview, and a growing `docs/baselines/` section with qualitative/quantitative evidence (e.g., `wiener_baseline.md`, `rl_baseline.md`, `adam_denoiser_baseline.md`).

---

## Installation & Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> **PyTorch note**: the ADAM+TinyDenoiser baseline depends on PyTorch (`torch>=2.2`). The CPU wheel installs via `requirements.txt`; install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) if you want GPU acceleration.

All outputs default to `src/mtf_aware_deblurring/forward_model_outputs/`; override via `--output-dir` when needed.

---

## Forward Model Usage

### Quick smoke test (synthetic scenes)
```bash
python -m mtf_aware_deblurring.forward_pipeline
```

### DIV2K batch with RGB processing and auto-downloading
```bash
python -m mtf_aware_deblurring.forward_pipeline ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb ^
  --limit 10 ^
  --auto-download ^
  --save-arrays --save-figures --save-pngs
```
Outputs land beneath `forward_model_outputs/div2k/<image_id>/` (arrays, figures, PNGs per pattern).

### Programmatic usage
```python
from pathlib import Path
from mtf_aware_deblurring import SyntheticData, run_forward_model

scene = SyntheticData("Checker Board").create_img(seed=0)
results = run_forward_model(scene, patterns=["box", "random", "legendre"])
psf = results["patterns"]["legendre"]["psf"]
```

---

## Baseline: Wiener Deconvolution

CLI (grayscale example):
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode grayscale ^
  --limit 10 ^
  --auto-download ^
  --wiener-k 1e-3 ^
  --save-recon
```

Artifacts:
- `forward_model_outputs/reconstruction/wiener/<image_id>/wiener/*.png` (per-pattern reconstructions).
- `forward_model_outputs/reconstruction/wiener/wiener_psnr.csv` (per-image PSNR entries).
- Qualitative crops + tables documented in `docs/baselines/wiener_baseline.md` and `docs/baselines/rl_baseline.md`.

You can adjust noise parameters (`--photon-budget`, `--read-noise`) or `--wiener-k` to explore trade-offs before moving on to ADMM/PnP baselines.
Add `--collect-only` to skip per-image folders and record only the consolidated CSV.
Current RGB DIV2K averages:
- Wiener (`k=1e-3`): Box 13.35 dB, Random 16.77 dB, Legendre 17.14 dB.
- Richardson-Lucy (`iterations=12`, `damping=0.7`, `tv_weight=1e-3`, `smooth_weight=0.4`, `smooth_sigma=1.5`): Box 19.01 dB, Random 19.00 dB, Legendre 19.02 dB.
See the docs in `docs/baselines/` for setup details.

## Baseline: Richardson-Lucy

CLI example (RGB, no recon dumps):
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb ^
  --limit 10 ^
  --auto-download ^
  --method rl ^
  --rl-iterations 12 ^
  --rl-damping 0.7 ^
  --rl-tv-weight 1e-3 ^
  --rl-smooth-weight 0.4 ^
  --rl-smooth-sigma 1.5 ^
  --collect-only
```
Artifacts mirror the Wiener layout but live under `forward_model_outputs/reconstruction/rl/` with an `rl_psnr.csv` summary.

Aggregated RGB results (800 images): Box 19.01 dB, Random 19.00 dB, Legendre 19.02 dB. Detailed setup in `docs/baselines/rl_baseline.md`.

## Baseline: ADAM + TinyDenoiser (Plug-and-Play)

We couple the forward-model data term with a lightweight residual CNN denoiser (`TinyDenoiser`) that was trained on 5,120 DIV2K patches via `scripts/train_tiny_denoiser.py`. The pretrained weights ship with the repo under `src/mtf_aware_deblurring/assets/tiny_denoiser_sigma15.pth`.

CLI example (full DIV2K sweep, RGB):
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb --limit 0 ^
  --method adam ^
  --adam-iters 30 ^
  --adam-lr 0.065 ^
  --adam-denoiser-weight 0.38 ^
  --adam-denoiser-interval 3 ^
  --denoiser-type dncnn ^
  --collect-only
```

Aggregated RGB results (800 images):

| Pattern  | Mean PSNR (dB) | Δ vs RL |
|----------|----------------|--------|
| box      | 22.16          | +3.15  |
| random   | 22.78          | +3.77  |
| legendre | 22.69          | +3.67  |

Swap `--denoiser-type tiny` (default) for lightweight experiments, `--denoiser-type dncnn` for a stronger pretrained prior, or `--denoiser-type unet` if you train the bundled UNet on your own noise model. For a quick grayscale smoke test (DIV2K, 5 frames, 128×128 crops, Poisson-Gaussian noise) we observed:

| Denoiser | Box | Random | Legendre | Notes |
|----------|-----|--------|----------|-------|
| tiny     | 19.99 | 19.95 | 20.37 | bundled residual CNN, CPU-friendly |
| dncnn    | **20.43** | **23.12** | **23.78** | converted from the public σ=15 checkpoint |
| unet     | 20.37 | 20.93 | 21.33 | trained via `scripts/train_unet_denoiser.py` (3 epochs, 50 DIV2K images) |

See `docs/baselines/adam_denoiser_baseline.md` for full RGB statistics, training details, and hyperparameters.


## Baseline: ADMM + TinyDenoiser (Plug-and-Play)

The ADMM variant solves the blur data term in closed form (FFT-domain normal equations) and reuses the TinyDenoiser as a plug-and-play proximal operator each iteration.

CLI example (RGB batch):

```bash
python -m mtf_aware_deblurring.pipelines.reconstruct ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb --limit 0 ^
  --method admm ^
  --admm-iters 60 ^
  --admm-rho 0.4 ^
  --admm-denoiser-weight 1.0 ^
  --collect-only
```

`--admm-rho` tunes the augmented-Lagrangian penalty; larger values make the solution lean more on the denoiser. `--admm-denoiser-weight` blends between the raw ADMM iterate and denoiser prediction (1.0 = full denoiser). Use `--denoiser-type dncnn` or `--denoiser-type unet` to switch to the heavier priors when GPUs are available. A matching 5-image grayscale sweep produced:

| Denoiser | Box | Random | Legendre | Notes |
|----------|-----|--------|----------|-------|
| tiny     | **24.09** | **25.24** | **25.78** | default Poisson-Gaussian tuned weights |
| dncnn    | 22.36 | 23.29 | 23.72 | assumes additive Gaussian noise; still improves over tiny ADAM |
| unet     | 24.10 | 25.17 | 25.74 | fine-tuned UNet prior |

Results will be added to `docs/baselines/` alongside the other methods as we scale these experiments.

## Baseline: ADMM + Diffusion Prior (PnP-Diffusion)

Building on Diffusion Posterior Sampling (Chung et al., CVPR 2023) and ADMM-Score (Wang et al., NeurIPS 2023), we expose an ADMM variant whose proximal step is driven by a score-based diffusion prior. The lightweight `TinyScoreUNet` backbone lives in `mtf_aware_deblurring.diffusion` and can load any noise-prediction checkpoint you train (or adapt from existing DDPM/DDIM work).

CLI example (RGB, DPS-style schedule):

```bash
python -m mtf_aware_deblurring.pipelines.reconstruct ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb --limit 0 ^
  --method admm_diffusion ^
  --admm-iters 40 ^
  --admm-rho 0.6 ^
  --diffusion-steps 16 ^
  --diffusion-guidance 1.2 ^
  --diffusion-noise-scale 0.8 ^
  --diffusion-prior-weights path/to/tiny_score_checkpoint.pth ^
  --collect-only
```

### Training / Fine-Tuning Denoisers

- `scripts/train_tiny_denoiser.py`: reproduces the lightweight residual CNN used in the ADAM baseline (fast on CPU).
- `scripts/train_unet_denoiser.py`: fine-tunes the new UNet denoiser on Poisson-Gaussian DIV2K patches; pass `--device cuda` to leverage AMD GPUs via ROCm/DirectML builds.

All denoiser scripts default to saving weights in `src/mtf_aware_deblurring/assets/`, and plug-and-play baselines pick them up automatically via `--denoiser-type` / `--denoiser-weights`.

Key flags:
- `--diffusion-prior-type tiny_score` (default) selects the bundled TinyScoreUNet architecture.
- `--diffusion-prior-weights` should point to a `.pth` file containing the score-model weights (matching your training procedure).
- `--diffusion-steps`, `--diffusion-guidance`, `--diffusion-noise-scale`, and the sigma limits control the DPS-style schedule, letting you emulate the step counts used in recent papers.

This baseline shares the FFT-domain `x`-updates with the denoiser-driven ADMM but swaps in the diffusion-based proximal operator, making it easy to plug in stronger priors as they become available.


---

## Repository Layout (Highlights)

- `src/mtf_aware_deblurring/forward_pipeline.py`  compatibility shim exposed via `python -m`.
- `src/mtf_aware_deblurring/pipelines/` - CLI entry points (`forward.py`, `reconstruct.py`) plus shared batch helpers.
- `src/mtf_aware_deblurring/reconstruction/` - reusable reconstruction algorithms (Wiener, Richardson-Lucy, ADAM+TinyDenoiser).
- `src/mtf_aware_deblurring/denoisers/` & `src/mtf_aware_deblurring/assets/` - TinyDenoiser architecture plus the pretrained weights used by the ADAM baseline.
- `src/mtf_aware_deblurring/diffusion/` - TinyScoreUNet definition, sigma scheduling, and the ADMM diffusion prior adapter.
- `src/mtf_aware_deblurring/forward_model_outputs/` - default artifact directories (`div2k/<id>/`, `reconstruction/<method>/`).
- `src/mtf_aware_deblurring/{datasets,patterns,optics,noise,metrics,synthetic,utils}.py`  reusable building blocks.
- `scripts/train_tiny_denoiser.py` - helper to regenerate the residual denoiser if you change the noise model.
- `docs/`  proposal, summaries, and baseline reports (`docs/baselines/wiener_baseline.md` with qualitative crops and tables).

---

## Current Status & Next Steps

-  Refactored forward model into a reusable module with CLI.
-  DIV2K integration with auto-download and RGB support.
-  Baseline coverage: Wiener, Richardson-Lucy, ADAM+TinyDenoiser, ADMM+TinyDenoiser, and the new ADMM+Diffusion prior (score-based) hooks.
-  Upcoming work:
  - Physics-aware PnP scheduling experiments (MTF-weighted denoiser schedules).
  - Extended ablations: photon budget sweeps, exposure code families, schedule variants.
  - Additional metrics (SSIM, LPIPS) and experiment logging in `docs/experiments/`.

For historical context, consult the [proposal summary](docs/proposal_summary.md) or the original [project proposal PDF](docs/project_proposal.pdf). Baseline details and figures live in [docs/baselines/wiener_baseline.md](docs/baselines/wiener_baseline.md).

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
  - `pipelines/reconstruct.py` orchestrates ADAM, ADMM (denoiser and diffusion variants), and shared batching/CSV/reporting.
  - Plug-and-play solvers support multiple priors: the bundled TinyDenoiser, a converted DnCNN sigma=15 model, a UNet denoiser trained via scripts/train_unet_denoiser.py, pretrained DRUNet color/gray backbones sourced from DPIR, and a TinyScoreUNet diffusion prior (see the baseline section below).
- **Device-aware training/runtime**:
  - `torch_utils.resolve_device` lets every CLI flag accept `cpu`, `cuda`, or `dml` (DirectML) so you can target NVIDIA GPUs, AMD GPUs, or CPU-only runs without code changes.
  - Training scripts (`scripts/train_*`) detect the same device flag, making it straightforward to fine-tune priors on NVIDIA (CUDA) or AMD (DirectML/ROCm) hardware.
- **Documentation**:
  - Proposal summary, forward-model overview, and a growing `docs/baselines/` section with qualitative/quantitative evidence (e.g., `wiener_baseline.md`, `rl_baseline.md`, `adam_denoiser_baseline.md`).

---

## Installation & Environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> **PyTorch note**: the ADAM+TinyDenoiser baseline depends on PyTorch (`torch>=2.2`). The CPU wheel installs via `requirements.txt`; install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) if you want GPU acceleration.

All outputs default to `src/mtf_aware_deblurring/forward_model_outputs/`; override via `--output-dir` when needed.

## Device & GPU Support

Every plug-and-play baseline accepts `--denoiser-device` (and, for diffusion, `--diffusion-device`) with one of:

- `cpu` — portable runs with no GPU.
- `cuda` — NVIDIA GPUs using the standard CUDA PyTorch wheels.
- `dml` — Windows/DirectML backend for AMD GPUs (`pip install torch-directml -f https://aka.ms/torch-directml`), or ROCm wheels on Linux.

Tips:
- Keep GPU-specific installs in their own conda/venv environments (e.g., `conda create -n amd python=3.10` then install `torch-directml` plus project deps).
- Verify the backend with `python -c "import torch, torch_directml; print(torch.__version__); print(torch_directml.device())"`.
- Training scripts accept the same device flag, so the checkpoints you generate for NVIDIA (CUDA) or AMD (DirectML/ROCm) can be re-used directly by the CLI.

---

## Forward Model Usage

### Quick smoke test (synthetic scenes)
```bash
python -m mtf_aware_deblurring.forward_pipeline
```

### DIV2K batch with RGB processing and auto-downloading

#### Windows
```bash
python -m mtf_aware_deblurring.forward_pipeline ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb ^
  --limit 10 ^
  --auto-download ^
  --save-arrays --save-figures --save-pngs
```

#### Linux / macOS
```bash
python -m mtf_aware_deblurring.forward_pipeline \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode rgb \
  --limit 10 \
  --auto-download \
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

### Windows
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

### Linux / macOS
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode grayscale \
  --limit 10 \
  --auto-download \
  --wiener-k 1e-3 \
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

### Windows
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

### Linux / macOS
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode rgb \
  --limit 10 \
  --auto-download \
  --method rl \
  --rl-iterations 12 \
  --rl-damping 0.7 \
  --rl-tv-weight 1e-3 \
  --rl-smooth-weight 0.4 \
  --rl-smooth-sigma 1.5 \
  --collect-only
```

Artifacts mirror the Wiener layout but live under `forward_model_outputs/reconstruction/rl/` with an `rl_psnr.csv` summary.

Aggregated RGB results (800 images): Box 19.01 dB, Random 19.00 dB, Legendre 19.02 dB. Detailed setup in `docs/baselines/rl_baseline.md`.

## Baseline: ADAM + Plug-and-Play Denoisers

The ADAM solver treats the coded-exposure forward model as the data term and injects a denoiser every few iterations (PnP-ADAM). You can swap priors via `--denoiser-type` and control how aggressively they are blended with `--adam-denoiser-weight` / `--adam-denoiser-interval`.

### Windows
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
  --denoiser-device cuda ^
  --collect-only
```

### Linux / macOS
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 \
  --method adam \
  --adam-iters 30 \
  --adam-lr 0.065 \
  --adam-denoiser-weight 0.38 \
  --adam-denoiser-interval 3 \
  --denoiser-type dncnn \
  --denoiser-device cpu \
  --collect-only
```

### Quick 5-image grayscale smoke test (DIV2K/X2, 128×128 crops)
| Denoiser | Box | Random | Legendre | Notes |
|----------|-----|--------|----------|-------|
| tiny (`--denoiser-type tiny`) | 19.99 | 19.95 | 20.37 | Bundled residual CNN (σ≈15) — fast CPU baseline |
| dncnn (`--denoiser-type dncnn`) | **20.43** | **23.12** | **23.78** | Converted from the public σ=15 MATLAB checkpoint; strongest of the three in ADAM |
| unet (`--denoiser-type unet`) | 20.37 | 20.93 | 21.33 | Fine-tuned via `scripts/train_unet_denoiser.py` (3 epochs, 50 DIV2K frames) |

The full RGB sweep (800 images) from `docs/baselines/adam_denoiser_baseline.md` still reports ~22–23 dB with the TinyDenoiser, but the table above illustrates how much room there is when swapping priors.

Available priors:
- **tiny** - the original 8-layer residual CNN (`scripts/train_tiny_denoiser.py`). Ships with the repo for CPU-friendly experiments.
- **dncnn** - automatically downloads/converts the sigma=15 model from the DnCNN project.
- **unet** - shallow UNet tailored for our Poisson-Gaussian forward model; run `scripts/train_unet_denoiser.py --device cuda` (NVIDIA) or pass `--device dml` for AMD/DirectML to regenerate weights.
- **drunet_color / drunet_gray** - DPIR pretrained DRUNet checkpoints auto-downloaded from the `deepinv/drunet` Hugging Face repo. Use `--denoiser-type drunet_color` for RGB or `--denoiser-type drunet_gray` for grayscale; weights are cached under `~/.cache/mtf_aware_deblurring/drunet/`.

All denoiser choices share the same CLI; just pass `--denoiser-type` and optionally `--denoiser-weights` / `--denoiser-device` to override the defaults.


## Baseline: ADMM + Plug-and-Play Denoisers

ADMM solves the data term exactly in the frequency domain, then applies a proximal prior (denoiser) before updating the dual variable. Compared to ADAM, ADMM typically delivers higher PSNR and is the preferred path once you have a strong prior.

### Windows
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct ^
  --div2k-root data ^
  --subset train --degradation bicubic --scale X2 ^
  --image-mode rgb --limit 0 ^
  --method admm ^
  --admm-iters 60 ^
  --admm-rho 0.4 ^
  --admm-denoiser-weight 1.0 ^
  --denoiser-type drunet_color ^
  --denoiser-device cuda ^
  --collect-only
```

### Linux / macOS
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 \
  --method admm \
  --admm-iters 60 \
  --admm-rho 0.4 \
  --admm-denoiser-weight 1.0 \
  --denoiser-type drunet_color \
  --denoiser-device cpu \
  --collect-only
```
> **DRUNet note:** --denoiser-type drunet_color targets RGB scenes while --denoiser-type drunet_gray handles grayscale inputs. The pretrained weights download once to ~/.cache/mtf_aware_deblurring/drunet/ and are reused on subsequent runs.



### Quick 5-image grayscale smoke test
| Denoiser | Box | Random | Legendre | Notes |
|----------|-----|--------|----------|-------|
| tiny | **24.09** | **25.24** | **25.78** | Surprisingly strong despite its small size. Ships with repo. |
| dncnn | 22.36 | 23.29 | 23.72 | Great for ADAM, but underperforms in ADMM because the Gaussian assumptions clash with the augmented Lagrangian step. |
| unet | 24.10 | 25.17 | 25.74 | Matches the Tiny prior today; expect improvements once the UNet is trained longer on Poisson-Gaussian noise. |

The same denoiser flags (`--denoiser-type/--denoiser-weights/--denoiser-device`) apply here, so you can reuse the checkpoints produced by the training scripts listed below.

## Baseline: ADMM + Diffusion Prior (PnP-Diffusion)

Building on Diffusion Posterior Sampling (Chung et al., CVPR 2023) and ADMM-Score (Wang et al., NeurIPS 2023), we expose an ADMM variant whose proximal step is driven by a score-based diffusion prior. The lightweight `TinyScoreUNet` backbone lives in `mtf_aware_deblurring.diffusion` and can load any noise-prediction checkpoint you train (or adapt from existing DDPM/DDIM work).

### Windows
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

### Linux / macOS
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data \
  --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 \
  --method admm_diffusion \
  --admm-iters 40 \
  --admm-rho 0.6 \
  --diffusion-steps 16 \
  --diffusion-guidance 1.2 \
  --diffusion-noise-scale 0.8 \
  --diffusion-prior-weights path/to/tiny_score_checkpoint.pth \
  --collect-only
```

Training the score model:
- Use `scripts/train_tiny_score_model.py --device cuda --max-images 200 --patches-per-image 64 --patch-size 96 --epochs 20` on an NVIDIA GPU (or `--device dml` for DirectML) to populate `src/mtf_aware_deblurring/assets/tiny_score_unet.pth`.
- The script performs denoising score matching on grayscale DIV2K patches; feel free to swap in your dataset or noise schedule (`--sigma-min`, `--sigma-max`).
- Pass the resulting checkpoint via `--diffusion-prior-weights` and optionally override `--diffusion-steps`, `--diffusion-guidance`, and `--diffusion-sigma-*` per experiment.

### Quick 5-image grayscale smoke test (TinyScoreUNet, 8 steps)
| Pattern | PSNR (dB) |
|---------|-----------|
| box     | 15.95 |
| random  | 16.39 |
| legendre| 16.46 |

These numbers are lower than the denoiser-driven ADMM runs because the TinyScoreUNet above was trained for only a few epochs. Nevertheless, the infrastructure is in place: once you plug in a stronger score model (pretrained ADM, DiT, etc.), ADMM+diffusion becomes a drop-in option alongside the other baselines.

### Training / Fine-Tuning Denoisers & Score Models

- `scripts/train_tiny_denoiser.py`: reproduces the lightweight residual CNN (σ≈15) used for the historical ADAM baseline. Works great on CPU.
- `scripts/train_unet_denoiser.py`: trains the UNet prior on Poisson-Gaussian patches. Pass `--device cuda` on NVIDIA or `--device dml` on Windows/AMD (DirectML). Outputs `assets/unet_denoiser_sigma15.pth`.
- `scripts/train_tiny_score_model.py`: denoising score matching for TinyScoreUNet. Use the same `--device` flag; the produced `tiny_score_unet.pth` feeds `--method admm_diffusion`.

All scripts default to saving weights in `src/mtf_aware_deblurring/assets/`, and every plug-and-play baseline accepts `--denoiser-type/--denoiser-weights` (or `--diffusion-prior-weights`) so you can hot-swap your checkpoints without touching code.

Key ADMM-diffusion flags:
- `--diffusion-prior-type tiny_score` (default) selects the TinyScoreUNet adapter.
- `--diffusion-prior-weights` points to the `.pth` generated above.
- `--diffusion-steps`, `--diffusion-guidance`, `--diffusion-noise-scale`, and the sigma bounds control the DPS-style schedule, letting you emulate ADMM-Score/DPS hyper-parameters.

The diffusion pipeline shares the same FFT-domain `x`-updates as the denoiser-driven ADMM variant—only the proximal operator changes—so stronger score models can be plugged in as soon as you train or import them.


---

## Repository Layout (Highlights)

- `src/mtf_aware_deblurring/forward_pipeline.py` — compatibility shim exposed via `python -m`.
- `src/mtf_aware_deblurring/pipelines/` - CLI entry points (`forward.py`, `reconstruct.py`) plus shared batch helpers.
- `src/mtf_aware_deblurring/reconstruction/` - reusable reconstruction algorithms (Wiener, Richardson-Lucy, ADAM+TinyDenoiser).
- `src/mtf_aware_deblurring/denoisers/` & `src/mtf_aware_deblurring/assets/` - TinyDenoiser architecture plus the pretrained weights used by the ADAM baseline.
- `src/mtf_aware_deblurring/diffusion/` - TinyScoreUNet definition, sigma scheduling, and the ADMM diffusion prior adapter.
- `src/mtf_aware_deblurring/forward_model_outputs/` - default artifact directories (`div2k/<id>/`, `reconstruction/<method>/`).
- `src/mtf_aware_deblurring/{datasets,patterns,optics,noise,metrics,synthetic,utils}.py` — reusable building blocks.
- `scripts/train_tiny_denoiser.py` - helper to regenerate the residual denoiser if you change the noise model.
- `docs/` — proposal, summaries, and baseline reports (`docs/baselines/wiener_baseline.md`, `docs/baselines/adam_denoiser_baseline.md`, `docs/baselines/admm_pnp_baseline.md`).

---

## Current Status & Next Steps

- ✓ Refactored forward model into a reusable module with CLI.
- ✓ DIV2K integration with auto-download and RGB support.
- ✓ Baseline coverage: Wiener, Richardson-Lucy, ADAM+TinyDenoiser, ADMM+TinyDenoiser, and the new ADMM+Diffusion prior (score-based) hooks.
- ⚙ Upcoming work:
  - Physics-aware PnP scheduling experiments (MTF-weighted denoiser schedules).
  - Extended ablations: photon budget sweeps, exposure code families, schedule variants.
  - Additional metrics (SSIM, LPIPS) and experiment logging in `docs/experiments/`.

For historical context, consult the [proposal summary](docs/proposal_summary.md) or the original [project proposal PDF](docs/project_proposal.pdf). Baseline details and figures live in [docs/baselines/wiener_baseline.md](docs/baselines/wiener_baseline.md).

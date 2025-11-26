# ADMM Plug-and-Play Baselines

## Experimental Setup

- **Dataset**: `DIV2K_train_LR_bicubic/X2`, grayscale 128×128 crops (first five training frames for the smoke test below). The full RGB sweep uses the same settings as the ADAM baseline (blur length 15 px, `T=31`, duty cycle 0.5, photon budget 1000, read-noise 0.01).
- **Forward model**: identical to Wiener/RL/ADAM.
- **Solver**: augmented Lagrangian ADMM with FFT-domain `x`-updates (`--admm-iters 40`, `--admm-rho 0.4`, `--admm-denoiser-weight 1.0`).
- **Command template**:
  ```bash
  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root data \
    --subset train --degradation bicubic --scale X2 \
    --image-mode grayscale --limit 5 \
    --patterns box random legendre \
  --method admm \
  --admm-iters 40 \
  --admm-rho 0.4 \
  --admm-denoiser-weight 1.0 \
  --admm-mtf-weighting-mode none \
  --denoiser-type <tiny|dncnn|unet> \
  --denoiser-device <cpu|cuda|dml> \
  --collect-only
  ```
- Outputs land in `forward_model_outputs/reconstruction/admm/`.

## Denoiser Comparison (5-image Smoke Test)

| Denoiser | Box | Random | Legendre | Notes |
|----------|-----|--------|----------|-------|
| Tiny (`--denoiser-type tiny`) | **24.09** | **25.24** | **25.78** | Bundled σ≈15 prior; still a strong performer under ADMM. |
| DnCNN (`--denoiser-type dncnn`) | 22.36 | 23.29 | 23.72 | Converted from the public σ=15 MATLAB weights; shines in ADAM but less so once the augmented Lagrangian kicks in. |
| UNet (`--denoiser-type unet`) | 24.10 | 25.17 | 25.74 | Fine-tuned via `scripts/train_unet_denoiser.py`. Expect gains once the UNet sees more epochs or RGB training. |

All three priors share the same CLI; only `--denoiser-type` (and optionally `--denoiser-weights`) changes.
If you want to ablate physics hooks: enable the scheduler with `--use-physics-scheduler` and choose an MTF weighting mode (`none` default; `gamma`, `wiener`, or `combined`), plus `--admm-mtf-sigma-adapt` for DRUNet sigma adaptation.

## ADMM + Diffusion Prior

We also expose `--method admm_diffusion`, which replaces the explicit denoiser proximal with a learned score model (`TinyScoreUNet`). The proximal update mirrors Diffusion Posterior Sampling / ADMM-Score: draw a noise level `σ`, query the score model, then nudge the iterate toward the posterior of the data term.

- **CLI**:
  ```bash
  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root data \
    --subset train --degradation bicubic --scale X2 \
    --image-mode grayscale --limit 5 \
    --method admm_diffusion \
    --admm-iters 30 \
    --admm-rho 0.5 \
    --diffusion-steps 8 \
    --diffusion-guidance 1.0 \
    --diffusion-noise-scale 0.5 \
    --diffusion-prior-weights src/mtf_aware_deblurring/assets/tiny_score_unet.pth \
    --diffusion-device cuda \
    --collect-only
  ```
- **Training**: run `scripts/train_tiny_score_model.py --device cuda --max-images 200 --patches-per-image 64 --patch-size 96 --epochs 20` to produce `tiny_score_unet.pth`. DirectML (`--device dml`) also works, but note that `GroupNorm` currently falls back to CPU on DirectML, so CUDA training is preferred.

### Current Smoke-Test Results (5 grayscale frames)
| Pattern | PSNR (dB) |
|---------|-----------|
| box     | 15.95 |
| random  | 16.39 |
| legendre| 16.46 |

These numbers reflect a lightly trained TinyScoreUNet (3 epochs). Expect major gains once a stronger score model (ADM/DiT or a longer schedule) is dropped in—the infrastructure already matches the formulations from ADMM-Score and DPS.

## Recommendations
- Start with `--denoiser-type tiny` or `unet` for production-quality ADMM results; bring in DnCNN when sweeping ADAM hyper-parameters.
- Train the UNet and TinyScoreUNet on the same noise statistics (photon budget, read noise) you plan to simulate.
- For AMD hardware, install `torch-directml` inside a dedicated conda environment (see README) and use `--denoiser-device dml` / `--diffusion-device dml`.

## Physics-Aware Sweep (RGB DIV2K/X2, 10 Images)

To tune the physics-aware knobs we swept `rho` and `denoiser_weight` with the scheduler, combined MTF weighting, and sigma adaptation on RGB DIV2K/X2 (first 10 train images, patterns box/random/legendre). Command template:
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root $HOME/datasets/DIV2K --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 10 --auto-download \
  --patterns box random legendre \
  --method admm \
  --admm-iters 60 \
  --admm-rho <rho> \
  --admm-denoiser-weight <weight> \
  --admm-mtf-weighting-mode combined \
  --admm-mtf-sigma-adapt \
  --use-physics-scheduler \
  --denoiser-type drunet_color \
  --denoiser-device cuda \
  --collect-only --enable-ssim
```

Top settings (PSNR averages, 10 images):
- **rho=1.04, weight=0.16** — box 25.34 dB, random 28.43 dB, legendre **29.38 dB** (best observed).
- rho=1.02, weight=0.16 — box 25.36 dB, random 28.42 dB, legendre 29.38 dB (tied within noise).
- rho=1.00, weight=0.16 — box 25.38 dB, random 28.41 dB, legendre 29.37 dB.
- rho=1.02, weight=0.15 — box 25.42 dB, random 28.06 dB, legendre 29.35 dB.

Recommendation: use rho≈1.04 and denoiser weight≈0.16 for physics-aware ADMM with DRUNet; nearby values (1.00–1.02, 0.15–0.16) are essentially tied. Keep `--admm-mtf-weighting-mode combined` and `--admm-mtf-sigma-adapt` enabled when using the scheduler.

# ADAM + TinyDenoiser Baseline Report

## Experimental Setup
- **Dataset**: `DIV2K_train_LR_bicubic/X2`, RGB 256×256 crops (all 800 training frames).
- **Forward model**: identical to the Wiener/RL baselines (box/random/legendre codes, `T=31`, blur length 15 px, duty cycle 0.5, photon budget 1000, read-noise 0.01, seed 0).
- **Denoiser**: an 8-layer residual CNN (`TinyDenoiser`) trained on 5,120 DIV2K patches with synthetic σ=15/255 Gaussian noise via `scripts/train_tiny_denoiser.py`. Weights ship with the repo under `src/mtf_aware_deblurring/assets/tiny_denoiser_sigma15.pth`.
- **Optimizer**: Plug-and-play ADAM with denoiser relaxations every 3 steps.
- **Command**:
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
    --collect-only
  ```
  (Add `--denoiser-device cuda` if a GPU is available.)
- **Runtime**: ~96 minutes on a 12‑core CPU (no GPU) for the full 800-image sweep.
- Outputs live under `forward_model_outputs/reconstruction/adam/adam_psnr.csv`.

## Quantitative Summary
| Pattern  | Mean PSNR (dB) | Δ vs. RL (dB) |
|----------|----------------|---------------|
| box      | 22.16          | +3.15         |
| random   | 22.78          | +3.77         |
| legendre | 22.69          | +3.67         |

The ADAM+denoiser baseline consistently outperforms the damped Richardson–Lucy baseline by 3–4 dB across all coded shutters. Legendre and random codes benefit the most because the denoiser suppresses the amplified Poisson noise that RL struggles with at high spatial frequencies.

## Notes & Recommendations
- The denoiser weight (0.38) and interval (3) strike a balance between detail recovery and runtime. Increasing the interval reduces compute but drops PSNR by >1 dB.
- The bundled TinyDenoiser targets σ≈15/255. Regenerate weights with `scripts/train_tiny_denoiser.py` if you change the photon budget/noise model significantly.
- Consider enabling `--save-recon` for qualitative crops—plug-and-play reconstructions have visibly fewer ringing artifacts than RL, especially for the legendre code.

## Alternative Priors (`--denoiser-type`)

With the recent refactor you can swap priors without changing code:

| Name | Flag | Training Script | Comments |
|------|------|-----------------|----------|
| TinyDenoiser | `--denoiser-type tiny` (default) | `scripts/train_tiny_denoiser.py` | CPU-friendly, ships with repo. |
| DnCNN σ=15 | `--denoiser-type dncnn` | *(auto-downloaded)* | We convert the public MATLAB weights to PyTorch on first use. |
| UNet PnP | `--denoiser-type unet` | `scripts/train_unet_denoiser.py` | Tailored for Poisson-Gaussian noise; accepts `--device cuda` or `--device dml`. |

On a small 5-image grayscale smoke test (DIV2K/X2, 128×128 crops) we observed:

| Denoiser | Box | Random | Legendre |
|----------|-----|--------|----------|
| Tiny | 19.99 dB | 19.95 dB | 20.37 dB |
| DnCNN | **20.43 dB** | **23.12 dB** | **23.78 dB** |
| UNet | 20.37 dB | 20.93 dB | 21.33 dB |

The full 800-image numbers in this report still refer to the Tiny prior, but DnCNN and UNet offer a clear upgrade path once you have GPU acceleration. Plug them in via `--denoiser-type` / `--denoiser-weights` and keep the rest of the CLI identical.

# Wiener Baseline Report

## Experimental Setup
- **Dataset**: DIV2K_train_LR_bicubic/X2 (low-resolution split), resized to 256x256 with grayscale conversion.
- **Forward model**: Patterns = [`box`, `random`, `legendre`], taps `T = 31`, blur length `15 px`, duty cycle `0.5`, photon budget `1000`, read-noise `0.01`, random seed `0`.
- **Wiener parameters**: global constant `k = 1e-3` applied identically to every exposure pattern.
- **Command**:
  ```bash
  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root data \
    --subset train --degradation bicubic --scale X2 \
    --image-mode grayscale --limit 1 \
    --auto-download --wiener-k 1e-3 --save-recon
  ```
- Outputs land in `src/mtf_aware_deblurring/forward_model_outputs/reconstruction/wiener/<image_id>/`.

## Qualitative Crops
![Wiener crops](wiener_crops.png)
*Crops (96x96 px) centered on `0001x2.png`: clean DIV2K frame and Wiener reconstructions for box, random, and Legendre codes.*

## Quantitative Summary
Per-image PSNR (dB):

| Image | Pattern | PSNR (dB) |
|-------|---------|-----------|
| 0001x2.png | box | 14.54 |
| 0001x2.png | random | 18.08 |
| 0001x2.png | legendre | 18.88 |

Pattern averages:

| Pattern | Mean PSNR (dB) | Samples |
|---------|----------------|---------|
| box | 14.54 | 1 |
| random | 18.08 | 1 |
| legendre | 18.88 | 1 |

These values establish the classical Wiener baseline for subsequent ADMM/PnP comparisons; additional images can be processed by increasing `--limit` and rerunning the command.

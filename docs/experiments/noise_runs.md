# DIV2K X2 Baselines: Noise1 vs Noise2

This note aggregates the full-batch DIV2K RGB results we produced during the recent sessions, along with the exact noise settings and CLI commands we used. All runs are collect-only (no recon saves) with PSNR/SSIM/LPIPS enabled.

## Noise settings

- **Noise1 (baseline):** blur length 15 px, photon budget 1000, read noise 0.01, denoiser sigma-scale 8.0 (default).
- **Noise2 (heavier):** blur length 30 px, photon budget 1000, read noise 0.05, denoiser sigma-scale 12.0.
- Forward model: DIV2K train, bicubic X2, RGB, patterns = box/random/legendre, taps 31, duty cycle 0.5.

## Command templates

Replace the noise block with the desired setting. All runs use CUDA for denoiser and LPIPS unless noted.

**RL tuned (tweaked damping/smoothing)**
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root $HOME/datasets/DIV2K --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 --auto-download \
  --patterns box random legendre \
  --method rl \
  --rl-iterations 12 --rl-damping 0.7 --rl-tv-weight 0.001 \
  --rl-smooth-weight 0.4 --rl-smooth-sigma 1.5 \
  --photon-budget <1000> --blur-length <15|30> --read-noise <0.01|0.05> \
  --collect-only --enable-ssim --enable-lpips --lpips-device cuda \
  --output-dir $HOME/mtf-smoke/<noise-tag>-rl-full
```

**ADAM + DnCNN (noise2 weight bumped to 0.30)**
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root $HOME/datasets/DIV2K --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 --auto-download \
  --patterns box random legendre \
  --method adam \
  --adam-iters 80 --adam-lr 0.04 --adam-denoiser-weight <0.25|0.30> \
  --adam-denoiser-interval 2 \
  --denoiser-type dncnn --denoiser-device cuda \
  --photon-budget <1000> --blur-length <15|30> --read-noise <0.01|0.05> \
  --collect-only --enable-ssim --enable-lpips --lpips-device cuda \
  --output-dir $HOME/mtf-smoke/<noise-tag>-adam-dncnn-full
```

**ADMM vanilla + DRUNet**
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root $HOME/datasets/DIV2K --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 --auto-download \
  --patterns box random legendre \
  --method admm \
  --admm-iters <60|80> --admm-rho <0.6|0.6> --admm-denoiser-weight <0.6|0.6> \
  --admm-mtf-weighting-mode none \
  --denoiser-type drunet_color --denoiser-device cuda \
  --denoiser-sigma-scale <8.0|12.0> \
  --photon-budget <1000> --blur-length <15|30> --read-noise <0.01|0.05> \
  --collect-only --enable-ssim --enable-lpips --lpips-device cuda \
  --output-dir $HOME/mtf-smoke/<noise-tag>-admm-vanilla-drunet-full
```
For the tuned single-image probe on noise2 we also tried ρ≈1.0–1.2, weight≈1.0–1.1, sigma-scale 12 (GPU unavailable; CPU runs only).

**ADMM physics-aware + DRUNet**
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root $HOME/datasets/DIV2K --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 0 --auto-download \
  --patterns box random legendre \
  --method admm \
  --admm-iters <60|80> --admm-rho <1.04|2.0> --admm-denoiser-weight <0.16|0.32> \
  --admm-mtf-weighting-mode combined --admm-mtf-sigma-adapt --use-physics-scheduler \
  --admm-mtf-floor <0.0|0.2> \
  --denoiser-type drunet_color --denoiser-device cuda \
  --denoiser-sigma-scale <8.0|12.0> \
  --photon-budget <1000> --blur-length <15|30> --read-noise <0.01|0.05> \
  --collect-only --enable-ssim --enable-lpips --lpips-device cuda \
  --output-dir $HOME/mtf-smoke/<noise-tag>-admm-phys-drunet-full
```

## Results (averages over 800 images unless noted)

**Noise1 (blur 15, read 0.01, sigma-scale 8)**

| Method | Params | Box (P/S/L) | Random (P/S/L) | Legendre (P/S/L) | Notes |
|--------|--------|-------------|----------------|------------------|-------|
| RL tuned | iters 12, damping 0.7, TV 0.001, smooth 0.4/σ1.5 | 19.0073 / 0.3879 / 0.6364 | 19.0044 / 0.3873 / 0.6185* | 19.0204 / 0.3877 / 0.6227 | *799 valid rows (one NaN PSNR). |
| Adam + DnCNN | lr 0.04, w 0.25, int 2 | 18.7047 / 0.4010 / 0.4965 | 22.3636 / 0.5401 / 0.3723 | 22.4949 / 0.5443 / 0.3668 | |
| Adam + DRUNet | lr 0.04, w 0.25, int 2 | 22.7221 / 0.5386 / 0.3868 | 23.2489 / 0.5735 / 0.3380 | 23.1631 / 0.5713 / 0.3382 | |
| ADMM vanilla + DnCNN | iters 60, ρ 0.6, w 0.6, σ-scale 8 | 19.7153 / 0.4528 / 0.4740 | 25.5632 / 0.7521 / 0.2803 | 25.8204 / 0.7642 / 0.2714 | |
| ADMM vanilla + DRUNet | iters 60, ρ 0.6, w 0.6, σ-scale 8 | 22.9932 / 0.6796 / 0.3731 | 23.9363 / 0.6876 / 0.3144 | 23.6947 / 0.6785 / 0.3189 | |
| ADMM physics-aware + DnCNN | iters 60, ρ 1.04, w 0.16 | 23.0927 / 0.5850 / 0.4274 | 25.8654 / 0.7468 / 0.2729 | 26.3215 / 0.7711 / 0.2605 | |
| ADMM physics-aware + DRUNet | iters 60, ρ 1.04, w 0.16 | 24.3233 / 0.7183 / 0.3856 | 28.1220 / 0.8041 / 0.2172 | 29.2713 / 0.8530 / 0.2114 | |

**Noise2 (blur 30, read 0.05, sigma-scale 12)**

| Method | Params | Box (P/S/L) | Random (P/S/L) | Legendre (P/S/L) | Notes |
|--------|--------|-------------|----------------|------------------|-------|
| RL tuned | same RL params as noise1 | 18.3464 / 0.3191 / 0.6863 | 18.6317 / 0.3242 / 0.6613 | 18.6339 / 0.3253 / 0.6650 | |
| Adam + DnCNN | lr 0.04, w 0.30, int 2 | 12.4512 / 0.1432 / 0.7004 | 17.0510 / 0.2812 / 0.5736 | 17.0791 / 0.2824 / 0.5718 | |
| ADMM physics-aware + DRUNet | iters 80, ρ 2.0, w 0.32, MTF combined/sigma-adapt/scheduler, MTF floor 0.2 | 20.7881 / 0.5406 / 0.5387 | 23.5913 / 0.6749 / 0.4216 | 24.0392 / 0.6925 / 0.4090 | |
| ADMM vanilla + DRUNet | iters 80, ρ 0.6, w 0.6 | 10.4316 / 0.1389 / 0.7245 | 13.4760 / 0.2103 / 0.6594 | 13.5670 / 0.2128 / 0.6556 | |
| ADMM vanilla + DRUNet (tuned) | iters 80, ρ 1.0, w 1.1 | 21.6900 / 0.5640 / 0.5182 | 14.4938 / 0.2550 / 0.6302 | 14.4321 / 0.2511 / 0.6311 | Single-image CPU probes suggest ρ≈1.2, w≈1.0–1.1 are best among vanilla variants tried. |

Missing/unfinished: no noise2 runs for Adam + DRUNet or ADMM physics-aware + DnCNN.

## Where to find the outputs

Metrics are synced to `src/mtf_aware_deblurring/forward_model_outputs/reconstruction/`:

- `noise1/` — all baseline noise runs (adam/admm variants, RL tuned).
- `noise2/` — heavier noise runs listed above, including both vanilla DRUNet variants.

Raw job outputs also live under `~/mtf-smoke/<run-name>/`.

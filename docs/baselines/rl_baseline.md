# Richardson-Lucy Baseline Report

## Experimental Setup
- **Dataset**: DIV2K_train_LR_bicubic/X2, 256×256 RGB crops, full training split (`--limit 0`).
- **Forward model**: same settings as the Wiener baseline (box/random/legendre codes, T=31, blur length 15 px, duty cycle 0.5, photon budget 1000, read-noise 0.01, seed 0).
- **RL parameters**: iterations=12, damping=0.7, TV weight=0.001, smoothing weight=0.4, smoothing sigma=1.5.
- **Command**:
  ```bash
  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root data \
    --subset train --degradation bicubic --scale X2 \
    --image-mode rgb --limit 0 \
    --method rl --rl-iterations 12 --rl-damping 0.7 \
    --rl-tv-weight 0.001 --rl-smooth-weight 0.4 --rl-smooth-sigma 1.5 \
    --collect-only
  ```
  (Add `--save-recon` without `--collect-only` if you need per-image reconstructions.)
- Outputs land in `src/mtf_aware_deblurring/forward_model_outputs/reconstruction/rl/` (CSV + optional PNGs).

## Quantitative Summary
Per-image PSNR values for all 800 frames × 3 patterns are stored in `forward_model_outputs/reconstruction/rl/rl_psnr.csv`. Aggregated averages:

| Pattern | Mean PSNR (dB) | Samples |
|---------|----------------|---------|
| box     | 19.01          | 800     |
| random  | 19.00          | 800     |
| legendre| 19.02          | 800     |

RL benefits strongly from the damping + smoothing regularization; without it, PSNR collapses under the Poisson–Gaussian noise. Even so, RL remains noisier than Wiener but serves as a classical iterative-deconvolution reference point for upcoming ADMM/PNP methods.

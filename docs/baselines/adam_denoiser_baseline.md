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

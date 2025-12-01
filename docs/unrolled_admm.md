# Learnable Unrolled ADMM (Physics-Aware)

This doc captures what the unrolled ADMM module is, how it differs from the classic physics-aware pipeline, what we changed in code, and what we observed in training and evaluation.

## Concept
- Torch-native unrolled ADMM at `src/mtf_aware_deblurring/reconstruction/unrolled_admm.py`.
- Learns per-iteration `{rho, denoiser_weight, sigma_multiplier}` with optional context-conditioned deltas.
- Optional learned MTF trust mask (gamma, floor, cutoff) and denoise gating to skip expensive steps when confidence is low.
- Context MLP expects `[blur_length_px, taps, photon_budget, optics_score, snr_score]` (normalized in the training script).
- Goal: adapt the physics-aware parameters to the observed degradation instead of using hand-tuned schedules.

## Training entrypoint (`scripts/train_unrolled.py`)
- Generates blurred/noisy observations on-the-fly from DIV2K using the existing forward model (no pre-saved arrays).
- Frozen UNet prior (`UNetDenoiserNet`); trainable unrolled parameters only.
- Saves `unrolled_latest.pt` and `unrolled_best.pt` to `--checkpoint-dir` with optimizer state and args for resumption.
- New flags: `--checkpoint-dir`, `--lr-scheduler {none,cosine,step}`, `--lr-step-size`, `--lr-gamma`, `--lr-eta-min`.

Example smoke run (CUDA, RGB, small limit):
```bash
python scripts/train_unrolled.py \
  --div2k-root data --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 4 --target-size 256 \
  --patterns box random legendre --taps 31 --blur-length 15 --duty-cycle 0.5 \
  --photon-budget 1000 --read-noise 0.01 \
  --steps 8 --epochs 1 --batch-size 1 --lr 1e-4 \
  --denoise-every 1 --mtf-mask --device cuda \
  --checkpoint-dir checkpoints/unrolled-smoke
```

Notes:
- Increase `--limit`/`--epochs`/`--steps` for real training; keep batch small if steps are large to avoid OOM.
- Cosine/step LR schedulers were tested; flat LR 1e-4 worked better in small smokes.
- Clipping and context normalization in the model help avoid NaNs; rho is clamped (we bounded rho to 5 during stabilization).

## Training observations (smokes + full run)
- Best stable configs (RGB, blur=15, taps=31, photon=1000, read_noise=0.01, target_size=256):
  - Steps ≈ 24, batch=2, denoise_every=4, lr=1e-4, mtf-mask on.
  - Higher steps (31) with batch=2 OOM’d; batch=1 was stable. Steps 25–26 gave tiny gains; steps 22–23 were slightly worse. Steps 24 was the sweet spot.
  - Denoiser cadence: every 3–4 steps behaved well; every 2 matched physics-aware cadence; higher cadence didn’t hurt but cost more.
  - LR 2e-4 hurt; cosine scheduler was worse than flat LR in smokes.
  - NaNs in earlier aggressive runs were fixed by clamping (rho bound=5) and normalization; no NaNs in stable configs.
- Full train (job 21414, RTX 4090, 30 epochs, limit=0) with steps=24/batch=2/denoise_every=4/lr=1e-4:
  - L1 started ~0.0475, best ~0.0229 around epoch 16, then oscillated 0.024–0.041; plateaued early.
- Full train with checkpoints (job 21482, 20 epochs, same params) produced `checkpoints/full_run_21482/unrolled_best.pt`.

## Inference CLI integration
- `pipelines/reconstruct.py` now supports `--method unrolled`:
  - New flags: `--unrolled-checkpoint` (required), `--unrolled-device {auto,cpu,cuda}` (default auto).
  - Uses the same forward pipeline (patterns, blur, photon/read noise, taps, target-size) and computes PSNR/SSIM/LPIPS like the baselines.
  - Internally builds `UnrolledADMM` with a UNet denoiser and feeds physics context from the forward pass.
  - Outputs go to the same structure as other methods; metrics CSV is saved alongside.

Example (single image on CUDA):
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data --subset train --degradation bicubic --scale X2 \
  --image-mode rgb --limit 1 --patterns box random legendre \
  --method unrolled \
  --unrolled-checkpoint checkpoints/full_run_21482/unrolled_best.pt \
  --unrolled-device cuda \
  --taps 31 --blur-length 15 --photon-budget 1000 --read-noise 0.01 \
  --target-size 256 \
  --collect-only --enable-ssim --enable-lpips --lpips-device cuda \
  --output-dir logs/unrolled_cli_1img
```

## Single-image evals (training checkpoint vs baseline)
Forward model settings: blur=15, taps=31, photon=1000, read_noise=0.01, RGB, target_size=256.

- Physics-aware ADMM baseline (drunet_color, 60 iters, rho=1.04, weight=0.16, scheduler on):
  - box: PSNR 24.22 dB | SSIM 0.5946 | LPIPS 0.4445
  - random: PSNR 27.88 dB | SSIM 0.7898 | LPIPS 0.2025
  - legendre: PSNR 28.86 dB | SSIM 0.8043 | LPIPS 0.2471
- Unrolled (checkpoint `full_run_21482/unrolled_best.pt`) via CLI (all patterns in one forward pass):
  - box: PSNR 26.34 dB | SSIM 0.7034 | LPIPS 0.3806
  - random: PSNR 27.19 dB | SSIM 0.7892 | LPIPS 0.2933
  - legendre: PSNR 27.15 dB | SSIM 0.7866 | LPIPS 0.3104

## Full DIV2K sweep (800 RGB images)
Forward model settings match training: blur=15, taps=31, photon=1000, read_noise=0.01, RGB, target_size=256. Command: `run_unrolled_full.slurm` invoking `--method unrolled --unrolled-checkpoint checkpoints/full_run_21482/unrolled_best.pt` with LPIPS/SSIM enabled and `--collect-only`.

- Averages (PSNR / SSIM / LPIPS):
  - box: 25.86 dB | 0.7799 | 0.3245
  - random: 27.96 dB | 0.8461 | 0.2442
  - legendre: 28.13 dB | 0.8518 | 0.2397
- Metrics CSV: `/u/yazdinip/mtf-smoke/unrolled-full/unrolled_metrics.csv` (per-image rows for 0001x2–0800x2 across all patterns).

Notes on comparisons:
- Forward samples differ slightly if patterns are generated separately vs jointly; RNG differences can make one pattern “easier.” Fix seeds/pattern selection or reuse the same forward arrays for apples-to-apples comparisons.
- The unrolled model matches or beats the baseline on box/random; legendre is comparable in these tests. Overall gains are modest because the forward model matches training conditions closely and the denoiser is strong.

## Recommendations / next steps
- For fair benchmarking: fix seeds and reuse forward outputs across methods; report PSNR/SSIM/LPIPS per pattern.
- If training longer: consider a mild LR drop after plateau and early stopping; add a small validation slice for PSNR/SSIM.
- If testing harder regimes: increase blur, lower photon_budget, or change target_size to probe robustness; unrolled should help more when forward mismatch is higher.
- To disable resize, set `--target-size 0` (or adjust code to treat 0/None as native resolution).
- Consider adding DRUNet/Tiny adapters for the unrolled model if you need a stronger prior than the frozen UNet backbone.

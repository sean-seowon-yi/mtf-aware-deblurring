# Physics-Aware ADMM Report (Nov 2025)

This report summarizes the evolution of the physics-aware ADMM pipeline, key code changes, and measured outcomes. It is organized chronologically and by component so changes and their effects are clear without referring to branch names or informal notes.

---

## Baseline (original/main)
- **Rationale:** Start simple—fixed ADMM hyperparameters, isotropic masks, and a scheduler that only nudges rho/weight slightly. Measure how much physics context alone helps vs fixed baselines.
- **Pipeline:** Fixed ADMM parameters (rho ~0.4, denoiser weight 1.0), optional heuristic scheduler with mild per-pattern tweaks, isotropic MTF trust mask (gamma+floor) when enabled.
- **Context plumbing:** `PhysicsContext` carries per-pattern MTF/SSNR/PSF; scheduler hook exposed via `--use-physics-scheduler`.
- **Results (grayscale DIV2K X2, DnCNN, 25 iters):** scheduler + isotropic mask improved PSNR modestly vs fixed-rho baseline (e.g., 5 imgs: box ~24.4, random ~25.9, legendre ~26.0; 20 imgs: box 22.45, random 24.18, legendre 24.62).
- **Heuristics removed:** anisotropic/dual-band masks, SSNR-driven rho scaling, sigma-driven denoiser blend (all regressed metrics).
- **Takeaway:** Physics context plus a mild scheduler already beats a fixed baseline, but the gains plateau quickly—use this stage as a correctness reference, not as the final configuration.

### Early flags and behavior
- `--admm-mtf-scale` / `--admm-mtf-floor`: isotropic mask; disabling by default was often better.
- Scheduler: small rho/weight nudges; no geometric ramp; rho_min/rho_ramp briefly added then removed.
- DRUNet sigma: explicit sigma often ignored when a schedule was set (latent bug).

---

## Pending-changes phase (MTF options, DRUNet, CUDA smoke)
- **Reasoning:** Once DRUNet was available, the question shifted from “does physics help?” to “which hooks actually move the needle?” The goal in this phase was *separation of concerns*: treat MTF masks, sigma adapt, and the scheduler as independent levers, exercise each on small RGB/CUDA smoke tests, and only then consider combinations. This avoided attributing gains to the wrong mechanism.
- **New knobs:** `--admm-mtf-weighting-mode` (`none` default, `gamma`, `wiener`, `combined`), Wiener tau/alpha/floor, `--admm-mtf-sigma-adapt` (MTF-quality sigma scaling). Optional rho ramp (later dropped).
- **Findings (CUDA, DRUNet color, DIV2K X2 RGB):**
  - *Mask off wins:* no MTF mask + scheduler beat gamma/Wiener/combined on 3–5 image smoke tests (30–60 iters).
  - *Scheduler helps:* scheduler ON lifted PSNR/SSIM vs fixed rho; rho ramp removed to avoid conflict.
  - *Sigma adapt mixed:* legendre up, box/random down → left off by default.
  - *Combined/Wiener:* unstable or neutral; gamma/none preferred.
- **Representative runs:**
  - 30 iters, 3 imgs, scheduler ON, no mask: box 25.21 / random 27.13 / legendre 26.72.
  - 30 iters, 3 imgs, scheduler ON + sigma adapt: box 24.62 / random 26.03 / legendre 26.97.
  - 60 iters, 2 imgs, scheduler ON, no mask: box 26.72 / random 28.85 / legendre 27.40.
  - 60 iters, 5 imgs, combined or Wiener: PSNR regressions vs no mask.
- **Artifacts:** `forward_model_outputs/reconstruction/admm/admm-*` (first5-60-*; cuda-combined; second-60-*; single; gpunode5-smoke, etc.).

Takeaways: default to no MTF mask; keep scheduler ON; sigma adapt optional; DRUNet sigma path needs fixing.

---

## Modification phase (current code)
- **Reasoning:** The previous phase made three things clear:
  1) The scheduler was consistently helpful, but its effect was muted by conservative rho updates.
  2) MTF masks were fragile and often neutral or harmful; they should be treated as secondary, not primary, physics hooks.
  3) Sigma plumbing into DRUNet was incomplete, so sigma-based heuristics could not realize their potential.
  
  The modification work therefore focused on *one clear storyline*: give the scheduler real authority (a deep-dive rho ramp), make sigma control explicit end-to-end, and simplify the ADMM core so future physics heuristics are easier to implement and reason about. Only after this refactor does it make sense to revisit MTF weighting and sigma adaptation in a controlled way.
### Major code changes
- **Adaptive scheduler:** optics/SNR-aware geometric rho ramp (deep start ~0.02× base, clamp for weak optics, end ~1.2×) and weight modulation. Replaces legacy pattern heuristics.
- **DRUNet sigma fix:** denoiser now honors explicit sigma argument; MTF-quality multipliers take effect.
- **MTF heuristics:** pattern-specific gamma/floor defaults; combined mask normalized; sigma multiplier `1 + 0.4*(1-Q)` logged per pattern.
- **ADMM core refactor:** `_core_pnp_admm` with weighted x-update, prox sigma plumbed, cleaner trace; optional internal rho ramp (unused).
- **CLI cleanup:** grouped args, defaults to new scheduler; logs MTF quality/sigma multipliers.

### What the scheduler actually does today (audit)
- Inputs consumed: `base_rho`, `base_weight`, and per-pattern optics score from `PhysicsContext.quality_metrics()` (mean MTF ×3 clipped to [0,1]). An SNR score is computed but **not used**.
- Per-iteration outputs: `rho` ramps geometrically from ~0.02× (clamped to 0.15× for optics_score < 0.15) to ~1.2× `base_rho`. `denoiser_weight` is **static** per pattern (`base_weight * (1 + 0.2*(1 - optics_score))` clipped to ≤1). `sigma` and `prior_choice` fields in the decision are unused downstream; `extras` only flow to traces.
- Interaction with ADMM core: when a scheduler is present, the internal rho ramp in `_core_pnp_admm` is disabled; only the scheduler’s ramp runs.
- Context usage: pattern contexts from the forward model are passed through, but only mean MTF is consumed; kernel/PSF/SSNR/photon budget are otherwise unused by the scheduler.

### Abandoned/low-yield ideas (documented for future avoidable work)
- **Relay (tiny → DRUNet) denoising:** Idea was to start with a cheap denoiser for early artifact cleanup, then switch to DRUNet. In practice, the scheduler’s low-rho start already makes DRUNet “gentle” early; adding a relay adds VRAM overhead and complexity with no observed gains on smoke tests.
- **Sigma map to DRUNet:** We briefly wired a Poisson-like spatial sigma map (`sqrt(I)`) into the DRUNet adapter and prox. On a 1-image CUDA test, box improved slightly but random/legendre collapsed (PSNR down ~5 dB for random). Likely cause: weights trained on uniform AWGN don’t benefit from a spatial map; without map-trained DRUNet weights, this is a distribution shift that hurts.
- **Weight ramping:** Not implemented; weight stays static. Current behavior keeps the anneal on `rho` only to avoid double-annealing and keep tuning simple.

### Core configs evaluated (CUDA, DRUNet color, 60 iters, DIV2K X2 RGB)
- Scheduler ON, no mask (rho 0.4): box 26.84 / random 26.74 / legendre 27.41 (5 imgs).
- Scheduler ON + sigma adapt (no mask): box 26.50 / random 28.30 / legendre 29.43 (5 imgs).
- Combined/gamma + sigma adapt (base rho 0.9, weight 0.20): box 25.30 / random 28.78 / legendre 29.56; SSIM 0.7006 / 0.8221 / 0.8503 (5 imgs).
- 10-img check:
  - rho 0.9 / weight 0.20 combined: box 24.49 / random 29.05 / legendre 29.25.
  - rho 1.15 / weight 0.18 combined: box 24.28 / random 28.77 / legendre 29.27 (no legendre gain).

### MTF weighting ablations (CUDA, 5 imgs, 60 iters, sigma adapt + scheduler)
- None: box 18.59 / random 19.86 / legendre 21.08 (fails).
- Gamma = Combined (same outputs in this codepath): box 25.30 / random 28.78 / legendre 29.56.
- Wiener: box 25.94 / random 24.32 / legendre 23.34 (random/legendre regress).

### Rho/weight sweeps (CUDA, 5 imgs, 60 iters, gamma/combined, sigma adapt, scheduler ON)
- Grid: rho {0.75, 0.85, 0.95, 1.05, 1.15} × weight {0.16, 0.18, 0.20, 0.22} × mode {gamma, combined}.
- Best balanced (total PSNR): rho ~0.75–0.95, weight 0.20 (gamma/combined identical). Example rho0.75_w0.20: box 25.46 / random 28.70 / legendre 29.52.
- Best per-pattern (5 imgs): box 25.68 (rho0.75_w0.16_gamma), random 28.82 (rho0.95_w0.22_combined), legendre 29.66 (rho1.15_w0.18_combined). Gains vs baseline are small (<0.2 dB).
- Sweep summary: `~/mtf-smoke/admm-sweeps-mod/sweep_summary.csv`. 50-image sweep in progress at `~/mtf-smoke/admm-sweeps-mod50` (log `~/mtf-smoke/admm-sweeps-mod50.log`).

### Current conclusion
- **Thought process:** We explored multiple axes (mask modes, rho/weight grids, sigma multipliers) and found diminishing returns beyond the scheduler+s sigma fixes. The sweeps confirm the plateau: combinations shuffle box/random/legendre within ~0.2 dB, so the default should emphasize simplicity and reproducibility rather than chasing those tiny swings. Legendre-heavy settings (rho 1.15 / weight 0.18) look promising on small subsets, but 10-image averages show no net gain; keep them as optional presets rather than defaults.

---

## How to reproduce
- **Single run (balanced default):**
  ```
  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root data \
    --subset train --degradation bicubic --scale X2 \
    --image-mode rgb --limit 5 \
    --patterns box random legendre \
    --method admm \
    --denoiser-type drunet_color --denoiser-device cuda \
    --admm-iters 60 --admm-rho 0.9 --admm-denoiser-weight 0.20 \
    --admm-mtf-weighting-mode gamma --admm-mtf-sigma-adapt \
    --enable-ssim --use-physics-scheduler --save-recon
  ```
- **Sweep (modification code):** `bash scripts/admm_sweep_modification.sh [OUT_ROOT] [DIV2K_ROOT] [IMG_LIMIT]` (CUDA, DRUNet color, scheduler + sigma adapt, 60 iters; rho/weight grid above; mode=gamma only). Run inside tmux on the GPU node to keep it alive. Summary at `[OUT_ROOT]/sweep_summary.csv`.

---

## Code change trace (high impact diffs)
- `reconstruction/admm_denoiser.py`: refactored to `_core_pnp_admm`; weighted x-update; prox sigma honored; MTF mask handling normalized; sigma multiplier logging.
- `reconstruction/prior_scheduler.py`: AdaptivePhysicsScheduler replaces heuristic scheduler; optics/SNR scoring; geometric rho ramp; weight modulation.
- `denoisers/drunet_adapter.py`: explicit sigma argument respected; denoiser callable signature now `f(x, sigma=None)`.
- `pipelines/reconstruct.py`: CLI regrouped; defaults to AdaptivePhysicsScheduler; MTF/sigma flags exposed; scheduler fed base rho/weight.

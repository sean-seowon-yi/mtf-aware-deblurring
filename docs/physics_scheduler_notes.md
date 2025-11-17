# Physics-Aware Prior Scheduler Progress (Dec 2025)

This note captures what we implemented, what was tried, and the current state of the physics-aware ADMM prior scheduler. It is meant as a living log to avoid re-running the same experiments and to clarify which heuristics are active in the codebase.

## Implemented plumbing
- **PhysicsContext**: Forward model now stores per-pattern physics metadata (code, PSF/kernel, MTF, SSNR, noise stats) and exposes it through `ForwardBatch.pattern_contexts` (`pipelines/forward.py`, `pipelines/common.py`).
- **Scheduler hook**: ADMM accepts a `PhysicsAwareScheduler` that can adjust `rho` and denoiser blend each iteration. A heuristic scheduler is provided and wired via `--use-physics-scheduler` in `pipelines/reconstruct.py`.
- **MTF trust mask**: The ADMM x-update applies a per-pattern trust mask derived from the MTF (`mtf^gamma` with a floor), down-weighting bands the optics cannot reliably transfer so the prior can dominate there.
- **Sigma plumbing**: The scheduler computes a sigma scale and passes it to denoisers that expose `sigma_scale` (e.g., DRUNet adapters), but the current mixing weight is not scaled by sigma (we rolled that back after it hurt PSNR).

## Heuristics tried (and current status)
- **Isotropic MTF weighting (active)**: Single trust mask per pattern (`gamma=mtf_scale`, `floor=mtf_floor`). Best small-set legendre numbers (~26.0–26.1 dB) came from this variant.
- **Anisotropic / dual-band MTF masks (reverted)**: Tried horizontal/vertical profiles and low/high band splits to damp only weak orientations; did not improve legendre PSNR and was removed.
- **SSNR-driven rho scaling (reverted)**: Used banded SSNR to adjust `rho`; degraded results, removed.
- **Sigma-driven denoiser blend (reverted)**: Scaled the denoiser mixing weight by the scheduler’s sigma; regressed legendre PSNR, removed. Sigma is still passed through to denoisers that natively support it.

## Benchmark snapshots
- **5-image DIV2K X2 grayscale, DnCNN, 25 iters, rho=0.45, scheduler ON (isotropic MTF)**: box ~24.4 dB, random ~25.9 dB, legendre ~26.0–26.1 dB (best observed on small set).
- **20-image DIV2K X2 grayscale, DnCNN, 25 iters, rho=0.45, scheduler ON (isotropic MTF)**: box 22.45 dB, random 24.18 dB, legendre 24.62 dB. Baseline (no scheduler) on same set was box 21.98 / random 23.51 / legendre 23.72.
- Other variants (anisotropic masks, SSNR rho, sigma-blend) did not surpass the above; most regressed legendre PSNR.

## Current code state (Dec 2025)
- ADMM uses isotropic MTF trust masks for all patterns; no anisotropic/dual-band logic.
- Scheduler adjusts `rho` and denoiser blend per iteration; sigma-scale is computed but only forwarded to denoisers that expose it (blend weight is not sigma-scaled).
- CLI flag `--use-physics-scheduler` enables the scheduler; without it, ADMM runs the original fixed-parameter baseline.
- DnCNN is the working denoiser in these runs; TinyDenoiser weights are incompatible with the current architecture, and DRUNet is available but not benchmarked here.

## Next candidates (not implemented)
- Pattern-specific rho/sigma schedules (legendre-only adjustments) while keeping isotropic MTF masks.
- Switch to DRUNet and use its native sigma scheduling with the physics context.
- Learned controller (policy) trained on logged runs to choose rho/weights/sigma from context.

# Documentation Index

Core references:
- `project_proposal.pdf` — original submission outlining motivation, related work, goals, and timeline.
- `proposal_summary.md` — condensed text version of the proposal for quick scanning.
- `forward_model_overview.md` — packaged forward simulator, DIV2K loader, and configuration options (includes Wiener/RL CLI examples).

Baselines and reports:
- `baselines/wiener_baseline.md` — setup, qualitative crops, and quantitative tables for the Wiener baseline.
- `baselines/rl_baseline.md` — Richardson–Lucy configuration, command, and PSNR summary.
- `baselines/adam_denoiser_baseline.md` — full DIV2K sweep for ADAM + Tiny/DnCNN/UNet priors.
- `baselines/admm_pnp_baseline.md` — ADMM + denoiser variants, physics-aware scheduler settings, and DRUNet sweeps.
- `physics_scheduler_notes.md` — chronological notes for the physics-aware ADMM scheduler and MTF experiments.
- `experiments/noise_runs.md` — noise-profile sweeps (Noise1/Noise2) with RL/ADAM/ADMM comparisons.

New learnable path:
- `unrolled_admm.md` — how to train and use the learnable unrolled ADMM/PnP module from the latest commit (`scripts/train_unrolled.py`, checkpoints, context features).

Operational aids:
- `gpu_job_guide.md` — Slurm and tmux tips for running baselines/sweeps on lab GPU nodes.
- `poster_script.md` and `CSC2529_Final_Poster.pdf` — poster assets and script for the course presentation.

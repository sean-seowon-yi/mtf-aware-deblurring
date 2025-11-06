# MTF-Aware Deblurring

Physics-aware motion deblurring toolkit that simulates coded-exposure capture, evaluates modulation transfer functions (MTFs), and prepares the groundwork for plug-and-play (PnP) reconstruction with adaptive priors. This repository is bootstrapped from the original Colab forward-model notebook and the CSC2529 project proposal.

## Highlights
- Forward simulator for box, random, and Legendre flutter-shutter patterns
- Poisson-Gaussian noise injection and spectral SNR diagnostics under photon budgets
- Extensible runner API (`ForwardModelRunner`, `run_forward_model`) for future PnP / HQS integration
- Documentation assets sourced from the course proposal with summarized objectives and timeline

## Getting Started
1. Create and activate a virtual environment (example uses `venv`):
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```
2. Install the core dependencies and project package:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```
3. Smoke-test the forward model (plots will open via Matplotlib):
   ```bash
   python -m mtf_aware_deblurring.forward_pipeline
   ```
   Generated figures and arrays are written to `forward_model_outputs/` when the saving flags are enabled.

## Repository Layout
- `src/mtf_aware_deblurring/runner.py` - core forward-imaging runner and CLI demo harness
- `src/mtf_aware_deblurring/forward_pipeline.py` - compatibility shim that re-exports the public API for `python -m ...`
- `src/mtf_aware_deblurring/{patterns,optics,noise,metrics,synthetic,utils}.py` - modular helpers for exposure codes, PSFs/MTFs, noise models, and synthetic scenes
- `docs/project_proposal.pdf` - original course proposal detailing motivation, prior work, milestones, and evaluation plan
- `docs/proposal_summary.md` - text summary extracted from the proposal for quick reference
- `pyproject.toml` / `requirements.txt` - packaging metadata and runtime dependencies

## Next Steps
- Wire up classical deconvolution baselines (Wiener, Richardson-Lucy) using the exported PSFs
- Integrate plug-and-play / HQS solvers with physics-aware denoising schedules informed by the simulated MTF
- Run the ablation grid (box vs coded exposure, schedule variants, code families) under equal photon budgets
- Extend documentation with experiment tracking, data management, and reproducibility checklists as the project matures

For full context, read the [proposal summary](docs/proposal_summary.md) or the original [project proposal PDF](docs/project_proposal.pdf).

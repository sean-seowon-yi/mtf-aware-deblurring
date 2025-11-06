# Contributing

## Development Environment
1. Create a fresh virtual environment.
2. Install runtime requirements via `pip install -r requirements.txt`.
3. Install the package in editable mode with `pip install -e .` to pick up local changes.

## Coding Guidelines
- Prefer pure NumPy for prototyping; isolate heavy GPU dependencies inside upcoming PnP modules.
- Keep plotting and I/O code optional via feature flags (`save_arrays`, `save_figures`).
- Add succinct docstrings for new public functions or classes.
- Run `python -m mtf_aware_deblurring.forward_pipeline` and verify plots render before submitting changes.

## Documentation
- Update `docs/proposal_summary.md` when milestones or objectives shift.
- Add experiment notes under `docs/` (e.g., `docs/experiments/<date>-<name>.md`).
- Ensure README sections stay synchronized with new modules or interfaces.

## Issue Tracking
- Tag issues as `simulation`, `reconstruction`, `docs`, or `infra` to group workstreams.
- Link relevant sections of the proposal when opening new feature requests to preserve context.

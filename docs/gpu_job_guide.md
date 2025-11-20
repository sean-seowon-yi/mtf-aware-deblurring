# GPU Job Guide (Slurm)

## SSH entrypoint
- Use your SSH config alias (e.g., `Host comps2` with `IdentityFile ~/.ssh/codex_lab`), then: `ssh comps2`
- Explicit command: `ssh -i ~/.ssh/codex_lab yazdinip@comps2.cs.toronto.edu`

## Lessons / fast path
- Start a tmux before grabbing the GPU shell (`tmux new -s codex_gpu`) so the `srun --pty bash -l` session stays alive for multiple runs.
- Keep env and caches in `/tmp/$USER/...` to avoid quota issues: `python3 -m venv /tmp/$USER/envs/CSC2529` and `export PIP_CACHE_DIR=/tmp/$USER/pip-cache`.
- Install project and missing deps in one go: `pip install --no-cache-dir -e . scipy` (SciPy is required for the reconstruction import path).
- When possible, attach to tmux and paste commands directly instead of `tmux send-keys` to dodge quoting errors.

## Quick interactive GPU shell
- Request a GPU node and get a PTY shell:
  ```
  srun --partition=gpunodes --gres=gpu:rtx_4090:1 \
       --cpus-per-task=8 --mem=24G --time=01:00:00 \
       --pty bash -l
  ```
- Inside the shell you can:
  - Check the GPU: `nvidia-smi`
  - Activate or create your venv (e.g., `source /tmp/$USER/envs/CSC2529/bin/activate` or `python3 -m venv /tmp/$USER/envs/CSC2529`)
  - Run project commands directly (see "Run a single-image job" below)

## Submit a batch GPU job
- Use the provided Slurm script: `scripts/test_admm_physics.sh`
  - It installs the project in editable mode inside a venv at `/tmp/$USER/envs/CSC2529` and uses a pip cache at `/tmp/$USER/pip-cache` to stay under home quota.
  - Defaults: dataset at `$HOME/datasets/DIV2K`, output at `$HOME/mtf-smoke/admm-physics`.
  - Submit with optional overrides (example runs 1 image):
    ```
    cd ~/mtf-aware-deblurring
    LIMIT=1 OUT_DIR=$HOME/mtf-smoke/admm-physics \
    sbatch scripts/test_admm_physics.sh
    ```
  - Monitor: `squeue -u $USER`
  - Logs land in `%x-%j.out` (e.g., `test-admm-physics-12345.out`)

## Run a single-image job interactively
- After getting a GPU shell, invoke the pipeline directly:
  ```
  cd ~/mtf-aware-deblurring
  source /tmp/$USER/envs/CSC2529/bin/activate  # or your env
  python -m pip install --upgrade pip wheel
  pip install --no-cache-dir -e .

  python -m mtf_aware_deblurring.pipelines.reconstruct \
    --div2k-root $HOME/datasets/DIV2K \
    --subset train \
    --degradation bicubic \
    --scale X2 \
    --image-mode rgb \
    --limit 1 \
    --auto-download \
    --method admm \
    --admm-iters 60 \
    --admm-rho 0.4 \
    --admm-denoiser-weight 1.0 \
    --admm-mtf-scale 0.35 \
    --admm-mtf-floor 0.05 \
    --denoiser-type drunet_color \
    --denoiser-device cuda \
    --use-physics-scheduler \
    --collect-only \
    --output-dir $HOME/mtf-smoke/admm-physics
  ```
- Adjust ADMM/denoiser flags as needed; `--limit 1` keeps it to one image.

## Tips
- Keep caches in `/tmp/$USER/...` to avoid quota issues in home.
- If the lab has different GPU SKUs, change `--gres=gpu:rtx_4090:1` to the available type (e.g., `rtx_a6000`).
- For CPU-only tests, switch to `--partition=cpunodes` and drop `--gres`.

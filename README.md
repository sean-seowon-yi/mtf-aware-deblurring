# Physics-Aware Deblurring with Coded Exposure

A research toolkit for coded-exposure motion deblurring with physics-aware reconstruction. Addresses the "Spectral Gap" where standard methods fail in low-light by blindly amplifying noise in spectral nulls.

For detailed report, refer to [report](report/Physics-Aware_Deblurring_Report.pdf)

For poster, refer to [poster](report/Physics-Aware_Deblurring_Poster.pdf)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

(For full CLI commands, refer to `src/mtf_aware_deblurring/pipelines`)

**Generate blurred images:**
```bash
python -m mtf_aware_deblurring.forward_pipeline \
  --div2k-root data --subset train --scale X2 \
  --image-mode rgb --limit 10 --auto-download
```

**Reconstruct with Physics-Aware ADMM:**
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data --subset train --scale X2 \
  --image-mode rgb --limit 10 --auto-download \
  --method admm --admm-iters 60 --admm-rho 0.4 \
  --admm-mtf-weighting-mode combined \
  --use-physics-scheduler \
  --denoiser-type drunet_color \
  --denoiser-device cuda --collect-only
```

---

## Reconstruction Methods

### Wiener Deconvolution
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --method wiener --wiener-k 1e-3 [dataset flags]
```

### Richardson-Lucy
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --method rl --rl-iterations 12 --rl-damping 0.7 [dataset flags]
```

### PnP-ADAM
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --method adam --adam-iters 30 --adam-lr 0.065 \
  --denoiser-type dncnn --denoiser-device cuda [dataset flags]
```
*Denoisers:* `tiny` (CPU), `dncnn` (best for ADAM), `unet`, `drunet_color/drunet_gray`

### Physics-Aware ADMM

**Standard regime** (Blur=15px, σ=0.01):
```bash
python -m mtf_aware_deblurring.pipelines.reconstruct \
  --div2k-root data --subset valid --scale X2 --image-mode rgb \
  --method admm --admm-iters 60 --admm-rho 0.4 \
  --admm-mtf-weighting-mode combined --use-physics-scheduler \
  --denoiser-type drunet_color --blur-length 15 \
  --photon-budget 1000 --read-noise 0.01 --collect-only
```

| Method | Box | Random | Legendre |
|--------|-----|--------|----------|
| Vanilla ADMM | 26.84 | 28.23 | 28.11 |
| Physics-Aware | **27.10** | **28.50** | **29.55** |
| Unrolled ADMM | 26.34 | 27.19 | 27.15 |

**Severe regime** (Blur=30px, σ=0.05):
```bash
# Same as above but: --blur-length 30 --read-noise 0.05 \
# --admm-rho 1.04 --admm-denoiser-weight 0.16 --admm-mtf-sigma-adapt
```

| Method | Box | Random | Legendre |
|--------|-----|--------|----------|
| Vanilla ADMM | 10.43 | 13.48 | 13.57 |
| Physics-Aware | **20.79** | **23.59** | **24.04** |
| Unrolled ADMM | 18.52 | 20.16 | 20.44 |

---

## Key Parameters

**ADMM Physics-Aware:**
- `--admm-mtf-weighting-mode`: `none`, `gamma`, `wiener`, `combined` (Trust Map strategy)
- `--use-physics-scheduler`: Dynamic ρ/weight adjustment based on MTF quality
- `--admm-mtf-sigma-adapt`: Boost sigma for poor kernels
- `--admm-denoiser-interval N`: Apply denoiser every N iterations (default: 2)

**Denoisers:**
- `--denoiser-type`: `tiny`, `dncnn`, `unet`, `drunet_color`, `drunet_gray`
- `--denoiser-device`: `cpu`, `cuda`, `dml`
- `--denoiser-sigma-scale`: Sigma scaling (default: 8.0)

---

## Unrolled ADMM

Train end-to-end learnable ADMM:
```bash
python scripts/train_unrolled.py \
  --div2k-root data --subset train --scale X2 \
  --image-mode grayscale --target-size 256 \
  --steps 8 --epochs 1 --mtf-mask --device cuda \
  --checkpoint-dir checkpoints/unrolled
```

Inference:
```python
from mtf_aware_deblurring.reconstruction import UnrolledADMM, UnrolledADMMConfig
from mtf_aware_deblurring.denoisers.unet_denoiser import UNetDenoiserNet

model = UnrolledADMM(UNetDenoiserNet(channels=1), UnrolledADMMConfig(), channels=1)
model.load_state_dict(torch.load("checkpoint.pt")["model_state"])
recon, _ = model(obs, kernel, mtf=mtf_map, context=ctx_features)
```

---

## Training Denoisers

```bash
python scripts/train_tiny_denoiser.py  # 10-layer residual CNN
python scripts/train_unet_denoiser.py --device cuda  # UNet for Poisson-Gaussian
```

---

## Repository Structure

```
src/mtf_aware_deblurring/
├── pipelines/          # CLI entry points
├── reconstruction/     # Deblurring algorithms
├── denoisers/          # Denoiser backends
├── datasets.py         # DIV2K loader
├── patterns.py         # Exposure codes
├── optics.py           # PSF/OTF/MTF
├── noise.py            # Poisson-Gaussian
├── metrics.py          # PSNR/SSIM/LPIPS
└── synthetic.py        # Test patterns

scripts/                # Training scripts
docs/baselines/         # Detailed results
```

---

## Citation

```bibtex
@inproceedings{yazdinia2025physics,
  title={Physics-Aware Deblurring with Coded Exposure},
  author={Yazdinia, Pedram and Yi, Seo Won},
  booktitle={CSC2529 Course Project},
  year={2025},
  organization={University of Toronto}
}
```

**License:** MIT

**Credits:** DRUNet ([deepinv/drunet](https://huggingface.co/deepinv/drunet)), DnCNN, DIV2K, CSC2529 @ UofT

**Repository:** https://github.com/sean-seowon-yi/mtf-aware-deblurring








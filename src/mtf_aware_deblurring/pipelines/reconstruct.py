from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np

from .common import run_forward_batches
from ..reconstruction import (
    run_wiener_baseline,
    run_richardson_lucy_baseline,
    run_adam_denoiser_baseline,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run reconstruction baselines on DIV2K using the forward model.")
    parser.add_argument("--div2k-root", type=Path, required=True, help="Path containing DIV2K_* folders.")
    parser.add_argument("--subset", default="train", choices=["train", "valid"], help="DIV2K subset.")
    parser.add_argument("--degradation", default="bicubic", help="DIV2K degradation label.")
    parser.add_argument("--scale", default="X2", help="DIV2K scale folder (X2, X3, X4, ...).")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to process (0 = all).")
    parser.add_argument("--target-size", type=int, default=256, help="Optional square resize after loading.")
    parser.add_argument("--image-mode", choices=["grayscale", "rgb"], default="grayscale", help="Color mode.")
    parser.add_argument("--auto-download", action="store_true", help="Download DIV2K subset automatically if missing.")
    parser.add_argument("--patterns", nargs="+", default=["box", "random", "legendre"], help="Exposure patterns.")
    parser.add_argument("--selected-pattern", help="Simulate only this pattern.")
    parser.add_argument("--taps", type=int, default=31, help="Number of exposure taps (T).")
    parser.add_argument("--blur-length", type=float, default=15.0, help="Motion blur length in pixels.")
    parser.add_argument("--duty-cycle", type=float, default=0.5, help="Duty cycle for random codes.")
    parser.add_argument("--photon-budget", type=float, default=1000.0, help="Photon budget for Poisson noise.")
    parser.add_argument("--read-noise", type=float, default=0.01, help="Read noise sigma in [0,1].")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--save-arrays", action="store_true", help="Save forward-model arrays.")
    parser.add_argument("--save-figures", action="store_true", help="Save forward-model figures.")
    parser.add_argument("--save-pngs", action="store_true", help="Save forward-model PNGs.")
    parser.add_argument("--save-recon", action="store_true", help="Save reconstructions as PNGs.")
    parser.add_argument("--output-dir", type=Path, help="Base directory for outputs.")
    parser.add_argument("--method", choices=["wiener", "rl", "adam"], default="wiener", help="Reconstruction method to run.")
    parser.add_argument("--wiener-k", type=float, default=1e-3, help="Wiener filter constant k.")
    parser.add_argument("--rl-iterations", type=int, default=30, help="Richardson-Lucy iteration count.")
    parser.add_argument("--rl-damping", type=float, default=1.0, help="Richardson-Lucy damping exponent.")
    parser.add_argument("--rl-tv-weight", type=float, default=0.0, help="Richardson-Lucy TV regularization weight.")
    parser.add_argument("--rl-smooth-weight", type=float, default=0.1, help="Richardson-Lucy Gaussian smoothing weight.")
    parser.add_argument("--rl-smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma.")
    parser.add_argument("--adam-iters", type=int, default=80, help="ADAM iterations.")
    parser.add_argument("--adam-lr", type=float, default=0.04, help="ADAM learning rate.")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="ADAM beta1.")
    parser.add_argument("--adam-beta2", type=float, default=0.995, help="ADAM beta2.")
    parser.add_argument("--adam-denoiser-weight", type=float, default=0.25, help="Blend weight for the deep denoiser (0-1).")
    parser.add_argument("--adam-denoiser-interval", type=int, default=2, help="Apply the denoiser every N ADAM steps.")
    parser.add_argument("--denoiser-weights", type=Path, help="Optional path to TinyDenoiser weights (.pth). Defaults to bundled weights.")
    parser.add_argument("--denoiser-device", choices=["cpu", "cuda"], help="Force denoiser device (defaults to auto).")
    parser.add_argument("--collect-only", action="store_true", help="Skip per-image folders; only write summary CSV.")
    return parser.parse_args(argv)


def save_recon_image(path: Path, method: str, pattern: str, image: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    png = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    filename = f"{method}_{pattern}.png"
    imageio.imwrite(path / filename, png)


def run_method(method: str, batch, args):
    if method == "wiener":
        return run_wiener_baseline(
            batch.image,
            batch.forward_outputs["patterns"],  # type: ignore[index]
            k=args.wiener_k,
        )
    if method == "rl":
        return run_richardson_lucy_baseline(
            batch.image,
            batch.forward_outputs["patterns"],  # type: ignore[index]
            iterations=args.rl_iterations,
            damping=args.rl_damping,
            tv_weight=args.rl_tv_weight,
            smooth_weight=args.rl_smooth_weight,
            smooth_sigma=args.rl_smooth_sigma,
        )
    if method == "adam":
        return run_adam_denoiser_baseline(
            batch.image,
            batch.forward_outputs["patterns"],  # type: ignore[index]
            iterations=args.adam_iters,
            lr=args.adam_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            denoiser_weight=args.adam_denoiser_weight,
            denoiser_interval=args.adam_denoiser_interval,
            denoiser_weights=args.denoiser_weights,
            denoiser_device=args.denoiser_device,
        )
    raise ValueError(f"Unsupported method: {method}")


def main(argv=None):
    args = parse_args(argv)
    summaries: List[Dict[str, object]] = []
    baseline_root: Optional[Path] = None

    persist_outputs = (not args.collect_only) or args.save_recon
    save_recon = args.save_recon
    method = args.method.lower()

    for batch in run_forward_batches(args, baseline_name=method, persist_outputs=persist_outputs):
        if baseline_root is None:
            baseline_root = batch.baseline_root

        recon_results = run_method(method, batch, args)

        for pattern, result in recon_results.items():
            if save_recon and batch.output_dir is not None:
                recon_dir = batch.output_dir / method
                save_recon_image(recon_dir, method, pattern, result.reconstruction)
            summaries.append(
                {
                    "image": batch.image_path.name,
                    "pattern": pattern,
                    "psnr": result.psnr,
                }
            )
            print(f"{batch.image_path.name} [{pattern}] PSNR: {result.psnr:.2f} dB")

    if not summaries or baseline_root is None:
        print("No images processed; nothing to report.")
        return

    pattern_order = {p: idx for idx, p in enumerate(args.patterns)} if hasattr(args, 'patterns') else {}
    summaries.sort(key=lambda row: (row['image'], pattern_order.get(row['pattern'], len(pattern_order))))

    csv_path = baseline_root / f"{method}_psnr.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "pattern", "psnr"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved PSNR summary to {csv_path}")

    avg_by_pattern: Dict[str, List[float]] = {}
    for row in summaries:
        avg_by_pattern.setdefault(row["pattern"], []).append(row["psnr"])
    for pattern, values in avg_by_pattern.items():
        mean_val = sum(values) / len(values)
        print(f"Average {pattern} PSNR: {mean_val:.2f} dB over {len(values)} images")


if __name__ == "__main__":
    main()

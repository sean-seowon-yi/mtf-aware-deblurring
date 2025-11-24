from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import imageio.v2 as imageio
import numpy as np

from .common import run_forward_batches
from ..metrics import lpips_distance
from ..reconstruction import (
    run_wiener_baseline,
    run_richardson_lucy_baseline,
    run_adam_denoiser_baseline,
    run_admm_denoiser_baseline,
    run_admm_diffusion_baseline,
    AdaptivePhysicsScheduler,
)


def _serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse Namespace to JSON-friendly dict."""
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            payload[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            payload[key] = str(value)
    return payload

def _resolve_lpips_device(choice: Optional[str]) -> str:
    if choice and choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run reconstruction baselines on DIV2K using the forward model."
    )

    # --- 1. DATASET & I/O ---
    g_data = parser.add_argument_group("Dataset & I/O")
    g_data.add_argument("--div2k-root", type=Path, required=True, help="Path containing DIV2K_* folders.")
    g_data.add_argument("--subset", default="train", choices=["train", "valid"], help="DIV2K subset.")
    g_data.add_argument("--degradation", default="bicubic", help="DIV2K degradation label.")
    g_data.add_argument("--scale", default="X2", help="DIV2K scale folder (X2, X3, X4, ...).")
    g_data.add_argument("--limit", type=int, default=10, help="Number of images to process (0 = all).")
    g_data.add_argument(
        "--benchmark-preset",
        choices=["small", "long"],
        help="Preset: small=10 imgs, long=200 imgs (overrides --limit unless set explicitly).",
    )
    g_data.add_argument("--target-size", type=int, default=256, help="Optional square resize after loading.")
    g_data.add_argument("--image-mode", choices=["grayscale", "rgb"], default="grayscale", help="Color mode.")
    g_data.add_argument("--auto-download", action="store_true", help="Download DIV2K subset automatically if missing.")
    g_data.add_argument("--output-dir", type=Path, help="Base directory for outputs.")
    g_data.add_argument("--save-recon", action="store_true", help="Save reconstructions as PNGs.")
    g_data.add_argument("--collect-only", action="store_true", help="Skip per-image folders; only write summary CSV.")
    g_data.add_argument("--enable-ssim", action="store_true", help="Also compute/log SSIM (default: PSNR only).")
    g_data.add_argument("--enable-lpips", action="store_true", help="Also compute/log LPIPS (perceptual metric).")
    g_data.add_argument(
        "--lpips-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for LPIPS computation (auto uses CUDA if available).",
    )
    g_data.add_argument(
        "--log-traces",
        action="store_true",
        help="Write per-iteration traces (ADMM denoiser only) to <method>_traces.jsonl.",
    )

    # Forward-model-only outputs
    g_data.add_argument("--save-arrays", action="store_true", help="Save forward-model arrays.")
    g_data.add_argument("--save-figures", action="store_true", help="Save forward-model figures.")
    g_data.add_argument("--save-pngs", action="store_true", help="Save forward-model PNGs.")

    # --- 2. FORWARD MODEL / OPTICS ---
    g_fwd = parser.add_argument_group("Forward Model / Optics")
    g_fwd.add_argument("--patterns", nargs="+", default=["box", "random", "legendre"], help="Exposure patterns.")
    g_fwd.add_argument("--selected-pattern", help="Simulate only this specific pattern from the list.")
    g_fwd.add_argument("--taps", type=int, default=31, help="Number of exposure taps (T).")
    g_fwd.add_argument("--blur-length", type=float, default=15.0, help="Motion blur length in pixels.")
    g_fwd.add_argument("--duty-cycle", type=float, default=0.5, help="Duty cycle for random codes.")
    g_fwd.add_argument("--photon-budget", type=float, default=1000.0, help="Photon budget for Poisson noise.")
    g_fwd.add_argument("--read-noise", type=float, default=0.01, help="Read noise sigma in [0,1].")
    g_fwd.add_argument("--seed", type=int, default=0, help="Base random seed.")

    # --- 3. RECONSTRUCTION METHOD SELECTION ---
    g_method = parser.add_argument_group("Reconstruction Method Selection")
    g_method.add_argument(
        "--method",
        choices=["wiener", "rl", "adam", "admm", "admm_diffusion"],
        default="wiener",
        help="Reconstruction algorithm to run.",
    )

    # --- 4. CLASSIC METHODS (WIENER / RL) ---
    g_classic = parser.add_argument_group("Classic Methods (Wiener / RL)")
    g_classic.add_argument("--wiener-k", type=float, default=1e-3, help="Wiener filter constant k.")
    g_classic.add_argument("--rl-iterations", type=int, default=30, help="Richardson-Lucy iteration count.")
    g_classic.add_argument("--rl-damping", type=float, default=1.0, help="Richardson-Lucy damping exponent.")
    g_classic.add_argument("--rl-tv-weight", type=float, default=0.0, help="Richardson-Lucy TV regularization weight.")
    g_classic.add_argument(
        "--rl-smooth-weight",
        type=float,
        default=0.1,
        help="Richardson-Lucy Gaussian smoothing weight.",
    )
    g_classic.add_argument("--rl-smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma.")

    # --- 5. ADAM OPTIMIZATION ---
    g_adam = parser.add_argument_group("ADAM Optimization")
    g_adam.add_argument("--adam-iters", type=int, default=80, help="ADAM iterations.")
    g_adam.add_argument("--adam-lr", type=float, default=0.04, help="ADAM learning rate.")
    g_adam.add_argument("--adam-beta1", type=float, default=0.9, help="ADAM beta1.")
    g_adam.add_argument("--adam-beta2", type=float, default=0.999, help="ADAM beta2.")
    g_adam.add_argument(
        "--adam-denoiser-weight",
        type=float,
        default=0.25,
        help="Blend weight for the deep denoiser (0-1).",
    )
    g_adam.add_argument(
        "--adam-denoiser-interval",
        type=int,
        default=2,
        help="Apply the denoiser every N ADAM steps.",
    )

    # --- 6. ADMM PNP CORE ---
    g_admm = parser.add_argument_group("ADMM PnP Core")
    g_admm.add_argument("--admm-iters", type=int, default=60, help="ADMM iteration count.")
    g_admm.add_argument(
        "--admm-rho",
        type=float,
        default=0.4,
        help="Augmented Lagrangian penalty parameter (higher = trust data more).",
    )
    g_admm.add_argument(
        "--admm-denoiser-weight",
        type=float,
        default=1.0,
        help="Relaxation weight for denoiser output (1.0 = pure denoiser, <1.0 mixes input back in).",
    )
    g_admm.add_argument(
        "--use-physics-scheduler",
        action="store_true",
        help="Enable heuristic physics-aware scheduler to override rho/weight by iteration.",
    )
    g_admm.add_argument(
        "--admm-denoiser-interval",
        type=int,
        default=2,
        help="Apply the ADMM denoiser every N iterations (1 applies it every iteration).",
    )

    # --- 7. ADMM MTF / PHYSICS AWARENESS ---
    g_mtf = parser.add_argument_group("ADMM MTF / Physics Awareness")
    g_mtf.add_argument(
        "--admm-mtf-weighting-mode",
        choices=["gamma", "wiener", "combined", "none"],
        default="none",
        help="Strategy for Weighted Least Squares. 'none' = standard ADMM.",
    )
    g_mtf.add_argument(
        "--admm-mtf-scale",
        type=float,
        default=0.0,
        help="MTF gamma exponent. 0.0 = auto heuristic based on pattern.",
    )
    g_mtf.add_argument(
        "--admm-mtf-floor",
        type=float,
        default=0.0,
        help="Min trust value for MTF mask. 0.0 = auto heuristic based on pattern.",
    )
    g_mtf.add_argument(
        "--admm-mtf-sigma-adapt",
        action="store_true",
        help="Enable Q-score-based sigma boosting for bad kernels (recommended for optics).",
    )
    # Advanced Wiener-like MTF params
    g_mtf.add_argument("--admm-mtf-wiener-alpha", type=float, default=0.5, help="[Adv] Wiener alpha.")
    g_mtf.add_argument("--admm-mtf-wiener-floor", type=float, default=0.05, help="[Adv] Wiener floor.")
    g_mtf.add_argument("--admm-mtf-wiener-tau-min", type=float, default=1e-4, help="[Adv] Tau min.")
    g_mtf.add_argument("--admm-mtf-wiener-tau-max", type=float, default=1e-1, help="[Adv] Tau max.")

    # --- 8. PRIOR CONFIGURATION (DENOISER / DIFFUSION) ---
    g_prior = parser.add_argument_group("Denoiser / Diffusion Prior")
    g_prior.add_argument(
        "--denoiser-type",
        choices=["tiny", "dncnn", "unet", "drunet_color", "drunet_gray"],
        default="tiny",
        help="Denoiser backbone for ADAM/ADMM.",
    )
    g_prior.add_argument(
        "--denoiser-weights",
        type=Path,
        help="Optional path to denoiser weights (.pth). Defaults to bundled weights.",
    )
    g_prior.add_argument(
        "--denoiser-device",
        choices=["cpu", "cuda", "dml"],
        help="Force denoiser device (defaults to auto).",
    )
    g_prior.add_argument(
        "--denoiser-sigma-scale",
        type=float,
        default=8.0,
        help="Scaling factor for sigma passed to denoiser (Default 8.0, try 2.0 for sharpness).",
    )

    # Diffusion-specific
    g_prior.add_argument(
        "--diffusion-prior-type",
        choices=["tiny_score"],
        default="tiny_score",
        help="Score model backbone for the diffusion prior.",
    )
    g_prior.add_argument("--diffusion-prior-weights", type=Path, help="Path to diffusion score-model weights (.pth).")
    g_prior.add_argument("--diffusion-steps", type=int, default=12, help="Diffusion-score steps per ADMM iteration.")
    g_prior.add_argument(
        "--diffusion-guidance",
        type=float,
        default=1.0,
        help="Guidance scale for the diffusion proximal operator.",
    )
    g_prior.add_argument(
        "--diffusion-noise-scale",
        type=float,
        default=1.0,
        help="Initial noise scale injected before diffusion refinement.",
    )
    g_prior.add_argument("--diffusion-sigma-min", type=float, default=0.01, help="Minimum sigma for diffusion schedule.")
    g_prior.add_argument("--diffusion-sigma-max", type=float, default=0.5, help="Maximum sigma for diffusion schedule.")
    g_prior.add_argument(
        "--diffusion-schedule",
        choices=["geom", "linear"],
        default="geom",
        help="Sigma schedule type for the diffusion prior.",
    )
    g_prior.add_argument(
        "--diffusion-device",
        choices=["cpu", "cuda", "dml"],
        help="Force device for the diffusion prior (defaults to auto).",
    )

    return parser.parse_args(argv)


def save_recon_image(path: Path, method: str, pattern: str, image: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    png = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    filename = f"{method}_{pattern}.png"
    imageio.imwrite(path / filename, png)


def run_method(method: str, batch, args):
    """Dispatcher for reconstruction methods."""
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
            denoiser_type=args.denoiser_type,
        )

    if method == "admm":
        scheduler = None
        if getattr(args, "use_physics_scheduler", False):
            scheduler = AdaptivePhysicsScheduler(
                total_iterations=args.admm_iters,
                base_rho=args.admm_rho,
                base_weight=args.admm_denoiser_weight,
            )

        return run_admm_denoiser_baseline(
            batch.image,
            batch.forward_outputs["patterns"],  # type: ignore[index]
            iterations=args.admm_iters,
            # Optimization params
            rho=args.admm_rho,
            denoiser_weight=args.admm_denoiser_weight,
            # Denoiser params
            denoiser_weights=args.denoiser_weights,
            denoiser_device=args.denoiser_device,
            denoiser_type=args.denoiser_type,
            denoiser_sigma_scale=args.denoiser_sigma_scale,
            # NEW: control how often the denoiser runs
            denoiser_interval=args.admm_denoiser_interval if hasattr(args, "admm_denoiser_interval") else 2,
            # Scheduler
            scheduler=scheduler,
            pattern_contexts=batch.pattern_contexts,
            # MTF / Physics params
            mtf_scale=args.admm_mtf_scale,
            mtf_floor=args.admm_mtf_floor,
            mtf_weighting_mode=args.admm_mtf_weighting_mode,
            mtf_sigma_adapt=args.admm_mtf_sigma_adapt,
            # Advanced Wiener params
            mtf_wiener_alpha=args.admm_mtf_wiener_alpha,
            mtf_wiener_floor=args.admm_mtf_wiener_floor,
            mtf_wiener_tau_min=args.admm_mtf_wiener_tau_min,
            mtf_wiener_tau_max=args.admm_mtf_wiener_tau_max,
        )


    if method == "admm_diffusion":
        return run_admm_diffusion_baseline(
            batch.image,
            batch.forward_outputs["patterns"],  # type: ignore[index]
            iterations=args.admm_iters,
            rho=args.admm_rho,
            diffusion_prior_type=args.diffusion_prior_type,
            diffusion_prior_weights=args.diffusion_prior_weights,
            diffusion_steps=args.diffusion_steps,
            diffusion_guidance=args.diffusion_guidance,
            diffusion_noise_scale=args.diffusion_noise_scale,
            diffusion_sigma_min=args.diffusion_sigma_min,
            diffusion_sigma_max=args.diffusion_sigma_max,
            diffusion_schedule=args.diffusion_schedule,
            diffusion_device=args.diffusion_device,
        )

    raise ValueError(f"Unsupported method: {method}")


def main(argv=None):
    args = parse_args(argv)

    # Apply benchmark presets if requested
    if args.benchmark_preset:
        if args.benchmark_preset == "small":
            args.limit = 10
        elif args.benchmark_preset == "long":
            args.limit = 200

    summaries: List[Dict[str, object]] = []
    baseline_root: Optional[Path] = None
    trace_path: Optional[Path] = None
    trace_file = None
    config_snapshot: Optional[Dict[str, Any]] = None

    persist_outputs = (not args.collect_only) or args.save_recon
    save_recon = args.save_recon
    method = args.method.lower()
    lpips_device = _resolve_lpips_device(getattr(args, "lpips_device", None)) if getattr(args, "enable_lpips", False) else None

    try:
        for batch in run_forward_batches(args, baseline_name=method, persist_outputs=persist_outputs):
            if baseline_root is None:
                baseline_root = batch.baseline_root
                if args.log_traces:
                    trace_path = baseline_root / f"{method}_traces.jsonl"
                    config_snapshot = _serialize_args(args)

            recon_results = run_method(method, batch, args)

            for pattern, result in recon_results.items():
                # Save reconstructions
                if save_recon and batch.output_dir is not None:
                    recon_dir = batch.output_dir / method
                    save_recon_image(recon_dir, method, pattern, result.reconstruction)

                # Per-image metrics row
                row = {
                    "image": batch.image_path.name,
                    "pattern": pattern,
                    "psnr": result.psnr,
                }
                if args.enable_ssim:
                    row["ssim"] = result.ssim
                if args.enable_lpips:
                    lpips_val = lpips_distance(batch.image, result.reconstruction, device=lpips_device or "cpu")
                    result.lpips = lpips_val
                    row["lpips"] = lpips_val
                summaries.append(row)

                # Console logging
                parts = [f"PSNR: {result.psnr:.2f} dB"]
                if args.enable_ssim:
                    parts.append(f"SSIM: {result.ssim:.4f}")
                if args.enable_lpips and result.lpips is not None:
                    parts.append(f"LPIPS: {result.lpips:.4f}")
                metrics_str = " | ".join(parts)
                print(f"{batch.image_path.name} [{pattern}] {metrics_str}")

                # Trace logging (ADMM denoiser)
                if args.log_traces and result.trace:
                    if trace_file is None and trace_path is not None:
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        trace_file = trace_path.open("w")

                    if trace_file is not None:
                        metrics_payload = {"psnr": result.psnr}
                        if args.enable_ssim:
                            metrics_payload["ssim"] = result.ssim
                        if args.enable_lpips and result.lpips is not None:
                            metrics_payload["lpips"] = result.lpips

                        trace_entry = {
                            "image": batch.image_path.name,
                            "pattern": pattern,
                            "metrics": metrics_payload,
                            "args": config_snapshot,
                            "trace": result.trace,
                        }
                        trace_file.write(json.dumps(trace_entry) + "\n")
    finally:
        if trace_file is not None:
            trace_file.close()

    if not summaries or baseline_root is None:
        print("No images processed; nothing to report.")
        return

    # Sort rows by image then pattern order for cleaner CSV
    pattern_order = {p: idx for idx, p in enumerate(args.patterns)} if hasattr(args, "patterns") else {}
    summaries.sort(key=lambda row: (row["image"], pattern_order.get(row["pattern"], len(pattern_order))))

    # CSV & aggregation
    fieldnames = ["image", "pattern", "psnr"]
    if args.enable_ssim:
        fieldnames.append("ssim")
    if args.enable_lpips:
        fieldnames.append("lpips")

    csv_suffix = "_metrics.csv" if (args.enable_ssim or args.enable_lpips) else "_psnr.csv"
    csv_path = baseline_root / f"{method}{csv_suffix}"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved metrics to {csv_path}")

    avg_by_pattern: Dict[str, Dict[str, List[float]]] = {}
    for row in summaries:
        metrics = avg_by_pattern.setdefault(row["pattern"], {})
        for key in fieldnames:
            if key in ("image", "pattern"):
                continue
            metrics.setdefault(key, []).append(row[key])

    for pattern, values in avg_by_pattern.items():
        parts = []
        if "psnr" in values:
            parts.append(f"PSNR {sum(values['psnr']) / len(values['psnr']):.2f} dB")
        if "ssim" in values:
            parts.append(f"SSIM {sum(values['ssim']) / len(values['ssim']):.4f}")
        if "lpips" in values:
            parts.append(f"LPIPS {sum(values['lpips']) / len(values['lpips']):.4f}")
        print(f"Average {pattern}: " + " | ".join(parts) + f" over {len(values.get('psnr', []))} images")


if __name__ == "__main__":
    main()

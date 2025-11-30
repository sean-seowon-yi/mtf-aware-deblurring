#!/usr/bin/env python3
"""
Train a learnable unrolled ADMM/PnP model using the existing forward model to synthesize pairs.

This script keeps the plumbing simple:
- Uses DIV2K for clean images.
- Runs the forward model on-the-fly to generate blurred/noisy observations + physics context.
- Trains the torch-native UnrolledADMM module with a frozen UNet denoiser backbone.
"""
from __future__ import annotations

import argparse
import random
from typing import Any, Dict, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, Dataset

from mtf_aware_deblurring.datasets import DIV2KDataset
from mtf_aware_deblurring.pipelines.forward import run_forward_model
from mtf_aware_deblurring.reconstruction import UnrolledADMM, UnrolledADMMConfig
from mtf_aware_deblurring.denoisers.unet_denoiser import UNetDenoiserNet
from mtf_aware_deblurring.reconstruction.prior_scheduler import PhysicsContext
from mtf_aware_deblurring.utils import load_input_image


def _image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _kernel_to_tensor(kernel: np.ndarray) -> torch.Tensor:
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim == 2:
        k = k[None, ...]
    return torch.from_numpy(k)


def _context_features(ctx: PhysicsContext) -> Tuple[float, float, float, float, float]:
    metrics = ctx.quality_metrics()
    # Normalize to roughly [0, 1] to avoid exploding context heads.
    blur_norm = float(np.clip(ctx.blur_length_px / 64.0, 0.0, 1.0))
    taps_norm = float(np.clip(ctx.taps / 64.0, 0.0, 1.0))
    photon_norm = float(np.clip(ctx.photon_budget / 2000.0, 0.0, 1.0))
    optics = float(np.clip(metrics["optics"], 0.0, 1.0))
    snr = float(np.clip(metrics["snr"], 0.0, 1.0))
    return (blur_norm, taps_norm, photon_norm, optics, snr)


class ForwardModelDataset(Dataset):
    """
    Lightweight dataset that generates blurred/noisy observations with physics context on the fly.
    """

    def __init__(
        self,
        *,
        div2k_root: str,
        subset: str,
        degradation: str,
        scale: str,
        limit: Optional[int],
        target_size: int,
        image_mode: str,
        patterns: Sequence[str],
        taps: int,
        blur_length: float,
        duty_cycle: float,
        photon_budget: float,
        read_noise: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset = DIV2KDataset(
            root=div2k_root,
            subset=subset,
            degradation=degradation,
            scale=scale,
            limit=None if limit in (None, 0) else limit,
            target_size=target_size,
            image_mode=image_mode,
        )
        self.patterns = list(patterns)
        self.taps = taps
        self.blur_length = blur_length
        self.duty_cycle = duty_cycle
        self.photon_budget = photon_budget
        self.read_noise = read_noise
        self.base_seed = seed
        self.context_dim = 5

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # DIV2KDataset implements iteration but not indexing; load manually.
        path = self.dataset.files[idx]
        clean = load_input_image(
            path,
            target_size=self.dataset.target_size,
            normalize=self.dataset.normalize,
            mode=self.dataset.image_mode,
        )
        fwd = run_forward_model(
            clean,
            patterns=self.patterns,
            T=self.taps,
            blur_length_px=self.blur_length,
            duty_cycle=self.duty_cycle,
            photon_budget=self.photon_budget,
            read_noise_sigma=self.read_noise,
            random_seed=self.base_seed + idx,
            show_plots=False,
            save_arrays=False,
            save_pngs=False,
            save_figures=False,
            verbose=False,
        )
        patterns_dict = fwd["patterns"]
        pattern_name = random.choice(self.patterns)
        data = patterns_dict[pattern_name]
        ctx = data.get("context", None)
        if not isinstance(ctx, PhysicsContext):
            raise RuntimeError("PhysicsContext missing from forward model output.")
        return {
            "clean": _image_to_tensor(fwd["scene"]),
            "noisy": _image_to_tensor(data["noisy"]),
            "kernel": _kernel_to_tensor(data["kernel"]),
            "mtf": _image_to_tensor(data["mtf"]),
            "context": torch.tensor(_context_features(ctx), dtype=torch.float32),
            "pattern": pattern_name,
            "image_path": str(path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learnable unrolled ADMM model.")
    g_data = parser.add_argument_group("Data / Forward Model")
    g_data.add_argument("--div2k-root", type=str, required=True, help="Path to DIV2K root.")
    g_data.add_argument("--subset", default="train", choices=["train", "valid"])
    g_data.add_argument("--degradation", default="bicubic")
    g_data.add_argument("--scale", default="X2")
    g_data.add_argument("--limit", type=int, default=0, help="Number of images (0 = all).")
    g_data.add_argument("--target-size", type=int, default=256)
    g_data.add_argument("--image-mode", choices=["grayscale", "rgb"], default="grayscale")
    g_data.add_argument("--patterns", nargs="+", default=["box", "random", "legendre"])
    g_data.add_argument("--taps", type=int, default=31)
    g_data.add_argument("--blur-length", type=float, default=15.0)
    g_data.add_argument("--duty-cycle", type=float, default=0.5)
    g_data.add_argument("--photon-budget", type=float, default=1000.0)
    g_data.add_argument("--read-noise", type=float, default=0.01)
    g_data.add_argument("--seed", type=int, default=0)

    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=2)
    g_train.add_argument("--batch-size", type=int, default=1)
    g_train.add_argument("--steps", type=int, default=8, help="Unrolled iterations.")
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--device", default=None, help="cpu, cuda, or dml (default: auto).")
    g_train.add_argument("--denoise-every", type=int, default=1)
    g_train.add_argument("--mtf-mask", action="store_true", help="Enable learnable MTF mask.")
    g_train.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Where to write checkpoints (best and latest).",
    )
    g_train.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "step"],
        default="none",
        help="Optional LR scheduler (cosine or step).",
    )
    g_train.add_argument("--lr-step-size", type=int, default=5, help="Step size for StepLR (epochs).")
    g_train.add_argument("--lr-gamma", type=float, default=0.5, help="Decay factor for StepLR.")
    g_train.add_argument("--lr-eta-min", type=float, default=1e-5, help="Min LR for CosineAnnealingLR.")

    return parser.parse_args()


def build_model(args: argparse.Namespace, context_dim: int, channels: int) -> UnrolledADMM:
    cfg = UnrolledADMMConfig(
        steps=args.steps,
        denoise_every=args.denoise_every,
        learn_mtf_mask=bool(args.mtf_mask),
    )
    denoiser = UNetDenoiserNet(channels=channels)
    model = UnrolledADMM(
        denoiser,
        cfg,
        channels=channels,
        context_dim=context_dim,
        device=args.device,
    )
    return model


def _log_args(args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            payload[k] = str(v)
        elif isinstance(v, (list, tuple)):
            payload[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            payload[k] = v
    print("[Args]", json.dumps(payload, indent=2, sort_keys=True))


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    _log_args(args)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    dataset = ForwardModelDataset(
        div2k_root=args.div2k_root,
        subset=args.subset,
        degradation=args.degradation,
        scale=args.scale,
        limit=args.limit if args.limit > 0 else None,
        target_size=args.target_size,
        image_mode=args.image_mode,
        patterns=args.patterns,
        taps=args.taps,
        blur_length=args.blur_length,
        duty_cycle=args.duty_cycle,
        photon_budget=args.photon_budget,
        read_noise=args.read_noise,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    channels = 1 if args.image_mode == "grayscale" else 3
    model = build_model(args, dataset.context_dim, channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.lr_eta_min
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(args.lr_step_size, 1), gamma=args.lr_gamma
        )

    best_loss = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            optimizer.zero_grad()
            noisy = batch["noisy"].to(device)
            kernel = batch["kernel"].to(device)
            mtf = batch["mtf"].to(device)
            context = batch["context"].to(device)
            clean = batch["clean"].to(device)

            recon, _ = model(noisy, kernel, mtf=mtf, context=context, return_trace=False)
            loss = F.l1_loss(recon, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if scheduler is not None:
            scheduler.step()
        epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"[Epoch {epoch + 1}] L1 loss: {epoch_loss:.4f}")

        # Always save the latest; also track the best-by-loss checkpoint.
        latest_path = ckpt_dir / "unrolled_latest.pt"
        _save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            args=args,
            loss=epoch_loss,
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = ckpt_dir / "unrolled_best.pt"
            _save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                args=args,
                loss=epoch_loss,
            )


if __name__ == "__main__":
    main()

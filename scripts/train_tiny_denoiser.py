from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyDenoiser(nn.Module):
    """Shallow residual CNN denoiser similar to a lite DnCNN."""

    def __init__(self, channels: int = 3, features: int = 64, depth: int = 8) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Predict residual noise and subtract it (DnCNN-style).
        return torch.clamp(x - self.body(x), 0.0, 1.0)


class PatchDataset(Dataset):
    """Preloads a set of clean patches and synthesizes noisy counterparts."""

    def __init__(
        self,
        root: Path,
        *,
        max_images: int,
        patches_per_image: int,
        patch_size: int,
        sigma: float,
    ) -> None:
        self.patch_size = patch_size
        self.sigma = sigma
        paths = sorted(root.glob("*.png"))[:max_images]
        if not paths:
            raise FileNotFoundError(f"No PNGs found under {root}")
        patches = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            h, w, _ = arr.shape
            if h < patch_size or w < patch_size:
                raise ValueError(f"Image {path} is smaller than patch size {patch_size}")
            for _ in range(patches_per_image):
                y0 = random.randint(0, h - patch_size)
                x0 = random.randint(0, w - patch_size)
                patch = arr[y0 : y0 + patch_size, x0 : x0 + patch_size]
                patches.append(patch.transpose(2, 0, 1))
        self.clean = torch.from_numpy(np.stack(patches, axis=0))  # (N, C, H, W)

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, idx: int):
        clean = self.clean[idx]
        noise = torch.randn_like(clean) * self.sigma
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean


def train(args: argparse.Namespace) -> None:
    root = Path(args.div2k_root)
    dataset = PatchDataset(
        root,
        max_images=args.max_images,
        patches_per_image=args.patches_per_image,
        patch_size=args.patch_size,
        sigma=args.noise_sigma / 255.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    model = TinyDenoiser(depth=args.depth, features=args.features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"Training on {len(dataset)} patches ({steps_per_epoch} steps/epoch) for {args.epochs} epochs.")

    step = 0
    for epoch in range(args.epochs):
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            denoised = model(noisy)
            loss = criterion(denoised, clean)
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.log_interval == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} - Loss: {loss.item():.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "noise_sigma": args.noise_sigma}, output)
    print(f"Saved weights to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny residual denoiser on DIV2K patches.")
    parser.add_argument("--div2k-root", type=Path, default=Path("data") / "DIV2K_train_LR_bicubic" / "X2")
    parser.add_argument("--output", type=Path, default=Path("src") / "mtf_aware_deblurring" / "assets" / "tiny_denoiser_sigma15.pth")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--patches-per-image", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--noise-sigma", type=float, default=15.0, help="Noise sigma in pixel space (0-255).")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

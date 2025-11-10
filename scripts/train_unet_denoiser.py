from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mtf_aware_deblurring.denoisers.unet_denoiser import UNetDenoiserNet, default_unet_denoiser_weights
from mtf_aware_deblurring.noise import add_poisson_gaussian
from mtf_aware_deblurring.torch_utils import resolve_device


class PatchDataset(Dataset):
    """Loads patches from DIV2K images and synthesizes Poisson-Gaussian noise."""

    def __init__(
        self,
        root: Path,
        *,
        max_images: int,
        patches_per_image: int,
        patch_size: int,
        photon_budget: float,
        read_noise: float,
        seed: int = 0,
    ) -> None:
        self.patch_size = patch_size
        self.photon_budget = photon_budget
        self.read_noise = read_noise
        paths = sorted(root.glob("*.png"))[:max_images]
        if not paths:
            raise FileNotFoundError(f"No PNGs found under {root}")
        rng = np.random.default_rng(seed)
        patches: List[np.ndarray] = []
        for path in paths:
            img = Image.open(path).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            h, w = arr.shape
            if h < patch_size or w < patch_size:
                continue
            for _ in range(patches_per_image):
                y0 = rng.integers(0, h - patch_size + 1)
                x0 = rng.integers(0, w - patch_size + 1)
                patch = arr[y0 : y0 + patch_size, x0 : x0 + patch_size]
                patches.append(patch[None, ...])  # keep channel dim
        if not patches:
            raise RuntimeError("Failed to extract any patches. Adjust patch_size or source data.")
        self.clean = torch.from_numpy(np.stack(patches, axis=0)).float()

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, idx: int):
        clean = self.clean[idx].numpy()
        noisy = add_poisson_gaussian(
            clean,
            photon_budget=self.photon_budget,
            read_noise_sigma=self.read_noise,
        )
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean).float()


def train(args: argparse.Namespace) -> None:
    dataset = PatchDataset(
        args.dataset_root,
        max_images=args.max_images,
        patches_per_image=args.patches_per_image,
        patch_size=args.patch_size,
        photon_budget=args.photon_budget,
        read_noise=args.read_noise,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = resolve_device(args.device)
    model = UNetDenoiserNet(channels=1, base_features=args.base_features).to(device)
    if args.init_weights and Path(args.init_weights).exists():
        state = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    steps_per_epoch = len(loader)
    print(f"Training UNet denoiser on {len(dataset)} patches ({steps_per_epoch} steps/epoch) for {args.epochs} epochs.")

    global_step = 0
    for epoch in range(args.epochs):
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            denoised = model(noisy)
            loss = criterion(denoised, clean)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Step {global_step}: loss={loss.item():.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "photon_budget": args.photon_budget,
            "read_noise": args.read_noise,
        },
        output,
    )
    print(f"Saved UNet denoiser weights to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UNet denoiser with Poisson-Gaussian noise.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "DIV2K_train_LR_bicubic" / "X2")
    parser.add_argument("--output", type=Path, default=default_unet_denoiser_weights())
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--patches-per-image", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--photon-budget", type=float, default=800.0)
    parser.add_argument("--read-noise", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda", "dml"], default="cuda" if torch and torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--init-weights", type=Path, help="Optional checkpoint to warm-start from.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

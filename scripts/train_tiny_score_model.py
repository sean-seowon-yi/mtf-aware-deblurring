from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mtf_aware_deblurring.diffusion.score_model import (
    TinyScoreUNet,
    DEFAULT_TINY_SCORE_WEIGHTS,
)
from mtf_aware_deblurring.torch_utils import resolve_device


class ScorePatchDataset(Dataset):
    """Loads clean grayscale patches and synthesizes noisy counterparts for score matching."""

    def __init__(
        self,
        root: Path,
        *,
        max_images: int,
        patches_per_image: int,
        patch_size: int,
        sigma_min: float,
        sigma_max: float,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        paths = sorted(root.glob("*.png"))[:max_images]
        if not paths:
            raise FileNotFoundError(f"No PNGs found under {root}")
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
                patches.append(arr[y0 : y0 + patch_size, x0 : x0 + patch_size][None, ...])
        if not patches:
            raise RuntimeError("Failed to extract any patches. Adjust patch_size or dataset-root.")
        self.clean = torch.from_numpy(np.stack(patches, axis=0)).float()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, idx: int):
        clean = self.clean[idx]
        sigma = torch.rand(1) * (self.sigma_max - self.sigma_min) + self.sigma_min
        noise = torch.randn_like(clean)
        noisy = clean + sigma * noise
        target_score = -(noisy - clean) / (sigma ** 2)
        return noisy, sigma.squeeze(0), target_score


def train(args: argparse.Namespace) -> None:
    dataset = ScorePatchDataset(
        args.dataset_root,
        max_images=args.max_images,
        patches_per_image=args.patches_per_image,
        patch_size=args.patch_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = resolve_device(args.device)
    model = TinyScoreUNet(in_channels=1, base_channels=args.base_channels, depth=args.depth).to(device)

    if args.init_weights and Path(args.init_weights).exists():
        checkpoint = torch.load(args.init_weights, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    steps_per_epoch = len(loader)
    print(f"Training TinyScoreUNet on {len(dataset)} patches ({steps_per_epoch} steps/epoch) for {args.epochs} epochs.")

    global_step = 0
    for epoch in range(args.epochs):
        for noisy, sigma, target in loader:
            noisy = noisy.to(device)
            sigma = sigma.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(noisy, sigma)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Step {global_step}: loss={loss.item():.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output)
    print(f"Saved TinyScoreUNet weights to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TinyScoreUNet diffusion prior via denoising score matching.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "DIV2K_train_LR_bicubic" / "X2")
    parser.add_argument("--output", type=Path, default=DEFAULT_TINY_SCORE_WEIGHTS)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--patches-per-image", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda", "dml"], default="cuda" if torch and torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--init-weights", type=Path, help="Optional checkpoint for warm start.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from ..torch_utils import resolve_device

DNCCN_MAT_URL = "https://github.com/cszn/DnCNN/raw/master/model/DnCNN3.mat"
DNCCN_MAT_SHA256 = "3962377d522b6728417e364bf7a589cf34ddfb592af0bd9769e53f8123d17677"


def default_dncnn_weights() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "dncnn_sigma15.pth"


def _download_file(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out_file:
        out_file.write(response.read())
    sha = hashlib.sha256()
    with open(tmp_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch when downloading {url}. "
            f"Expected {expected_sha256} but received {digest}."
        )
    tmp_path.replace(target)


def _load_mat_layers(mat_path: Path):
    import h5py  # lazy import to avoid forcing the dependency when unused

    with h5py.File(mat_path, "r") as f:
        layers = f["net"]["layers"]
        conv_weights: list[np.ndarray] = []
        conv_biases: list[np.ndarray] = []
        for idx in range(layers.shape[0]):
            layer = f[layers[idx, 0]]
            layer_type = "".join(chr(c[0]) for c in layer["type"]).strip().lower()
            if layer_type != "conv":
                continue
            w = np.array(f[layer["weights"][0, 0]], dtype=np.float32)
            b = np.array(f[layer["weights"][1, 0]], dtype=np.float32)
            if w.ndim == 3:  # final conv squeezes the singleton output dim
                w = np.expand_dims(w, axis=0)
            if b.ndim > 1:
                b = b.reshape(-1)
            conv_weights.append(w)
            conv_biases.append(b)
    return conv_weights, conv_biases


def _convert_mat_to_state_dict(mat_path: Path) -> dict:
    if torch is None:
        raise ImportError("PyTorch is required to convert DnCNN weights. Install torch>=2.0.")
    conv_weights, conv_biases = _load_mat_layers(mat_path)
    depth = len(conv_weights)
    model = DnCNNDenoiserNet(channels=1, depth=depth, use_batchnorm=False)
    conv_modules = [m for m in model.body if isinstance(m, nn.Conv2d)]
    if len(conv_modules) != depth:
        raise RuntimeError("Mismatch between parsed DnCNN layers and model definition.")

    with torch.no_grad():
        for conv, w_np, b_np in zip(conv_modules, conv_weights, conv_biases):
            weight = torch.from_numpy(w_np)
            if weight.shape[1] != conv.in_channels:
                raise RuntimeError(
                    f"Unexpected in_channels for DnCNN conv: got {weight.shape[1]}, expected {conv.in_channels}."
                )
            if weight.shape[0] != conv.out_channels:
                if weight.shape[0] == 1 and conv.out_channels > 1:
                    weight = weight.repeat(conv.out_channels, 1, 1, 1)
                else:
                    raise RuntimeError(
                        f"Unexpected out_channels for DnCNN conv: got {weight.shape[0]}, expected {conv.out_channels}."
                    )
            conv.weight.copy_(weight)

            bias = torch.from_numpy(b_np.astype(np.float32))
            if bias.shape[0] != conv.out_channels:
                if bias.shape[0] == 1 and conv.out_channels > 1:
                    bias = bias.repeat(conv.out_channels)
                else:
                    raise RuntimeError(
                        f"Unexpected bias shape for DnCNN conv: got {bias.shape[0]}, expected {conv.out_channels}."
                    )
            conv.bias.copy_(bias)

    return model.state_dict()


def _prepare_dncnn_weights(pth_path: Path) -> Path:
    if pth_path.exists():
        return pth_path
    pth_path.parent.mkdir(parents=True, exist_ok=True)
    mat_path = pth_path.with_suffix(".mat")
    if not mat_path.exists():
        _download_file(DNCCN_MAT_URL, mat_path, DNCCN_MAT_SHA256)
    state_dict = _convert_mat_to_state_dict(mat_path)
    torch.save(state_dict, pth_path)
    return pth_path


class DnCNNDenoiserNet(nn.Module if nn is not None else object):
    """DnCNN-style residual denoiser (default depth 20, no batch-norm)."""

    def __init__(
        self,
        channels: int = 1,
        features: int = 64,
        depth: int = 20,
        use_batchnorm: bool = False,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to instantiate DnCNNDenoiserNet.")
        super().__init__()
        if depth < 2:
            raise ValueError("DnCNN depth must be at least 2.")

        layers: list[nn.Module] = [
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        if torch is None:
            raise ImportError("PyTorch is required to run DnCNNDenoiserNet.")
        return torch.clamp(x - self.body(x), 0.0, 1.0)


class DnCNNWrapper:
    """NumPy-friendly wrapper for the DnCNN denoiser."""

    def __init__(self, model: DnCNNDenoiserNet, *, device: Optional[str] = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use DnCNNWrapper.")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if torch is None:
            raise ImportError("PyTorch is required to use DnCNNWrapper.")
        arr = np.asarray(image, dtype=np.float32)
        squeeze_channel = False
        if arr.ndim == 2:
            arr = arr[..., None]
            squeeze_channel = True
        channels = arr.shape[2]
        arr_ch_first = arr.transpose(2, 0, 1)  # (C, H, W)
        tensor = torch.from_numpy(arr_ch_first[:, None, ...]).to(self.device)  # (C,1,H,W)
        with torch.inference_mode():
            denoised = self.model(tensor).squeeze(1)  # (C,H,W)
        out = denoised.cpu().numpy().transpose(1, 2, 0)
        if squeeze_channel:
            out = out[..., 0]
        return np.clip(out.astype(np.float32), 0.0, 1.0)


def build_dncnn_denoiser(
    *,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> DnCNNWrapper:
    if torch is None:
        raise ImportError("PyTorch is required to build the DnCNN denoiser. Install torch>=2.0.")
    path = Path(weights_path) if weights_path is not None else default_dncnn_weights()
    path = _prepare_dncnn_weights(path)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model = DnCNNDenoiserNet(
        channels=1,
        depth=state_dict_count_convs(state_dict),
        use_batchnorm=False,
    )
    model.load_state_dict(state_dict, strict=True)
    return DnCNNWrapper(model, device=device)


def state_dict_count_convs(state_dict: dict) -> int:
    # Count how many conv layers are present based on weight tensors
    return sum(1 for key in state_dict if key.endswith(".weight") and state_dict[key].ndim == 4)


__all__ = [
    "DnCNNDenoiserNet",
    "DnCNNWrapper",
    "build_dncnn_denoiser",
    "default_dncnn_weights",
]

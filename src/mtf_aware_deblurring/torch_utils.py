from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    import torch_directml
except ImportError:  # pragma: no cover
    torch_directml = None


def resolve_device(name: Optional[str] = None):
    """
    Resolve a torch device from a user-provided string, supporting CPU, CUDA, and DirectML.

    Parameters
    ----------
    name:
        'cpu', 'cuda', 'dml', or None. When None we prefer CUDA, then CPU.
    """
    if torch is None:
        raise ImportError("PyTorch is required to resolve torch devices.")
    if name is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    normalized = name.lower()
    if normalized in ("cpu", "cuda"):
        return torch.device(normalized)
    if normalized in ("dml", "directml"):
        if torch_directml is None:
            raise RuntimeError(
                "torch-directml is not installed; install it via `pip install torch-directml` "
                "to use the DirectML backend on AMD GPUs."
            )
        return torch_directml.device()
    raise ValueError(f"Unsupported device '{name}'. Expected one of: cpu, cuda, dml.")


__all__ = ["resolve_device"]

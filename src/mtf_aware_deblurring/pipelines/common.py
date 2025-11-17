from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

from ..datasets import DIV2KDataset
from ..reconstruction.prior_scheduler import PhysicsContext
from .forward import ensure_div2k_available, run_forward_model
from ..utils import default_output_dir


@dataclass
class ForwardBatch:
    """Container for per-image forward model outputs used by reconstruction pipelines."""

    index: int
    image_path: Path
    image: np.ndarray
    forward_outputs: Dict[str, object]
    output_dir: Optional[Path]
    baseline_root: Path
    pattern_contexts: Dict[str, PhysicsContext] = field(default_factory=dict)


def _resolve_limit(value: Optional[int]) -> Optional[int]:
    if value in (None, 0):
        return None
    return max(int(value), 0)


def run_forward_batches(
    args,
    baseline_name: str,
    *,
    persist_outputs: bool = True,
) -> Iterator[ForwardBatch]:
    """
    Shared iterator that runs the forward model over a dataset and yields results
    for downstream reconstruction pipelines (Wiener, ADMM, etc.).
    """

    ensure_args = SimpleNamespace(
        div2k_root=args.div2k_root,
        subset=args.subset,
        degradation=args.degradation,
        scale=args.scale,
        auto_download=getattr(args, "auto_download", False),
    )
    ensure_div2k_available(ensure_args)

    dataset = DIV2KDataset(
        root=args.div2k_root,
        subset=args.subset,
        degradation=args.degradation,
        scale=args.scale,
        limit=_resolve_limit(getattr(args, "limit", None)),
        target_size=getattr(args, "target_size", 256),
        image_mode=getattr(args, "image_mode", "grayscale"),
    )

    output_override = getattr(args, "output_dir", None)
    if output_override is None:
        baseline_root = default_output_dir() / "reconstruction" / baseline_name
    else:
        baseline_root = Path(output_override)
    baseline_root.mkdir(parents=True, exist_ok=True)

    save_arrays = getattr(args, "save_arrays", False) if persist_outputs else False
    save_pngs = getattr(args, "save_pngs", False) if persist_outputs else False
    save_figures = getattr(args, "save_figures", False) if persist_outputs else False

    patterns: Sequence[str] = getattr(args, "patterns", ("box", "random", "legendre"))

    for idx, (path, image) in enumerate(dataset):
        if persist_outputs:
            relative = baseline_root / path.stem
            relative.mkdir(parents=True, exist_ok=True)
            output_dir = relative
        else:
            relative = baseline_root
            output_dir = None

        forward_outputs = run_forward_model(
            image,
            patterns=patterns,
            selected_pattern=getattr(args, "selected_pattern", None),
            T=getattr(args, "taps", 31),
            blur_length_px=getattr(args, "blur_length", 15.0),
            duty_cycle=getattr(args, "duty_cycle", 0.5),
            random_seed=getattr(args, "seed", 0) + idx,
            photon_budget=getattr(args, "photon_budget", 1000.0),
            read_noise_sigma=getattr(args, "read_noise", 0.01),
            show_plots=False,
            save_arrays=save_arrays,
            save_pngs=save_pngs,
            save_figures=save_figures,
            output_dir=output_dir,
            verbose=False,
        )

        contexts: Dict[str, PhysicsContext] = {}
        patterns_dict = forward_outputs.get("patterns", {})  # type: ignore[assignment]
        if isinstance(patterns_dict, dict):
            for name, data in patterns_dict.items():
                if isinstance(data, dict):
                    ctx = data.get("context")
                    if isinstance(ctx, PhysicsContext):
                        contexts[name] = ctx

        yield ForwardBatch(
            index=idx,
            image_path=path,
            image=image,
            forward_outputs=forward_outputs,
            output_dir=output_dir,
            baseline_root=baseline_root,
            pattern_contexts=contexts,
        )


__all__ = ["ForwardBatch", "run_forward_batches"]

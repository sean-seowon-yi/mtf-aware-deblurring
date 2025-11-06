from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .metrics import spectral_snr
from .noise import add_poisson_gaussian
from .optics import fft_convolve2d, kernel2d_from_psf1d, motion_psf_from_code, mtf_from_kernel
from .patterns import make_exposure_code, resolve_legendre_prime
from .synthetic import SyntheticData
from .utils import axes_as_list, configure_matplotlib_defaults, default_output_dir, finalize_figure

DEFAULT_SEED = 0


class ForwardModelRunner:
    """Run and visualize the forward imaging model for one or more exposure patterns."""

    def __init__(
        self,
        img: np.ndarray,
        patterns: Sequence[str] = ("box", "random", "legendre"),
        *,
        selected_pattern: Optional[str] = None,
        T: int = 31,
        blur_length_px: float = 15.0,
        duty_cycle: float = 0.5,
        random_seed: int = 0,
        photon_budget: float = 1000.0,
        read_noise_sigma: float = 0.01,
        show_plots: bool = True,
        save_arrays: bool = False,
        save_pngs: bool = False,
        save_figures: bool = False,
        output_dir: Optional[Path] = None,
        legendre_params: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.scene = img

        if selected_pattern is not None:
            self.patterns = (selected_pattern,)
        else:
            self.patterns = tuple(patterns)

        self.T = int(T)
        self.blur_length_px = float(blur_length_px)
        self.duty_cycle = float(duty_cycle)
        self.random_seed = int(random_seed)
        self.photon_budget = float(photon_budget)
        self.read_noise_sigma = float(read_noise_sigma)
        self.show_plots = bool(show_plots)

        self.legendre_params = dict(legendre_params or {})

        self.save_arrays = bool(save_arrays)
        self.save_pngs = bool(save_pngs)
        self.save_figures = bool(save_figures)

        self._output_dir_req = output_dir
        self.output_dir: Optional[Path] = None
        self.arrays_dir: Optional[Path] = None
        self.figures_dir: Optional[Path] = None

        self.results: Dict[str, Dict[str, Any]] = {}

        self.verbose = bool(verbose)
        self._logger = logger or (lambda s: print(s))
        self.warnings: list[str] = []

    # ---------- Utility: conditional logging ----------

    def _log(self, msg: str) -> None:
        if self.verbose:
            self._logger(msg)

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self._log(f"Warning: {msg}")

    # ---------- Public API ----------

    def run(self) -> Dict[str, Any]:
        if self.save_arrays or self.save_pngs or self.save_figures:
            self.output_dir, self.arrays_dir, self.figures_dir = self._prepare_output_dirs(self._output_dir_req)

        self._plot_scene()

        for idx, pattern in enumerate(self.patterns):
            self.results[pattern] = self._process_pattern(pattern, idx)

        self._plot_psf_1d()
        self._plot_blurred_images()
        self._plot_mtf_images()
        self._plot_mtf_slice()
        self._plot_noisy_images()
        self._plot_ssnr_images()

        if self.save_arrays:
            self._save_arrays()
        if self.save_pngs:
            self._save_pngs()

        if self.save_arrays or self.save_pngs or self.save_figures:
            self._log(f"Saved to: {self.output_dir}")

        return {
            "scene": self.scene,
            "patterns": self.results,
            "output_dir": self.output_dir,
            "arrays_dir": self.arrays_dir,
            "figures_dir": self.figures_dir,
            "warnings": list(self.warnings),
        }

    # ---------- Directory / I/O ----------

    def _prepare_output_dirs(self, output_dir: Optional[Path]) -> Tuple[Path, Path, Path]:
        if output_dir is None:
            output_dir = default_output_dir()
        output_dir = Path(output_dir)
        arrays_dir = output_dir / "arrays"
        figures_dir = output_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, arrays_dir, figures_dir

    def _save_arrays(self) -> None:
        assert self.arrays_dir is not None
        np.save(self.arrays_dir / "scene.npy", self.scene)
        for pattern, data in self.results.items():
            safe = pattern.lower().replace(" ", "_")
            np.save(self.arrays_dir / f"code_{safe}.npy", data["code"])
            np.save(self.arrays_dir / f"psf_{safe}.npy", data["psf"])
            np.save(self.arrays_dir / f"k_{safe}.npy", data["kernel"])
            np.save(self.arrays_dir / f"y_{safe}.npy", data["blurred"])
            np.save(self.arrays_dir / f"y_{safe}_noisy.npy", data["noisy"])
            np.save(self.arrays_dir / f"mtf_{safe}.npy", data["mtf"])
            np.save(self.arrays_dir / f"ssnr_{safe}.npy", data["ssnr"])

    def _save_pngs(self) -> None:
        assert self.output_dir is not None
        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError:
            self._warn("imageio is not available; skipping PNG exports.")
            return

        scene_png = np.clip(self.scene * 255, 0, 255).astype(np.uint8)
        imageio.imwrite(self.output_dir / "scene.png", scene_png)
        for pattern, data in self.results.items():
            safe = pattern.lower().replace(" ", "_")
            png = np.clip(data["noisy"] * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(self.output_dir / f"y_{safe}_noisy.png", png)

    # ---------- Core per-pattern pipeline ----------

    def _process_pattern(self, pattern: str, idx: int) -> Dict[str, Any]:
        label = pattern.replace("_", " ").title()
        pattern_seed = self.random_seed + idx

        seed_arg: Optional[int] = pattern_seed
        current_legendre_params: Optional[Dict[str, Any]] = None
        if pattern == "legendre":
            pattern_seed = self.random_seed
            current_legendre_params = dict(self.legendre_params)
            if "prime" not in current_legendre_params:
                current_legendre_params["prime"] = resolve_legendre_prime(self.T, None)
            if "rotation" not in current_legendre_params:
                prime_used = int(current_legendre_params["prime"])
                current_legendre_params["rotation"] = int(pattern_seed) % prime_used
            seed_arg = None

        code = self._make_code(pattern, seed_arg, current_legendre_params)
        psf = motion_psf_from_code(code, length_px=self.blur_length_px)
        kernel = kernel2d_from_psf1d(psf)
        blurred = fft_convolve2d(self.scene, kernel)
        noisy = add_poisson_gaussian(
            blurred,
            photon_budget=self.photon_budget,
            read_noise_sigma=self.read_noise_sigma,
            rng_seed=self.random_seed + idx,
        )
        mtf = mtf_from_kernel(kernel, self.scene.shape)
        ssnr = spectral_snr(blurred, noisy)

        metadata: Dict[str, Any] = {"pattern": pattern}
        if current_legendre_params is not None:
            metadata.update(current_legendre_params)

        self._print_pattern_summary(label, code, current_legendre_params)

        return {
            "label": label,
            "code": code,
            "psf": psf,
            "kernel": kernel,
            "blurred": blurred,
            "noisy": noisy,
            "mtf": mtf,
            "ssnr": ssnr,
            "duty_cycle": float(code.mean()),
            "metadata": metadata,
        }

    def _make_code(
        self,
        pattern: str,
        seed_arg: Optional[int],
        legendre_params: Optional[Mapping[str, Any]],
    ) -> np.ndarray:
        return make_exposure_code(
            self.T,
            pattern=pattern,  # type: ignore[arg-type]
            seed=seed_arg,
            duty_cycle=self.duty_cycle,
            legendre_params=legendre_params,
        )

    def _print_pattern_summary(
        self, label: str, code: np.ndarray, legendre_params: Optional[Mapping[str, Any]]
    ) -> None:
        if not self.verbose:
            return
        self._log(f"{label} code (T={self.T}): {code.astype(int)}")
        self._log(f"  Duty cycle: {code.mean():.3f}")
        if legendre_params is not None:
            prime_info = int(legendre_params["prime"])
            rotation_info = int(legendre_params.get("rotation", 0))
            mode_info = legendre_params.get("append_mode", "auto")
            flip_info = bool(legendre_params.get("flip", False))
            self._log(
                f"  Prime: {prime_info}, rotation: {rotation_info}, "
                f"append_mode: {mode_info}, flip: {flip_info}"
            )

    # ---------- Plotting ----------

    def _plot_scene(self) -> None:
        fig, ax = plt.subplots()
        ax.imshow(self.scene, cmap="gray")
        ax.set_title("Synthetic Scene")
        ax.axis("off")
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "scene.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _plot_psf_1d(self) -> None:
        fig, ax = plt.subplots()
        for pattern in self.patterns:
            data = self.results[pattern]
            ax.plot(data["psf"], label=data["label"])
        ax.set_title("1D PSF (horizontal)")
        ax.legend()
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "psf_1d.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _plot_blurred_images(self) -> None:
        fig, axs = plt.subplots(1, len(self.patterns) + 1, figsize=(4 * (len(self.patterns) + 1), 3))
        axes = axes_as_list(axs)
        axes[0].imshow(self.scene, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        for ax, pattern in zip(axes[1:], self.patterns):
            data = self.results[pattern]
            ax.imshow(data["blurred"], cmap="gray")
            ax.set_title(data["label"])
            ax.axis("off")
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "blurred.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _plot_mtf_images(self) -> None:
        gamma = 0.5
        ncols = max(len(self.patterns), 1)
        fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 3))
        axes = axes_as_list(axs)
        for ax, pattern in zip(axes, self.patterns):
            data = self.results[pattern]
            ax.imshow(data["mtf"] ** gamma, cmap="gray")
            ax.set_title(f"MTF ({data['label']})")
            ax.axis("off")
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "mtf.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _plot_mtf_slice(self) -> None:
        fig, ax = plt.subplots()
        mid = self.scene.shape[0] // 2
        for pattern in self.patterns:
            data = self.results[pattern]
            ax.plot(data["mtf"][mid], label=data["label"])
        ax.set_title("MTF horizontal slice (normalized)")
        ax.set_xlabel("Spatial frequency index")
        ax.set_ylabel("Magnitude")
        ax.legend()
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "mtf_slice.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _plot_noisy_images(self) -> None:
        fig, axs = plt.subplots(1, len(self.patterns) + 1, figsize=(4 * (len(self.patterns) + 1), 3))
        axes = axes_as_list(axs)
        axes[0].imshow(self.scene, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        for ax, pattern in zip(axes[1:], self.patterns):
            data = self.results[pattern]
            ax.imshow(data["noisy"], cmap="gray")
            ax.set_title(f"{data['label']} + noise")
            ax.axis("off")
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "noisy.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    def _plot_ssnr_images(self) -> None:
        ncols = max(len(self.patterns), 1)
        fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 3))
        axes = axes_as_list(axs)
        for ax, pattern in zip(axes, self.patterns):
            data = self.results[pattern]
            ax.imshow(data["ssnr"], cmap="gray")
            ax.set_title(f"Spectral SNR ({data['label']})")
            ax.axis("off")
        if self.save_figures and self.figures_dir is not None:
            finalize_figure(fig, self.figures_dir / "spectral_snr.png", self.show_plots)
        else:
            if self.show_plots:
                plt.show()
            plt.close(fig)

    # ---------- Helpers ----------

    def _print_pattern_summary(
        self, label: str, code: np.ndarray, legendre_params: Optional[Mapping[str, Any]]
    ) -> None:
        if not self.verbose:
            return
        self._log(f"{label} code (T={self.T}): {code.astype(int)}")
        self._log(f"  Duty cycle: {code.mean():.3f}")
        if legendre_params is not None:
            prime_info = int(legendre_params["prime"])
            rotation_info = int(legendre_params.get("rotation", 0))
            mode_info = legendre_params.get("append_mode", "auto")
            flip_info = bool(legendre_params.get("flip", False))
            self._log(
                f"  Prime: {prime_info}, rotation: {rotation_info}, "
                f"append_mode: {mode_info}, flip: {flip_info}"
            )


def run_forward_model(
    img: np.ndarray,
    patterns: Sequence[str] = ("box", "random", "legendre"),
    T: int = 31,
    blur_length_px: float = 15.0,
    duty_cycle: float = 0.5,
    random_seed: int = 0,
    photon_budget: float = 1000.0,
    read_noise_sigma: float = 0.01,
    show_plots: bool = True,
    output_dir: Optional[Path] = None,
    legendre_params: Optional[Mapping[str, Any]] = None,
    *,
    selected_pattern: Optional[str] = None,
    save_arrays: bool = False,
    save_pngs: bool = False,
    save_figures: bool = False,
    verbose: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    configure_matplotlib_defaults()
    img_array = np.asarray(img, dtype=np.float32)

    runner = ForwardModelRunner(
        img=img_array,
        patterns=patterns,
        selected_pattern=selected_pattern,
        T=T,
        blur_length_px=blur_length_px,
        duty_cycle=duty_cycle,
        random_seed=random_seed,
        photon_budget=photon_budget,
        read_noise_sigma=read_noise_sigma,
        show_plots=show_plots,
        save_arrays=save_arrays,
        save_pngs=save_pngs,
        save_figures=save_figures,
        output_dir=output_dir,
        legendre_params=legendre_params,
        verbose=verbose,
        logger=logger,
    )
    return runner.run()


def main() -> None:
    checkerboard_scene = SyntheticData("Checker Board").create_img(DEFAULT_SEED)
    run_forward_model(
        checkerboard_scene,
        patterns=["box", "random", "legendre"],
        show_plots=True,
        verbose=False,
    )

    ring_scene = SyntheticData("Rings").create_img(DEFAULT_SEED)
    run_forward_model(
        ring_scene,
        patterns=["box", "random", "legendre"],
        show_plots=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()

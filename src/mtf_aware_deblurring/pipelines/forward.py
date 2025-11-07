from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np

from ..datasets import DIV2KDataset
from ..metrics import spectral_snr
from ..noise import add_poisson_gaussian
from ..optics import fft_convolve2d, kernel2d_from_psf1d, motion_psf_from_code, mtf_from_kernel
from ..patterns import make_exposure_code, resolve_legendre_prime
from ..synthetic import SyntheticData
from ..utils import axes_as_list, configure_matplotlib_defaults, default_output_dir, finalize_figure

DEFAULT_SEED = 0
DIV2K_BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K"


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
        image_shape = self.scene.shape[:2]
        mtf = mtf_from_kernel(kernel, image_shape)
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
        self._display_image(ax, self.scene, "Input Scene")
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
        self._display_image(axes[0], self.scene, "Original")
        for ax, pattern in zip(axes[1:], self.patterns):
            data = self.results[pattern]
            self._display_image(ax, data["blurred"], data["label"])
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
        self._display_image(axes[0], self.scene, "Original")
        for ax, pattern in zip(axes[1:], self.patterns):
            data = self.results[pattern]
            self._display_image(ax, data["noisy"], f"{data['label']} + noise")
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
            ssnr_img = data["ssnr"]
            if ssnr_img.ndim == 3:
                ssnr_img = np.mean(ssnr_img, axis=-1)
            ax.imshow(ssnr_img, cmap="gray")
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

    def _display_image(self, ax, image: np.ndarray, title: str) -> None:
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title)
        ax.axis("off")


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MTF-aware forward model on synthetic scenes or DIV2K frames."
    )
    parser.add_argument(
        "--div2k-root",
        type=Path,
        help="Path to the DIV2K dataset root (expects DIV2K_*_LR_<method>/ directories).",
    )
    parser.add_argument(
        "--subset",
        default="train",
        choices=["train", "valid"],
        help="DIV2K subset to use when --div2k-root is provided.",
    )
    parser.add_argument(
        "--degradation",
        default="bicubic",
        help="DIV2K degradation label (e.g., bicubic, unknown).",
    )
    parser.add_argument(
        "--scale",
        default="X2",
        help="DIV2K scale folder (X2, X3, X4, ...).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of DIV2K frames to process.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Optional square resize applied to DIV2K frames before simulation.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["grayscale", "rgb"],
        default="grayscale",
        help="Color mode for loaded DIV2K frames.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download the requested DIV2K subset if it is missing.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["box", "random", "legendre"],
        help="Exposure patterns to evaluate.",
    )
    parser.add_argument(
        "--selected-pattern",
        help="If provided, only this pattern will be simulated.",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=31,
        help="Number of shutter taps (T) for exposure patterns.",
    )
    parser.add_argument(
        "--blur-length",
        type=float,
        default=15.0,
        help="Total motion blur length in pixels.",
    )
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=0.5,
        help="Duty cycle used for random codes.",
    )
    parser.add_argument(
        "--photon-budget",
        type=float,
        default=1000.0,
        help="Photon budget for Poisson noise simulation.",
    )
    parser.add_argument(
        "--read-noise",
        type=float,
        default=0.01,
        help="Read-noise standard deviation in [0,1] range.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed for pattern generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for outputs when saving arrays/figures.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Persist Matplotlib figures to disk.",
    )
    parser.add_argument(
        "--save-pngs",
        action="store_true",
        help="Persist PNG renderings of noisy observations.",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Persist NumPy arrays (scene, PSF, MTF, etc.).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display figures interactively (may be slow for many images).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about each pattern.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also run the synthetic checkerboard/rings demo when using DIV2K.",
    )
    return parser.parse_args(argv)


def expected_div2k_dir(args: argparse.Namespace) -> Path:
    return Path(args.div2k_root) / f"DIV2K_{args.subset}_LR_{args.degradation}" / args.scale.upper()


def download_div2k_subset(args: argparse.Namespace, target_dir: Path) -> None:
    root = Path(args.div2k_root)
    root.mkdir(parents=True, exist_ok=True)
    zip_name = f"DIV2K_{args.subset}_LR_{args.degradation}_{args.scale.upper()}.zip"
    url = f"{DIV2K_BASE_URL}/{zip_name}"
    zip_path = root / zip_name

    if zip_path.exists():
        print(f"Using existing archive: {zip_path}")
    else:
        print(f"Downloading {zip_name} from {url} ...")
        try:
            with urlopen(url) as resp, open(zip_path, "wb") as out_file:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out_file.write(chunk)
        except (URLError, HTTPError) as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to download DIV2K archive from {url}") from exc

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root)

    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass

    if not target_dir.exists():
        raise FileNotFoundError(
            f"Expected directory {target_dir} was not created after extracting {zip_name}. "
            "Check the archive contents or download manually."
        )


def ensure_div2k_available(args: argparse.Namespace) -> None:
    target_dir = expected_div2k_dir(args)
    if target_dir.exists():
        return
    if not args.auto_download:
        raise FileNotFoundError(
            f"DIV2K subset not found at {target_dir}. "
            "Download it manually or rerun with --auto-download."
        )
    download_div2k_subset(args, target_dir)


def _maybe_resolve_output_dir(base: Optional[Path]) -> Optional[Path]:
    if base is None:
        return None
    base.mkdir(parents=True, exist_ok=True)
    return base


def _run_synthetic_demo(args: argparse.Namespace) -> None:
    checkerboard_scene = SyntheticData("Checker Board").create_img(args.seed)
    run_forward_model(
        checkerboard_scene,
        patterns=args.patterns,
        selected_pattern=args.selected_pattern,
        T=args.taps,
        blur_length_px=args.blur_length,
        duty_cycle=args.duty_cycle,
        random_seed=args.seed,
        photon_budget=args.photon_budget,
        read_noise_sigma=args.read_noise,
        show_plots=args.show_plots,
        save_arrays=args.save_arrays,
        save_pngs=args.save_pngs,
        save_figures=args.save_figures,
        output_dir=_maybe_resolve_output_dir(args.output_dir),
        verbose=args.verbose,
    )

    ring_scene = SyntheticData("Rings").create_img(args.seed)
    run_forward_model(
        ring_scene,
        patterns=args.patterns,
        selected_pattern=args.selected_pattern,
        T=args.taps,
        blur_length_px=args.blur_length,
        duty_cycle=args.duty_cycle,
        random_seed=args.seed,
        photon_budget=args.photon_budget,
        read_noise_sigma=args.read_noise,
        show_plots=args.show_plots,
        save_arrays=args.save_arrays,
        save_pngs=args.save_pngs,
        save_figures=args.save_figures,
        output_dir=_maybe_resolve_output_dir(args.output_dir),
        verbose=args.verbose,
    )


def _run_div2k_batch(args: argparse.Namespace) -> None:
    assert args.div2k_root is not None
    ensure_div2k_available(args)
    dataset = DIV2KDataset(
        root=args.div2k_root,
        subset=args.subset,
        degradation=args.degradation,
        scale=args.scale,
        limit=args.limit,
        target_size=args.target_size,
        image_mode=args.image_mode,
    )

    if args.output_dir is not None:
        base_output = args.output_dir
    else:
        base_output = default_output_dir() / "div2k"

    base_output.mkdir(parents=True, exist_ok=True)

    for idx, (path, image) in enumerate(dataset):
        relative_dir = base_output / path.stem
        relative_dir.mkdir(parents=True, exist_ok=True)
        run_forward_model(
            image,
            patterns=args.patterns,
            selected_pattern=args.selected_pattern,
            T=args.taps,
            blur_length_px=args.blur_length,
            duty_cycle=args.duty_cycle,
            random_seed=args.seed + idx,
            photon_budget=args.photon_budget,
            read_noise_sigma=args.read_noise,
            show_plots=args.show_plots,
            save_arrays=args.save_arrays,
            save_pngs=args.save_pngs,
            save_figures=args.save_figures,
            output_dir=relative_dir,
            verbose=args.verbose,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.div2k_root is None:
        _run_synthetic_demo(args)
        return

    if args.include_synthetic:
        _run_synthetic_demo(args)

    _run_div2k_batch(args)


if __name__ == "__main__":
    main()

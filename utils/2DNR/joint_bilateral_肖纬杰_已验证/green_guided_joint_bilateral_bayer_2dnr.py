#!/usr/bin/env python
"""
Standalone Bayer RAW 2D noise reduction.

This module extracts the Bayer-domain noise reduction logic from
Infinite-ISP's modules/bayer_noise_reduction implementation and exposes it as
a small RAW-in/RAW-out utility.

Algorithm:
    Green-guided joint bilateral filtering on a Bayer mosaic.

Expected input:
    A 2D Bayer RAW array in sensor code values, e.g. uint16 values in
    [0, 2**bit_depth - 1].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUPPORTED_BAYER_PATTERNS = {"rggb", "bggr", "grbg", "gbrg"}


@dataclass(frozen=True)
class BayerRaw2DNRConfig:
    bayer_pattern: str = "rggb"
    bit_depth: int = 12
    filter_window: int = 9
    r_std_dev_s: float = 1.0
    r_std_dev_r: float = 0.1
    g_std_dev_s: float = 1.0
    g_std_dev_r: float = 0.08
    b_std_dev_s: float = 1.0
    b_std_dev_r: float = 0.1


def _normalize_pattern(pattern: str) -> str:
    pattern = pattern.lower()
    if pattern not in SUPPORTED_BAYER_PATTERNS:
        supported = ", ".join(sorted(SUPPORTED_BAYER_PATTERNS))
        raise ValueError(
            f"Unsupported Bayer pattern {pattern!r}; use one of {supported}"
        )
    return pattern


def _validate_raw(raw: np.ndarray, config: BayerRaw2DNRConfig) -> None:
    if raw.ndim != 2:
        raise ValueError(f"Bayer RAW input must be 2D, got shape {raw.shape}")
    if raw.shape[0] % 2 != 0 or raw.shape[1] % 2 != 0:
        raise ValueError(f"Bayer RAW height and width must be even, got {raw.shape}")
    if config.bit_depth <= 0 or config.bit_depth > 32:
        raise ValueError(f"bit_depth must be in [1, 32], got {config.bit_depth}")
    if config.filter_window <= 0:
        raise ValueError(f"filter_window must be positive, got {config.filter_window}")

    std_devs = (
        config.r_std_dev_s,
        config.r_std_dev_r,
        config.g_std_dev_s,
        config.g_std_dev_r,
        config.b_std_dev_s,
        config.b_std_dev_r,
    )
    if any(value <= 0 for value in std_devs):
        raise ValueError(
            f"All spatial/range standard deviations must be positive: {std_devs}"
        )


def _channel_positions(pattern: str) -> dict[str, tuple[int, int]]:
    positions = {}
    for channel, position in zip(pattern, ((0, 0), (0, 1), (1, 0), (1, 1))):
        if channel != "g":
            positions[channel] = position
    return positions


def _make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _gaussian_kernel(size: int, std_dev: float, stride: int) -> np.ndarray:
    size = _make_odd(size)
    if size <= 0:
        size = 3

    center = (size - 1) / 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            dy = stride * (y - center)
            dx = stride * (x - center)
            kernel[y, x] = np.exp(-((dy * dy) + (dx * dx)) / (2 * std_dev * std_dev))

    return kernel / np.sum(kernel)


def _convolve_reflect_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)

    for y in range(kernel_height):
        for x in range(kernel_width):
            out += (
                kernel[y, x]
                * padded[y : y + image.shape[0], x : x + image.shape[1]]
            )

    return out


def _green_interpolation_kernel() -> np.ndarray:
    kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float32,
    )
    return kernel / np.sum(kernel)


def _fast_joint_bilateral_filter(
    image: np.ndarray,
    guide: np.ndarray,
    spatial_kernel_size: int,
    std_dev_s: float,
    std_dev_r: float,
    stride: int,
) -> np.ndarray:
    spatial_kernel_size = _make_odd(spatial_kernel_size)
    spatial_kernel = _gaussian_kernel(spatial_kernel_size, std_dev_s, stride)
    pad = spatial_kernel_size // 2
    image_pad = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    guide_pad = np.pad(guide, ((pad, pad), (pad, pad)), mode="reflect")

    norm = np.zeros(image.shape, dtype=np.float32)
    weighted_sum = np.zeros(image.shape, dtype=np.float32)
    range_scale = 2 * std_dev_r * std_dev_r

    for y in range(spatial_kernel_size):
        for x in range(spatial_kernel_size):
            shifted_image = image_pad[y : y + image.shape[0], x : x + image.shape[1]]
            shifted_guide = guide_pad[y : y + guide.shape[0], x : x + guide.shape[1]]
            range_weights = np.exp(-((guide - shifted_guide) ** 2) / range_scale)
            weights = spatial_kernel[y, x] * range_weights
            norm += weights
            weighted_sum += weights * shifted_image

    return weighted_sum / norm


def apply_bayer_raw_2dnr(raw: np.ndarray, config: BayerRaw2DNRConfig) -> np.ndarray:
    """
    Apply Bayer RAW 2DNR and return a RAW array with the same shape and dtype.

    The returned array remains a single-channel Bayer mosaic. It is not
    demosaiced and is not converted to RGB/YUV.
    """
    pattern = _normalize_pattern(config.bayer_pattern)
    _validate_raw(raw, config)

    input_dtype = raw.dtype
    max_value = float((1 << config.bit_depth) - 1)
    in_img = raw.astype(np.float32, copy=False) / max_value
    in_img = np.clip(in_img, 0.0, 1.0)

    height, width = in_img.shape
    positions = _channel_positions(pattern)
    r_y, r_x = positions["r"]
    b_y, b_x = positions["b"]

    in_img_r = in_img[r_y:height:2, r_x:width:2]
    in_img_b = in_img[b_y:height:2, b_x:width:2]

    green_estimate = _convolve_reflect_2d(in_img, _green_interpolation_kernel())
    green_estimate = np.clip(green_estimate, 0.0, 1.0)

    interp_g = in_img.copy()
    interp_g[r_y:height:2, r_x:width:2] = green_estimate[r_y:height:2, r_x:width:2]
    interp_g[b_y:height:2, b_x:width:2] = green_estimate[b_y:height:2, b_x:width:2]

    interp_g_at_r = green_estimate[r_y:height:2, r_x:width:2]
    interp_g_at_b = green_estimate[b_y:height:2, b_x:width:2]

    filter_window = _make_odd(config.filter_window)
    rb_filter_window = int((filter_window + 1) / 2)

    out_img_r = _fast_joint_bilateral_filter(
        in_img_r,
        interp_g_at_r,
        rb_filter_window,
        config.r_std_dev_s,
        config.r_std_dev_r,
        stride=2,
    )
    out_img_g = _fast_joint_bilateral_filter(
        interp_g,
        interp_g,
        filter_window,
        config.g_std_dev_s,
        config.g_std_dev_r,
        stride=1,
    )
    out_img_b = _fast_joint_bilateral_filter(
        in_img_b,
        interp_g_at_b,
        rb_filter_window,
        config.b_std_dev_s,
        config.b_std_dev_r,
        stride=2,
    )

    out_img = out_img_g.copy()
    out_img[r_y:height:2, r_x:width:2] = out_img_r
    out_img[b_y:height:2, b_x:width:2] = out_img_b

    out_scaled = np.clip(out_img, 0.0, 1.0) * max_value
    if np.issubdtype(input_dtype, np.integer):
        return np.rint(out_scaled).astype(input_dtype)

    return out_scaled.astype(input_dtype, copy=False)


def load_raw_file(path: str | Path, width: int, height: int, dtype: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    expected = width * height
    if raw.size != expected:
        raise ValueError(
            f"RAW size mismatch: expected {expected} pixels for {width}x{height}, "
            f"got {raw.size}"
        )
    return raw.reshape((height, width))


def save_raw_file(path: str | Path, raw: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.tofile(output_path)


def load_safetensors_file(path: str | Path, tensor_key: str) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as tensors:
        keys = list(tensors.keys())
        if tensor_key not in keys:
            raise KeyError(
                f"{path} does not contain tensor {tensor_key!r}; available keys: {keys}"
            )
        return tensors.get_tensor(tensor_key)


def save_safetensors_file(path: str | Path, tensor_key: str, raw: np.ndarray) -> None:
    from safetensors.numpy import save_file

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({tensor_key: np.ascontiguousarray(raw)}, str(output_path))


def _default_dtype(bit_depth: int) -> str:
    return "uint8" if bit_depth <= 8 else "uint16"


def _infer_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if path.suffix.lower() == ".safetensors":
        return "safetensors"
    return "raw"


def _ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )


def _process_safetensors_file(
    input_path: Path,
    output_path: Path,
    config: BayerRaw2DNRConfig,
    tensor_key: str,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_path, overwrite)
    raw = load_safetensors_file(input_path, tensor_key)
    denoised = apply_bayer_raw_2dnr(raw, config)
    save_safetensors_file(output_path, tensor_key, denoised)


def _process_raw_file(
    input_path: Path,
    output_path: Path,
    config: BayerRaw2DNRConfig,
    width: int | None,
    height: int | None,
    dtype: str,
    overwrite: bool,
) -> None:
    if width is None or height is None:
        raise ValueError("RAW file input requires --width and --height")
    _ensure_can_write(output_path, overwrite)
    raw = load_raw_file(input_path, width, height, dtype)
    denoised = apply_bayer_raw_2dnr(raw, config)
    save_raw_file(output_path, denoised)


def _iter_safetensors_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.safetensors") if path.is_file())


def process_path(args: argparse.Namespace, config: BayerRaw2DNRConfig) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    dtype = args.dtype or _default_dtype(args.bit_depth)

    if input_path.is_dir():
        input_format = _infer_input_format(input_path, args.input_format)
        if input_format != "safetensors":
            raise ValueError("Directory input is only supported for .safetensors files")

        files = _iter_safetensors_files(input_path)
        if args.limit is not None:
            files = files[: args.limit]
        if not files:
            raise FileNotFoundError(f"No .safetensors files found under {input_path}")

        for index, file_path in enumerate(files, start=1):
            rel_path = file_path.relative_to(input_path)
            target_path = output_path / rel_path
            print(f"[{index}/{len(files)}] {rel_path}")
            _process_safetensors_file(
                file_path,
                target_path,
                config,
                args.tensor_key,
                args.overwrite,
            )
        return

    if not input_path.is_file():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    input_format = _infer_input_format(input_path, args.input_format)
    if input_format == "safetensors":
        target_path = output_path
        if output_path.exists() and output_path.is_dir():
            target_path = output_path / input_path.name
        _process_safetensors_file(
            input_path,
            target_path,
            config,
            args.tensor_key,
            args.overwrite,
        )
        return

    _process_raw_file(
        input_path,
        output_path,
        config,
        args.width,
        args.height,
        dtype,
        args.overwrite,
    )


def _run_self_test() -> None:
    raw = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 64) % 4096
    config = BayerRaw2DNRConfig(
        bayer_pattern="rggb",
        bit_depth=12,
        filter_window=3,
        r_std_dev_s=1.0,
        r_std_dev_r=0.1,
        g_std_dev_s=1.0,
        g_std_dev_r=0.08,
        b_std_dev_s=1.0,
        b_std_dev_r=0.1,
    )
    out = apply_bayer_raw_2dnr(raw, config)
    assert out.shape == raw.shape
    assert out.dtype == raw.dtype
    assert out.min() >= 0
    assert out.max() <= 4095
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Bayer RAW-domain 2DNR and write Bayer RAW output."
    )
    parser.add_argument("--input", help="Input .raw/.safetensors file or directory")
    parser.add_argument("--output", help="Output file or directory")
    parser.add_argument(
        "--input-format",
        choices=("auto", "raw", "safetensors"),
        default="auto",
        help="Input format. Directory input supports safetensors only.",
    )
    parser.add_argument(
        "--tensor-key",
        default="raw",
        help="Tensor key to read/write for safetensors input.",
    )
    parser.add_argument("--width", type=int, help="RAW width")
    parser.add_argument("--height", type=int, help="RAW height")
    parser.add_argument("--bit-depth", type=int, default=12, help="RAW bit depth")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("uint8", "uint16", "float32"),
        help=(
            "Storage dtype of the input RAW file. Defaults to uint8 for <=8-bit, "
            "otherwise uint16."
        ),
    )
    parser.add_argument(
        "--bayer-pattern",
        default="rggb",
        help="Bayer pattern: rggb, bggr, grbg, or gbrg. Case-insensitive.",
    )
    parser.add_argument("--filter-window", type=int, default=9)
    parser.add_argument("--r-std-dev-s", type=float, default=1.0)
    parser.add_argument("--r-std-dev-r", type=float, default=0.1)
    parser.add_argument("--g-std-dev-s", type=float, default=1.0)
    parser.add_argument("--g-std-dev-r", type=float, default=0.08)
    parser.add_argument("--b-std-dev-s", type=float, default=1.0)
    parser.add_argument("--b-std-dev-r", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N safetensors files from a directory.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _run_self_test()
        return

    missing = [name for name in ("input", "output") if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            f"Missing required arguments: {', '.join('--' + item for item in missing)}"
        )

    config = BayerRaw2DNRConfig(
        bayer_pattern=args.bayer_pattern,
        bit_depth=args.bit_depth,
        filter_window=args.filter_window,
        r_std_dev_s=args.r_std_dev_s,
        r_std_dev_r=args.r_std_dev_r,
        g_std_dev_s=args.g_std_dev_s,
        g_std_dev_r=args.g_std_dev_r,
        b_std_dev_s=args.b_std_dev_s,
        b_std_dev_r=args.b_std_dev_r,
    )

    process_path(args, config)


if __name__ == "__main__":
    main()

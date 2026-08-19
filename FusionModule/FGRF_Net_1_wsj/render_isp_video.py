"""Render FGRF fused SigmaStar RAW frames through the supplied ISP to MP4.

The OpenCV ISP in ``opencv_fixed_raw_compare_isp`` is DNG-oriented and does
not have a command for this SigmaStar raw-stream layout.  This adapter keeps
its demosaic, highlight reconstruction, colour conversion and sRGB stages,
then applies a per-frame adaptive Reinhard shoulder before video encoding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--isp-root", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-long-edge", type=int, default=0)
    parser.add_argument("--enable-highlight-recovery", action="store_true")
    parser.add_argument(
        "--white-balance",
        choices=("folder", "dynamic", "unity"),
        default="folder",
        help="folder uses R/G/B coefficients parsed from the SigmaStar path",
    )
    parser.add_argument(
        "--tone-map",
        choices=("adaptive-reinhard", "reinhard", "hard-clip"),
        default="adaptive-reinhard",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def natural_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name.lower())


def estimate_neutral(config: dict[str, Any]) -> list[float]:
    """Estimate AsShotNeutral from the scene's black-corrected 2DNR RAW."""
    base = config["base"]
    path = Path(base["path"])
    height, width = int(config["height"]), int(config["width"])
    expected = height * width
    raw = np.fromfile(path, dtype=np.uint16, count=expected)
    if raw.size != expected:
        raise ValueError(f"Cannot estimate white balance from {path}: expected {expected} samples")
    raw = raw.reshape(height, width).astype(np.float32)
    raw = np.maximum(raw - float(base.get("black_level", 300.0)), 0.0)
    red = float(np.mean(raw[0::2, 0::2]))
    green = float(0.5 * (np.mean(raw[0::2, 1::2]) + np.mean(raw[1::2, 0::2])))
    blue = float(np.mean(raw[1::2, 1::2]))
    if green <= 1e-6:
        return [1.0, 1.0, 1.0]
    return [float(np.clip(red / green, 0.05, 4.0)), 1.0, float(np.clip(blue / green, 0.05, 4.0))]


def folder_white_balance(config: dict[str, Any]) -> tuple[list[float], list[float], str]:
    """Parse SigmaStar R/G/B coefficients and convert gains to AsShotNeutral."""
    candidates = [config.get("noisy", {}).get("path", ""), config.get("base", {}).get("path", "")]
    match = next((re.search(r"R=(\d+),G=(\d+),B=(\d+)", path) for path in candidates if path), None)
    if match is None:
        raise ValueError("Cannot parse R=...,G=...,B=... white-balance coefficients from config paths")
    red, green, blue = (float(value) for value in match.groups())
    if min(red, green, blue) <= 0:
        raise ValueError(f"Invalid SigmaStar white-balance coefficients: {match.group(0)}")
    gains = [red / green, 1.0, blue / green]
    neutral = [1.0 / gains[0], 1.0, 1.0 / gains[2]]
    return neutral, gains, match.group(0)


def build_metadata(config: dict[str, Any], dng_color: Any, white_balance: str) -> dict[str, Any]:
    base = config["base"]
    # SigmaStar's @RG stream is RGGB.  B = XYZ_D65_TO_SRGB @ XYZ_D50_TO_D65
    # makes the ISP's DNG colour transform an identity camera-RGB transform;
    # no SigmaStar DNG colour matrices were provided with these raw streams.
    camera_to_srgb = dng_color.XYZ_D65_TO_SRGB @ dng_color.XYZ_D50_TO_D65
    if white_balance == "folder":
        neutral, gains, wb_source = folder_white_balance(config)
    elif white_balance == "dynamic":
        neutral = estimate_neutral(config)
        gains = [1.0 / neutral[0], 1.0, 1.0 / neutral[2]]
        wb_source = "black-corrected 2DNR channel means"
    else:
        neutral = [1.0, 1.0, 1.0]
        gains = [1.0, 1.0, 1.0]
        wb_source = "unity"
    return {
        "black": float(base.get("black_level", 300.0)),
        "white": float(base.get("white_level", 4095.0)),
        "cfa": [0, 1, 1, 2],
        "neutral": neutral,
        "color_matrix_1": camera_to_srgb.reshape(-1).tolist(),
        "color_matrix_2": [],
        "camera_calibration_1": np.eye(3).reshape(-1).tolist(),
        "camera_calibration_2": [],
        "analog_balance": [1.0, 1.0, 1.0],
        "calibration_illuminant_1": 21,
        "calibration_illuminant_2": 21,
        "default_crop_origin": [0, 0],
        "default_crop_size": [int(config["width"]), int(config["height"])],
        "active_area": [0, 0, int(config["height"]), int(config["width"])],
        "camera_model": "SigmaStar",
        "white_balance_source": wb_source,
        "white_balance_gains_rgb": gains,
    }


def adaptive_reinhard(linear: np.ndarray) -> tuple[np.ndarray, float]:
    finite = linear[np.isfinite(linear)]
    if finite.size == 0:
        return np.zeros_like(linear), 1.0
    shoulder = float(np.percentile(finite, 99.5))
    scale = 0.9 / max(shoulder, 1e-6)
    mapped = linear * np.float32(scale)
    return mapped / (1.0 + mapped), scale


def resize_long_edge(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    if max_long_edge <= 0 or max(image.shape[:2]) <= max_long_edge:
        return image
    scale = max_long_edge / max(image.shape[:2])
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_frame(raw_path: Path, metadata: dict[str, Any], isp_modules: dict[str, Any], tone_map: str, highlight: bool, max_long_edge: int) -> tuple[np.ndarray, float]:
    pipeline = isp_modules["pipeline"]
    dng_color = isp_modules["dng_color"]
    highlight_recovery = isp_modules["highlight_recovery"]
    raw = np.fromfile(raw_path, dtype=np.uint16)
    expected = int(metadata["active_area"][2]) * int(metadata["active_area"][3])
    if raw.size != expected:
        raise ValueError(f"{raw_path} has {raw.size} uint16 samples, expected {expected}")
    raw = raw.reshape((int(metadata["active_area"][2]), int(metadata["active_area"][3])))
    raw01 = pipeline.normalize_raw(raw, metadata)
    camera_rgb = pipeline.demosaic_edge_aware(raw01, metadata["cfa"])
    if highlight:
        camera_rgb = highlight_recovery.reconstruct_camera_highlights(
            camera_rgb, metadata["neutral"]
        )
    linear, _, _ = dng_color.camera_to_linear_srgb(camera_rgb, metadata)
    if tone_map == "adaptive-reinhard":
        linear, scale = adaptive_reinhard(linear)
    elif tone_map == "reinhard":
        linear = linear / (1.0 + linear)
        scale = 1.0
    else:
        scale = 1.0
    rgb = pipeline.to_uint8(pipeline.gamma_srgb(linear))
    return resize_long_edge(rgb, max_long_edge), scale


def main() -> int:
    args = parse_args()
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(f"RAW directory does not exist: {args.raw_dir}")
    if not args.isp_root.is_dir():
        raise FileNotFoundError(f"ISP root does not exist: {args.isp_root}")
    sys.path.insert(0, str(args.isp_root.resolve()))
    from opencv_source_match import dng_color, highlight_recovery, pipeline

    config = load_config(args.config)
    metadata = build_metadata(config, dng_color, args.white_balance)
    frames = sorted(args.raw_dir.glob("*.raw"), key=natural_key)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise FileNotFoundError(f"No .raw frames found in {args.raw_dir}")

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    first, first_scale = render_frame(
        frames[0], metadata,
        {"pipeline": pipeline, "dng_color": dng_color, "highlight_recovery": highlight_recovery},
        args.tone_map, args.enable_highlight_recovery, args.max_long_edge,
    )
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(args.output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open MP4 writer: {args.output_mp4}")
    scales = [first_scale]
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for index, frame_path in enumerate(frames[1:], start=1):
            image, scale = render_frame(
                frame_path, metadata,
                {"pipeline": pipeline, "dng_color": dng_color, "highlight_recovery": highlight_recovery},
                args.tone_map, args.enable_highlight_recovery, args.max_long_edge,
            )
            if image.shape[:2] != (height, width):
                raise ValueError(f"Frame size changed at {frame_path}: {image.shape[:2]} != {(height, width)}")
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            scales.append(scale)
            if (index + 1) % 20 == 0 or index + 1 == len(frames):
                print(f"rendered {index + 1}/{len(frames)} frames", flush=True)
    finally:
        writer.release()

    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "adapter": "FGRF_Net_1/render_isp_video.py",
        "raw_dir": str(args.raw_dir.resolve()),
        "output_mp4": str(args.output_mp4.resolve()),
        "frame_count": len(frames),
        "fps": args.fps,
        "size_wh": [width, height],
        "bayer_pattern": "RGGB",
        "highlight_recovery": bool(args.enable_highlight_recovery),
        "tone_map": args.tone_map,
        "white_balance": metadata["white_balance_source"],
        "white_balance_gains_rgb": metadata["white_balance_gains_rgb"],
        "adaptive_reinhard_scale_min": float(min(scales)),
        "adaptive_reinhard_scale_max": float(max(scales)),
        "metadata_source": "SigmaStar config and selected white-balance mode",
        "estimated_as_shot_neutral": metadata["neutral"],
    }
    report_path = args.output_mp4.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote MP4: {args.output_mp4}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

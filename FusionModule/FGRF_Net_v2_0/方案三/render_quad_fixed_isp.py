#!/usr/bin/env python3
"""Render 2DNR, 3DNR, v2 fusion, and forward RAFT flow as a fixed-scale quad MP4."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from raw_io import RawSequence


_NUMBER = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fused-raw-dir", type=Path, required=True)
    parser.add_argument("--isp-root", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fixed-scale", type=float, default=0.0)
    parser.add_argument("--scale-percentile", type=float, default=99.5)
    parser.add_argument("--scale-sample-stride", type=int, default=16)
    parser.add_argument("--flow-max-magnitude", type=float, default=0.0)
    parser.add_argument("--flow-percentile", type=float, default=99.0)
    parser.add_argument("--flow-sample-stride", type=int, default=16)
    parser.add_argument("--white-balance", choices=("folder", "dynamic", "unity"), default="folder")
    parser.add_argument("--no-highlight-recovery", dest="highlight_recovery", action="store_false")
    parser.set_defaults(highlight_recovery=True)
    parser.add_argument("--encoder", choices=("h264_nvenc", "hevc_nvenc", "libx264"), default="h264_nvenc")
    parser.add_argument("--cq", type=int, default=12)
    parser.add_argument("--pixel-format", choices=("yuv420p", "yuv444p"), default="yuv420p")
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--no-labels", action="store_true", help="Do not burn panel labels into the video; write a sidecar layout TXT instead.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def numbered_files(directory: Path, suffix: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    indexed: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        match = _NUMBER.search(path.stem)
        if match is not None:
            index = int(match.group(1))
            if index in indexed:
                raise ValueError(f"Multiple {suffix} files map to frame {index}: {directory}")
            indexed[index] = path
    if not indexed:
        raise FileNotFoundError(f"No {suffix} files found in {directory}")
    return indexed


def folder_white_balance(config: dict[str, Any]) -> tuple[list[float], list[float], str]:
    paths = (config.get("noisy", {}).get("path", ""), config.get("base", {}).get("path", ""))
    match = next((re.search(r"R=(\d+),G=(\d+),B=(\d+)", path) for path in paths if path), None)
    if match is None:
        raise ValueError("Cannot parse R=...,G=...,B=... white balance from the config")
    red, green, blue = (float(value) for value in match.groups())
    if min(red, green, blue) <= 0.0:
        raise ValueError(f"Invalid white-balance values: {match.group(0)}")
    gains = [red / green, 1.0, blue / green]
    return [1.0 / gains[0], 1.0, 1.0 / gains[2]], gains, match.group(0)


def build_metadata(config: dict[str, Any], dng_color: Any, white_balance: str) -> dict[str, Any]:
    base = config["base"]
    if white_balance == "folder":
        neutral, gains, source = folder_white_balance(config)
    elif white_balance == "unity":
        neutral, gains, source = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], "unity"
    else:
        raise ValueError("dynamic white balance is not supported by fixed-scale comparison")
    camera_to_srgb = dng_color.XYZ_D65_TO_SRGB @ dng_color.XYZ_D50_TO_D65
    return {
        "black": float(base["black_level"]),
        "white": float(base["white_level"]),
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
        "white_balance_source": source,
        "white_balance_gains_rgb": gains,
    }


def raw_to_linear(raw: np.ndarray, metadata: dict[str, Any], modules: dict[str, Any], highlight: bool) -> np.ndarray:
    normalized = modules["pipeline"].normalize_raw(raw, metadata)
    camera_rgb = modules["pipeline"].demosaic_edge_aware(normalized, metadata["cfa"])
    if highlight:
        camera_rgb = modules["highlight"].reconstruct_camera_highlights(camera_rgb, metadata["neutral"])
    linear, _, _ = modules["dng"].camera_to_linear_srgb(camera_rgb, metadata)
    return linear


def global_raw_scale(
    readers: tuple[Callable[[int], np.ndarray], ...],
    frame_count: int,
    metadata: dict[str, Any],
    modules: dict[str, Any],
    highlight: bool,
    percentile: float,
    stride: int,
) -> tuple[float, float]:
    if not 0.0 < percentile <= 100.0 or stride < 1:
        raise ValueError("Invalid scale percentile or sample stride")
    samples: list[np.ndarray] = []
    for index in range(frame_count):
        for reader in readers:
            linear = raw_to_linear(reader(index), metadata, modules, highlight)
            samples.append(linear[::stride, ::stride].reshape(-1).copy())
        if (index + 1) % 25 == 0 or index + 1 == frame_count:
            print(f"RAW scale prepass {index + 1}/{frame_count}", flush=True)
    white_point = float(np.percentile(np.concatenate(samples), percentile))
    if not np.isfinite(white_point) or white_point <= 0.0:
        raise ValueError(f"Invalid global white point: {white_point}")
    return 0.9 / white_point, white_point


def load_flow(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    flow = np.load(path, allow_pickle=False)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError(f"Invalid flow: {path}")
    height, width = target_hw
    old_height, old_width = flow.shape[:2]
    if flow.shape[:2] != target_hw:
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= width / old_width
        flow[..., 1] *= height / old_height
    return flow.astype(np.float32, copy=False)


def global_flow_magnitude(files: dict[int, Path], frame_count: int, percentile: float, stride: int) -> float:
    if frame_count < 2:
        return 1.0
    samples = []
    for index in range(frame_count - 1):
        flow = np.load(files[index], allow_pickle=False)
        samples.append(np.hypot(flow[::stride, ::stride, 0], flow[::stride, ::stride, 1]).reshape(-1))
    magnitude = float(np.percentile(np.concatenate(samples), percentile))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError(f"Invalid global flow magnitude: {magnitude}")
    return magnitude


def flow_to_rgb(flow: np.ndarray, maximum: float) -> np.ndarray:
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hsv = np.empty((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod((angle + np.pi) * (179.0 / (2.0 * np.pi)), 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * (255.0 / maximum), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def tone_map(linear: np.ndarray, scale: float, pipeline: Any) -> np.ndarray:
    mapped = linear * np.float32(scale)
    mapped = mapped / (1.0 + mapped)
    return pipeline.to_uint8(pipeline.gamma_srgb(mapped))


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(width, height) / 1600.0)
    thickness, margin = max(1, round(scale * 2)), max(12, round(scale * 14))
    (_, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (max(320, round(width * 0.45)), text_height + baseline + 2 * margin), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0.0, dst=output)
    cv2.putText(output, text, (margin, margin + text_height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def ffmpeg_command(args: argparse.Namespace, size: tuple[int, int]) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required")
    width, height = size
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y" if args.overwrite else "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(args.fps), "-i", "pipe:0", "-an", "-c:v", args.encoder,
    ]
    if args.encoder.endswith("_nvenc"):
        if args.lossless:
            command.extend(["-preset", "lossless", "-tune", "lossless", "-rc", "constqp", "-qp", "0"])
        else:
            command.extend(["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0", "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8"])
        if args.encoder == "h264_nvenc" and args.pixel_format == "yuv420p":
            command.extend(["-profile:v", "high", "-level:v", "5.1"])
        elif args.encoder == "h264_nvenc":
            command.extend(["-profile:v", "high444p"])
    elif args.lossless:
        command.extend(["-preset", "veryslow", "-crf", "0"])
    else:
        command.extend(["-preset", "slow", "-crf", str(args.cq)])
    command.extend(["-pix_fmt", args.pixel_format, "-movflags", "+faststart", str(args.output_mp4)])
    return command


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0 or args.fixed_scale < 0.0 or not 0 <= args.cq <= 51:
        raise ValueError("Invalid fps, fixed scale, or CQ")
    if not args.isp_root.is_dir():
        raise FileNotFoundError(f"ISP root not found: {args.isp_root}")
    if args.output_mp4.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_mp4}; pass --overwrite")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    height, width = int(config["height"]), int(config["width"])
    base = RawSequence(config["base"]["path"], height, width, config["base"].get("dtype", "uint16"))
    temporal = RawSequence(config["temporal"]["path"], height, width, config["temporal"].get("dtype", "uint16"))
    fused_files = numbered_files(args.fused_raw_dir, ".raw")
    forward_files = numbered_files(Path(config["flow"]["forward_dir"]), ".npy")
    frame_count = min(base.frame_count, temporal.frame_count)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    for index in range(frame_count):
        if index not in fused_files:
            raise FileNotFoundError(f"Missing fused RAW frame {index}")
    for index in range(max(0, frame_count - 1)):
        if index not in forward_files:
            raise FileNotFoundError(f"Missing forward flow frame {index}")

    sys.path.insert(0, str(args.isp_root.resolve()))
    from opencv_source_match import dng_color, highlight_recovery, pipeline

    modules = {"dng": dng_color, "highlight": highlight_recovery, "pipeline": pipeline}
    metadata = build_metadata(config, dng_color, args.white_balance)
    expected_samples = height * width

    def fused_reader(index: int) -> np.ndarray:
        raw = np.fromfile(fused_files[index], dtype=np.uint16)
        if raw.size != expected_samples:
            raise ValueError(f"Invalid fused RAW size: {fused_files[index]}")
        return raw.reshape(height, width)

    readers = (base.read_uint16, temporal.read_uint16, fused_reader)
    if args.fixed_scale > 0.0:
        fixed_scale, white_point = args.fixed_scale, 0.9 / args.fixed_scale
    else:
        fixed_scale, white_point = global_raw_scale(readers, frame_count, metadata, modules, args.highlight_recovery, args.scale_percentile, args.scale_sample_stride)
    flow_maximum = args.flow_max_magnitude or global_flow_magnitude(forward_files, frame_count, args.flow_percentile, args.flow_sample_stride)
    print(f"fixed_scale={fixed_scale:.8f} white_point={white_point:.8f} flow_max={flow_maximum:.4f}px", flush=True)

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    size = (2 * width, 2 * height)
    command = ffmpeg_command(args, size)
    print("Encoding with:", " ".join(command), flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(frame_count):
            raw_panels = [tone_map(raw_to_linear(reader(index), metadata, modules, args.highlight_recovery), fixed_scale, pipeline) for reader in readers]
            if not args.no_labels:
                raw_panels = [add_label(panel, label) for panel, label in zip(raw_panels, ("2DNR", "3DNR", "FGRF-Net v2 Fusion"))]
            if index < frame_count - 1:
                flow_panel = flow_to_rgb(load_flow(forward_files[index], (height, width)), flow_maximum)
            else:
                flow_panel = np.zeros((height, width, 3), dtype=np.uint8)
            if not args.no_labels:
                flow_panel = add_label(flow_panel, f"Forward RAFT Flow {index:04d}->{index + 1:04d}" if index < frame_count - 1 else "Forward RAFT Flow unavailable")
            quad = np.concatenate((np.concatenate((raw_panels[0], raw_panels[1]), axis=1), np.concatenate((raw_panels[2], flow_panel), axis=1)), axis=0)
            process.stdin.write(quad.tobytes())
            if (index + 1) % 20 == 0 or index + 1 == frame_count:
                print(f"rendered {index + 1}/{frame_count}", flush=True)
    except BrokenPipeError as error:
        raise RuntimeError("ffmpeg stopped before rendering completed") from error
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {process.returncode}")

    report = {
        "tool": "FGRF_Net_v2_0/render_quad_fixed_isp.py", "config": str(args.config.resolve()),
        "fused_raw_dir": str(args.fused_raw_dir.resolve()), "output_mp4": str(args.output_mp4.resolve()),
        "frame_count": frame_count, "fps": args.fps, "panel_size_wh": [width, height], "quad_size_wh": list(size),
        "raw_panels": ["2DNR", "3DNR", "FGRF-Net v2 Fusion"], "flow_panel": "forward RAFT; direction=hue, magnitude=value",
        "fixed_linear_scale": fixed_scale, "global_linear_white_point": white_point,
        "scale_percentile": args.scale_percentile, "highlight_recovery": args.highlight_recovery,
        "white_balance": metadata["white_balance_source"], "white_balance_gains_rgb": metadata["white_balance_gains_rgb"],
        "flow_maximum_pixels": flow_maximum, "flow_percentile": args.flow_percentile,
        "encoder": args.encoder, "pixel_format": args.pixel_format, "lossless": args.lossless, "cq": args.cq,
        "labels_in_video": not args.no_labels,
    }
    report_path = args.output_mp4.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    layout_path = args.output_mp4.with_suffix(".txt")
    layout_path.write_text(
        "FGRF-Net v2 Base16 + Separable-Conv fixed-scale quad comparison layout\n"
        f"Video: {args.output_mp4.name}\n"
        f"Panel resolution: {width}x{height}\n"
        f"Quad resolution: {size[0]}x{size[1]}\n\n"
        "Top-left: 2DNR\n"
        "Top-right: 3DNR\n"
        "Bottom-left: FGRF-Net v2 Base16 + Separable-Conv Fusion\n"
        "Bottom-right: Forward RAFT Flow\n",
        encoding="utf-8",
    )
    print(f"Wrote MP4: {args.output_mp4}\nWrote report: {report_path}\nWrote layout: {layout_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

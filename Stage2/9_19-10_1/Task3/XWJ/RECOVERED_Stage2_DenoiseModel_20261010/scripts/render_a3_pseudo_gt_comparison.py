#!/usr/bin/env python3
"""Render fixed-ISP input versus A3 pseudo-GT application comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json
from raw_fusion.finetune.a4 import _read_aligned_video


_RAW_SHAPE = (1080, 1920)


def _srgb_transfer(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _simple_isp(raw12: np.ndarray, black: int, white_balance: tuple[float, float, float], exposure: float) -> np.ndarray:
    if raw12.shape != _RAW_SHAPE or raw12.dtype != np.uint16:
        raise ValueError(f"expected uint16 RAW {_RAW_SHAPE}, got {raw12.shape} {raw12.dtype}")
    denominator = float(_WHITE_LEVEL - black)
    linear16 = np.rint(np.clip((raw12.astype(np.float32) - black) / denominator, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb16 = cv2.cvtColor(linear16, cv2.COLOR_BayerRGGB2RGB)
    rgb_linear = rgb16.astype(np.float32) / 65535.0
    rgb_linear *= np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    rgb_linear *= np.float32(exposure)
    return np.rint(np.clip(_srgb_transfer(np.clip(rgb_linear, 0.0, 1.0)), 0.0, 1.0) * 255.0).astype(np.uint8)


def _pack_to_raw(normalized: np.ndarray, black: np.ndarray) -> np.ndarray:
    dn = normalized * (_WHITE_LEVEL - black[:, None, None]) + black[:, None, None]
    packed = np.rint(np.clip(dn, 0.0, _WHITE_LEVEL)).astype("<u2")
    raw = np.empty(_RAW_SHAPE, dtype="<u2")
    raw[0::2, 0::2], raw[0::2, 1::2] = packed[0], packed[1]
    raw[1::2, 0::2], raw[1::2, 1::2] = packed[2], packed[3]
    return raw


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 62), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def _video_encoder() -> str:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    available = result.stdout + result.stderr
    if " libx264 " in available:
        return "libx264"
    if " libx265 " in available:
        return "libx265"
    return "mpeg4"


def _render_gain(
    *,
    supervision_dir: Path,
    pseudo_gt_dir: Path,
    gain: str,
    output_dir: Path,
    dataset_config: Path,
    max_frames: int,
    fps: int,
) -> dict[str, object]:
    dataset = _load_json(dataset_config)
    sequence = dataset["sequences"][gain]
    white_balance = tuple(float(value) for value in sequence["white_balance"])
    exposure = float(sequence["isp_gain"])
    if len(white_balance) != 3:
        raise ValueError(f"{gain}: fixed white balance must have three values")

    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    reference_black = _channel_vector(report["black_levels"])
    input_video, _ = _read_aligned_video(supervision_dir, gain, max_frames)
    frame_dir = pseudo_gt_dir / gain
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"input_vs_a3_pseudogt_{gain}.mp4"
    preview_path = output_dir / f"input_vs_a3_pseudogt_{gain}_frame0000.png"
    if video_path.exists() or preview_path.exists():
        raise FileExistsError(f"output already exists for {gain}: {output_dir}")

    width = _RAW_SHAPE[1] * 2
    height = _RAW_SHAPE[0]
    encoder = _video_encoder()
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-c:v", encoder,
    ]
    if encoder == "libx265":
        command.extend(["-x265-params", "lossless=1", "-pix_fmt", "yuv444p"])
    elif encoder == "libx264":
        command.extend(["-crf", "18", "-pix_fmt", "yuv444p"])
    else:
        command.extend(["-q:v", "3", "-pix_fmt", "yuv420p"])
    command.append(str(video_path))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    first_frame: np.ndarray | None = None
    try:
        for index, normalized in enumerate(input_video):
            input_raw = _pack_to_raw(normalized, reference_black)
            a3_raw = np.fromfile(frame_dir / f"out_{index:04d}.raw", dtype="<u2")
            if a3_raw.size != _RAW_SHAPE[0] * _RAW_SHAPE[1]:
                raise ValueError(f"{gain} frame {index}: unexpected A3 RAW size {a3_raw.size}")
            a3_raw = a3_raw.reshape(_RAW_SHAPE)
            comparison = np.concatenate(
                (
                    _label(_simple_isp(input_raw, 252, white_balance, exposure), f"noisy input {gain} frame {index:04d}"),
                    _label(_simple_isp(a3_raw, 252, white_balance, exposure), f"A3 pseudo-GT {gain} frame {index:04d}"),
                ),
                axis=1,
            )
            if first_frame is None:
                first_frame = comparison
            process.stdin.write(np.ascontiguousarray(comparison).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {gain}: {stderr[-2000:]}")
    except BaseException:
        process.kill()
        process.wait()
        video_path.unlink(missing_ok=True)
        raise
    if first_frame is None:
        raise RuntimeError(f"no frames rendered for {gain}")
    # OpenCV writes BGR while the comparison buffer is RGB.
    cv2.imwrite(str(preview_path), first_frame[:, :, ::-1])
    return {
        "gain": gain,
        "frame_count": len(input_video),
        "fixed_black": 252,
        "fixed_white": int(_WHITE_LEVEL),
        "white_balance": list(white_balance),
        "exposure": exposure,
        "video_encoder": encoder,
        "video_pix_fmt": "yuv444p" if encoder in ("libx264", "libx265") else "yuv420p",
        "video_lossless": encoder == "libx265",
        "video": str(video_path.resolve()),
        "preview": str(preview_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--pseudo-gt-dir", type=Path, default=Path("artifacts/finetune_mis20s1/pseudo_gt/a3_step005000"))
    parser.add_argument("--dataset-config", type=Path, default=Path("/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/configs/dataset_mis20s1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/application_validation/input_vs_a3_step005000_fixed"))
    parser.add_argument("--gains", default="128x,645x")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.max_frames < 1 or args.fps < 1:
        raise ValueError("max-frames and fps must be positive")
    results = {}
    for gain in (value.strip() for value in args.gains.split(",")):
        if gain:
            results[gain] = _render_gain(
                supervision_dir=args.supervision_dir,
                pseudo_gt_dir=args.pseudo_gt_dir,
                gain=gain,
                output_dir=args.output_dir,
                dataset_config=args.dataset_config,
                max_frames=args.max_frames,
                fps=args.fps,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({"protocol": "fixed ISP noisy input vs A3 pseudo-GT", "gains": results}, indent=2) + "\n", encoding="ascii")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

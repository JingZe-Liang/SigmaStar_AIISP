#!/usr/bin/env python3
"""Render noisy input versus a downstream RAW fusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json
from raw_fusion.finetune.a4 import _read_aligned_video
from render_input_vs_a3_pseudogt import _RAW_SHAPE, _label, _pack_to_raw, _simple_isp


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
    model_raw: Path,
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
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    reference_black = _channel_vector(report["black_levels"])
    input_video, _ = _read_aligned_video(supervision_dir, gain, max_frames)
    expected_values = max_frames * _RAW_SHAPE[0] * _RAW_SHAPE[1]
    model_array = np.memmap(model_raw, dtype="<u2", mode="r", shape=(max_frames, *_RAW_SHAPE))
    if model_array.size != expected_values:
        raise ValueError(f"{gain}: downstream RAW size does not match {max_frames} frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"input_vs_downstream_probe_{gain}.mp4"
    preview_path = output_dir / f"input_vs_downstream_probe_{gain}_frame0000.png"
    width = _RAW_SHAPE[1] * 2
    encoder = _video_encoder()
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{_RAW_SHAPE[0]}",
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
            comparison = np.concatenate(
                (
                    _label(_simple_isp(input_raw, 252, white_balance, exposure), f"noisy input {gain} frame {index:04d}"),
                    _label(_simple_isp(np.asarray(model_array[index]), 252, white_balance, exposure), f"downstream probe {gain} frame {index:04d}"),
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
        raise RuntimeError(f"{gain}: no frames rendered")
    cv2.imwrite(str(preview_path), first_frame[:, :, ::-1])
    return {
        "gain": gain,
        "frame_count": max_frames,
        "fixed_black": 252,
        "fixed_white": int(_WHITE_LEVEL),
        "white_balance": list(white_balance),
        "exposure": exposure,
        "video_encoder": encoder,
        "video_pix_fmt": "yuv444p" if encoder in ("libx264", "libx265") else "yuv420p",
        "video_lossless": encoder == "libx265",
        "model_raw": str(model_raw.resolve()),
        "video": str(video_path.resolve()),
        "preview": str(preview_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset_mis20s1_a3_step005000.json"))
    parser.add_argument("--inference-root", type=Path, default=Path("artifacts/finetune_mis20s1/downstream_a3_step005000/inference"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/downstream_a3_step005000/application_validation"))
    parser.add_argument("--gains", default="128x,645x")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    results = {}
    for gain in (value.strip() for value in args.gains.split(",")):
        if gain:
            results[gain] = _render_gain(
                supervision_dir=args.supervision_dir,
                model_raw=args.inference_root / gain / f"{gain}_0000_{args.max_frames - 1:04d}.raw",
                gain=gain,
                output_dir=args.output_dir,
                dataset_config=args.dataset_config,
                max_frames=args.max_frames,
                fps=args.fps,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"protocol": "fixed ISP noisy input vs downstream A3 probe", "gains": results}, indent=2) + "\n",
        encoding="ascii",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

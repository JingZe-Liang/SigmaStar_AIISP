#!/usr/bin/env python3
"""Render an aligned MIS20S1 Input/A3/A9 fixed-ISP comparison video."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_a8_sc635hai import _sha256, _video_encoder  # noqa: E402
from raw_fusion.finetune.a3 import _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.a4 import _read_aligned_video  # noqa: E402
from raw_fusion.finetune.cross_sensor import (  # noqa: E402
    denormalise_native,
    render_fixed_isp,
)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _cell(image: np.ndarray, label: str) -> np.ndarray:
    overview = cv2.resize(image, (640, 362), interpolation=cv2.INTER_AREA)
    detail = _fit_image(image[260:1060, 240:1680], 640, 598)
    separator = np.full((4, 640, 3), 24, dtype=np.uint8)
    cell = np.concatenate((overview, separator, detail), axis=0)
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 46), (10, 10, 10), -1)
    cv2.putText(
        cell,
        label,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument("--candidate-prediction", type=Path, required=True)
    parser.add_argument(
        "--baseline-prediction",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/prediction_1000x.npy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    video, _ = _read_aligned_video(args.supervision_dir, "1000x", 8)
    baseline = np.asarray(
        np.load(args.baseline_prediction, mmap_mode="r")[:8], dtype=np.float32
    )
    candidate = np.asarray(np.load(args.candidate_prediction), dtype=np.float32)
    if baseline.shape != video.shape or candidate.shape != video.shape:
        raise ValueError("MIS20S1 input, A3, and A9 prediction shapes must match")
    report = _load_json(
        args.supervision_dir / "references/a1_reference_report.json"
    )
    reference_manifest = _load_json(
        args.supervision_dir / "stream_manifests/04_shdarkroom_10x.json"
    )
    black = _channel_vector(report["black_levels"])
    metadata = reference_manifest["filename_metadata"]
    wb = (
        float(metadata["wb_r"]) / float(metadata["wb_g"]),
        1.0,
        1.0,
        float(metadata["wb_b"]) / float(metadata["wb_g"]),
    )
    exposure = 4.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "mis20s1_1000x_input_a3_a9_fixed_isp.mp4"
    preview_path = args.output_dir / "mis20s1_1000x_input_a3_a9_fixed_isp_frame0000.png"
    encoder = _video_encoder()
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x964",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-c:v",
        encoder,
    ]
    if encoder == "libx265":
        command.extend(["-x265-params", "lossless=1", "-pix_fmt", "yuv444p"])
    elif encoder == "libx264":
        command.extend(["-crf", "16", "-pix_fmt", "yuv444p"])
    else:
        command.extend(["-q:v", "2", "-pix_fmt", "yuv420p"])
    command.append(str(video_path))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    first_frame: np.ndarray | None = None
    try:
        for frame_index in range(len(video)):
            images = [
                render_fixed_isp(
                    denormalise_native(values[frame_index], black),
                    black,
                    wb,
                    exposure,
                )
                for values in (video, baseline, candidate)
            ]
            labels = (
                f"Input | frame {frame_index}",
                f"A3 baseline | frame {frame_index}",
                f"A9 unified step-100 | frame {frame_index}",
            )
            frame = np.concatenate(
                [_cell(image, label) for image, label in zip(images, labels, strict=True)],
                axis=1,
            )
            if first_frame is None:
                first_frame = frame.copy()
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")
    except BaseException:
        process.kill()
        process.wait()
        video_path.unlink(missing_ok=True)
        raise
    if first_frame is None:
        raise RuntimeError("no MIS20S1 video frames were rendered")
    if not cv2.imwrite(
        str(preview_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"failed to write {preview_path}")
    result = {
        "frames": len(video),
        "fps": args.fps,
        "fixed_isp_wb": list(wb),
        "fixed_isp_exposure": exposure,
        "video": str(video_path.resolve()),
        "video_sha256": _sha256(video_path),
        "preview": str(preview_path.resolve()),
        "preview_sha256": _sha256(preview_path),
        "layout": "Input | A3 baseline | A9 unified step-100; overview and center crop",
    }
    (args.output_dir / "video_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

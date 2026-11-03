#!/usr/bin/env python3
"""Evaluate A9 on held-out SC465SL 2000x and render a fixed-ISP video."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_a8_sc635hai import (  # noqa: E402
    _candidate_disjoint_metrics,
    _load_reference,
    _load_specs,
    _save_json,
    _sha256,
    _static_summary,
    _video_encoder,
)
from raw_fusion.finetune.a4 import load_checkpoint_model  # noqa: E402
from raw_fusion.finetune.cross_sensor import (  # noqa: E402
    NativeReference,
    NativeSequenceSpec,
    channel_vector,
    denormalise_native,
    evaluate_disjoint_prediction,
    infer_disjoint_clips,
    open_sequence,
    read_normalized_clip,
    render_fixed_isp,
    split_clip_indices,
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


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 46), (10, 10, 10), -1)
    cv2.putText(
        result,
        text,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return result


def _comparison_cell(
    image: np.ndarray,
    label: str,
    crop: tuple[int, int, int, int],
) -> np.ndarray:
    overview = cv2.resize(image, (640, 362), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = crop
    detail = _fit_image(image[y0:y1, x0:x1], 640, 598)
    separator = np.full((4, 640, 3), 24, dtype=np.uint8)
    return _label(np.concatenate((overview, separator, detail), axis=0), label)


def _render_video(
    spec: NativeSequenceSpec,
    reference: NativeReference,
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    fps: int,
) -> dict[str, object]:
    reader = open_sequence(spec)
    black = channel_vector(reference.black_levels)
    baseline = np.load(baseline_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    clips = split_clip_indices(spec, 0)
    crop = dict(spec.panel_crops)["motion_center"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "sc465sl_2000x_input_a3_a9_fixed_isp"
    video_path = output_dir / f"{stem}.mp4"
    preview_path = output_dir / f"{stem}_frame0000.png"
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
        str(fps),
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
    offset = 0
    try:
        for clip_index, indices in enumerate(clips):
            input_normalized = read_normalized_clip(
                reader, spec.adapter, indices, black
            )
            for local_index, source_index in enumerate(indices):
                frames_dn = (
                    denormalise_native(input_normalized[local_index], black),
                    denormalise_native(
                        np.asarray(baseline[offset + local_index], dtype=np.float32),
                        black,
                    ),
                    denormalise_native(
                        np.asarray(candidate[offset + local_index], dtype=np.float32),
                        black,
                    ),
                )
                images = [
                    render_fixed_isp(
                        frame, black, spec.fixed_isp_wb, spec.fixed_isp_exposure
                    )
                    for frame in frames_dn
                ]
                labels = (
                    f"Input | clip {clip_index + 1} | frame {source_index}",
                    f"A3 baseline | clip {clip_index + 1} | frame {source_index}",
                    f"A9 unified step-100 | clip {clip_index + 1} | frame {source_index}",
                )
                frame = np.concatenate(
                    [
                        _comparison_cell(image, label, crop)
                        for image, label in zip(images, labels, strict=True)
                    ],
                    axis=1,
                )
                if frame.shape != (964, 1920, 3):
                    raise ValueError(f"unexpected rendered frame shape {frame.shape}")
                if first_frame is None:
                    first_frame = frame.copy()
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            offset += len(indices)
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
        raise RuntimeError("no SC465SL video frames were rendered")
    if not cv2.imwrite(
        str(preview_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"failed to write {preview_path}")
    return {
        "frames": offset,
        "fps": fps,
        "encoder": encoder,
        "video": str(video_path.resolve()),
        "video_sha256": _sha256(video_path),
        "preview": str(preview_path.resolve()),
        "preview_sha256": _sha256(preview_path),
        "layout": "Input | A3 baseline | A9 unified step-100; overview and motion crop",
    }


def _gate(
    baseline_even: dict[str, object],
    baseline_odd: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    baseline_directions = (baseline_even, baseline_odd)
    baseline_psnr = float(
        np.mean([item["a3"]["psnr_db"] for item in baseline_directions])
    )
    baseline_hf = float(
        np.mean([item["a3_high_frequency_ratio"] for item in baseline_directions])
    )
    baseline_temporal = float(
        np.mean([item["a3_temporal_mad_dn"] for item in baseline_directions])
    )
    baseline_bias = max(
        abs(float(item["a3_output_bias_dn"])) for item in baseline_directions
    )
    baseline_clip = max(
        float(item["normalized_clip_fraction"]) for item in baseline_directions
    )
    summary = candidate["summary"]
    checks = {
        "psnr_guard": bool(
            summary["mean_candidate_psnr_db"] >= baseline_psnr - 0.10
        ),
        "high_frequency_guard": bool(
            abs(summary["mean_candidate_high_frequency_ratio"] - 1.0)
            <= abs(baseline_hf - 1.0) + 0.02
        ),
        "temporal_mad_guard": bool(
            np.mean(
                [
                    candidate["even_input_vs_odd_reference"]["candidate_temporal_mad_dn"],
                    candidate["odd_input_vs_even_reference"]["candidate_temporal_mad_dn"],
                ]
            )
            <= baseline_temporal + 0.50
        ),
        "bias_guard": bool(
            summary["maximum_candidate_output_bias_abs_dn"] <= baseline_bias + 2.0
        ),
        "legal_raw_fraction": bool(
            summary["minimum_legal_raw_fraction"] == 1.0
        ),
        "clip_fraction_guard": bool(
            summary["maximum_normalized_clip_fraction"] <= baseline_clip + 0.005
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a3_cross_sensor_zero_shot_20261012.json",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "artifacts/a3_cross_sensor_zero_shot_20261012",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    evaluation_path = args.output_dir / "evaluation.json"
    if evaluation_path.is_file():
        raise FileExistsError(f"completed evaluation already exists: {evaluation_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_specs(args.config)["sc465sl_darkroom18_2000x"]
    baseline_dir = (
        args.baseline_root
        / "sequences/SC465SL/sc465sl_darkroom18_2000x"
    )
    reference = _load_reference(baseline_dir / "reference")
    device = torch.device(args.device)
    model = load_checkpoint_model(args.candidate_checkpoint, device, ema=True)
    checkpoint_sha256 = _sha256(args.candidate_checkpoint)
    inference_dir = args.output_dir / "inference"
    even_path = inference_dir / "prediction_even_input.npy"
    odd_path = inference_dir / "prediction_odd_input.npy"
    try:
        infer_disjoint_clips(
            model,
            spec,
            reference,
            0,
            device,
            even_path,
            tile_batch=1,
            resume_key=f"{checkpoint_sha256}:sc465sl-2000x:even",
        )
        infer_disjoint_clips(
            model,
            spec,
            reference,
            1,
            device,
            odd_path,
            tile_batch=1,
            resume_key=f"{checkpoint_sha256}:sc465sl-2000x:odd",
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    candidate_even = _candidate_disjoint_metrics(
        evaluate_disjoint_prediction(spec, reference, 0, even_path)
    )
    candidate_odd = _candidate_disjoint_metrics(
        evaluate_disjoint_prediction(spec, reference, 1, odd_path)
    )
    candidate = {
        "even_input_vs_odd_reference": candidate_even,
        "odd_input_vs_even_reference": candidate_odd,
        "summary": _static_summary(candidate_even, candidate_odd),
    }
    baseline_even_path = baseline_dir / "inference/prediction_even_input.npy"
    baseline_odd_path = baseline_dir / "inference/prediction_odd_input.npy"
    baseline_even = evaluate_disjoint_prediction(
        spec, reference, 0, baseline_even_path
    )
    baseline_odd = evaluate_disjoint_prediction(
        spec, reference, 1, baseline_odd_path
    )
    result = {
        "protocol": "A9 unified SC465SL held-out 2000x evaluation",
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "weights": "ema",
        "held_out_sequence": spec.name,
        "baseline_a3": {
            "even_input_vs_odd_reference": baseline_even,
            "odd_input_vs_even_reference": baseline_odd,
        },
        "candidate": candidate,
        "automatic_gate": _gate(baseline_even, baseline_odd, candidate),
        "fixed_isp_video": _render_video(
            spec,
            reference,
            baseline_even_path,
            even_path,
            args.output_dir / "fixed_isp_video",
            args.fps,
        ),
    }
    _save_json(evaluation_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

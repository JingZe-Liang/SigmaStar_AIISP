#!/usr/bin/env python3
"""Evaluate sc450ai A8 adaptation and export a fixed-ISP comparison video."""

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
    _candidate_stress_metrics,
    _load_profile,
    _load_specs,
    _mis20s1_guard,
    _save_json,
    _sha256,
    _video_encoder,
)
from raw_fusion.finetune.a3 import _load_json  # noqa: E402
from raw_fusion.finetune.a4 import load_checkpoint_model  # noqa: E402
from raw_fusion.finetune.cross_sensor import (  # noqa: E402
    NativeSequenceSpec,
    NativeStressProfile,
    channel_vector,
    contiguous_clip_indices,
    denormalise_native,
    evaluate_stress_prediction,
    infer_stress_clips,
    open_sequence,
    read_normalized_clip,
    render_fixed_isp,
)


def _single_clip_spec(
    source: NativeSequenceSpec, clip_start: int
) -> NativeSequenceSpec:
    return NativeSequenceSpec(
        name=source.name,
        sensor=source.sensor,
        path=source.path,
        width=source.width,
        height=source.height,
        frames_used=source.frames_used,
        bit_shift=source.bit_shift,
        cfa=source.cfa,
        nominal_black_dn=source.nominal_black_dn,
        reference_mode=source.reference_mode,
        clip_starts=(clip_start,),
        fixed_isp_wb=source.fixed_isp_wb,
        fixed_isp_exposure=source.fixed_isp_exposure,
        panel_crops=source.panel_crops,
    )


def _slice_baseline_tail(source_path: Path, output_path: Path) -> Path:
    source = np.load(source_path, mmap_mode="r")
    if source.shape[0] < 8:
        raise ValueError("baseline prediction does not contain one complete clip")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(source[-8:], dtype=np.float16))
    return output_path


def _automatic_gate(
    baseline_train_tail: dict[str, object],
    candidate_train_tail: dict[str, object],
    baseline_holdout: dict[str, object],
    candidate_holdout: dict[str, object],
    replay_guard: dict[str, object],
) -> dict[str, object]:
    checks = {
        "train_tail_legal_raw_fraction": (
            candidate_train_tail["legal_raw_fraction"] == 1.0
        ),
        "train_tail_clip_fraction_guard": (
            candidate_train_tail["normalized_clip_fraction"]
            <= baseline_train_tail["normalized_clip_fraction"] + 0.005
        ),
        "train_tail_high_frequency_guard": (
            candidate_train_tail["spatial_high_frequency_ratio"]
            >= baseline_train_tail["spatial_high_frequency_ratio"] - 0.03
        ),
        "train_tail_static_flicker_guard": (
            candidate_train_tail["static_temporal_ratio"]
            <= baseline_train_tail["static_temporal_ratio"] + 0.05
        ),
        "holdout_legal_raw_fraction": candidate_holdout["legal_raw_fraction"] == 1.0,
        "holdout_clip_fraction_guard": (
            candidate_holdout["normalized_clip_fraction"]
            <= baseline_holdout["normalized_clip_fraction"] + 0.005
        ),
        "holdout_high_frequency_recovery": (
            candidate_holdout["spatial_high_frequency_ratio"]
            >= baseline_holdout["spatial_high_frequency_ratio"] + 0.02
        ),
        "holdout_motion_guard": (
            candidate_holdout["motion_temporal_ratio"]
            >= baseline_holdout["motion_temporal_ratio"] - 0.01
        ),
        "holdout_static_flicker_guard": (
            candidate_holdout["static_temporal_ratio"]
            <= baseline_holdout["static_temporal_ratio"] + 0.05
        ),
        "mis20s1_psnr_guard": replay_guard["delta"]["psnr_db"] >= -0.10,
        "mis20s1_high_frequency_guard": (
            replay_guard["delta"]["high_frequency_ratio"] >= -0.03
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "note": (
            "Neither sc450ai sequence has a clean target. High-frequency recovery "
            "is necessary but not sufficient; fixed-ISP video review is mandatory."
        ),
    }


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
    profile: NativeStressProfile,
    a3_prediction_path: Path,
    candidate_prediction_path: Path,
    output_dir: Path,
    fps: int,
) -> dict[str, object]:
    reader = open_sequence(spec)
    black = channel_vector(profile.black_levels)
    a3_prediction = np.load(a3_prediction_path, mmap_mode="r")
    candidate_prediction = np.load(candidate_prediction_path, mmap_mode="r")
    clips = contiguous_clip_indices(spec)
    crop = dict(spec.panel_crops)["black_subject"]
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "sc450ai_1435x_input_a3_a8_fixed_isp.mp4"
    preview_path = output_dir / "sc450ai_1435x_input_a3_a8_fixed_isp_frame0000.png"
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
                input_dn = denormalise_native(input_normalized[local_index], black)
                a3_dn = denormalise_native(
                    np.asarray(a3_prediction[offset + local_index], dtype=np.float32),
                    black,
                )
                candidate_dn = denormalise_native(
                    np.asarray(
                        candidate_prediction[offset + local_index], dtype=np.float32
                    ),
                    black,
                )
                images = [
                    render_fixed_isp(
                        frame, black, spec.fixed_isp_wb, spec.fixed_isp_exposure
                    )
                    for frame in (input_dn, a3_dn, candidate_dn)
                ]
                labels = (
                    f"Input | clip {clip_index + 1} | frame {source_index}",
                    f"A3 baseline | clip {clip_index + 1} | frame {source_index}",
                    f"A8 sc450ai | clip {clip_index + 1} | frame {source_index}",
                )
                cells = [
                    _comparison_cell(image, text, crop)
                    for image, text in zip(images, labels, strict=True)
                ]
                frame = np.concatenate(cells, axis=1)
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
        raise RuntimeError("no video frame was rendered")
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
        "layout": (
            "Input | A3 baseline | A8 candidate; overview above fitted "
            "black-subject crop"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a8_sensor_adaptation/sc450ai/training_20261013/checkpoint_step_000500.pth",
    )
    parser.add_argument(
        "--cross-sensor-config",
        type=Path,
        default=REPO_ROOT / "configs/a3_cross_sensor_extension_20261012.json",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "artifacts/a3_cross_sensor_extension_20261012",
    )
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--a3-native-prediction",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/native_1000x/native_prediction_1000x.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a8_sensor_adaptation/sc450ai/evaluation_step500_20261013",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    evaluation_path = args.output_dir / "evaluation.json"
    if evaluation_path.is_file():
        raise FileExistsError(f"completed evaluation already exists: {evaluation_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_specs(args.cross_sensor_config)
    train_source = specs["sc450ai_804x_lightchanging"]
    train_tail_spec = _single_clip_spec(train_source, 142)
    holdout_spec = specs["sc450ai_1435x_motion_black_object"]
    train_dir = (
        args.baseline_root
        / "sequences/sc450ai/sc450ai_804x_lightchanging"
    )
    holdout_dir = (
        args.baseline_root
        / "sequences/sc450ai/sc450ai_1435x_motion_black_object"
    )
    train_profile = _load_profile(train_dir / "profile")
    holdout_profile = _load_profile(holdout_dir / "profile")
    device = torch.device(args.device)
    model = load_checkpoint_model(args.candidate_checkpoint, device, ema=True)
    try:
        train_tail_prediction = (
            args.output_dir / "train_domain_tail/inference/prediction_stress.npy"
        )
        infer_stress_clips(
            model,
            train_tail_spec,
            train_profile,
            device,
            train_tail_prediction,
            tile_batch=1,
            resume_key=f"{_sha256(args.candidate_checkpoint)}:train-tail",
        )
        holdout_prediction = (
            args.output_dir / "dynamic_holdout/inference/prediction_stress.npy"
        )
        infer_stress_clips(
            model,
            holdout_spec,
            holdout_profile,
            device,
            holdout_prediction,
            tile_batch=1,
            resume_key=f"{_sha256(args.candidate_checkpoint)}:holdout",
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_train_tail = _candidate_stress_metrics(
        evaluate_stress_prediction(
            train_tail_spec, train_profile, train_tail_prediction
        )
    )
    baseline_train_tail_path = _slice_baseline_tail(
        train_dir / "inference/prediction_stress.npy",
        args.output_dir / "train_domain_tail/baseline_a3/prediction_stress.npy",
    )
    baseline_train_tail = evaluate_stress_prediction(
        train_tail_spec, train_profile, baseline_train_tail_path
    )
    candidate_holdout = _candidate_stress_metrics(
        evaluate_stress_prediction(holdout_spec, holdout_profile, holdout_prediction)
    )
    baseline_holdout = _load_json(holdout_dir / "metrics.json")
    baseline_holdout = {
        key: value
        for key, value in baseline_holdout.items()
        if key not in ("panel_manifest", "panel_count")
    }
    replay_guard = _mis20s1_guard(
        args.candidate_checkpoint,
        args.a3_native_prediction,
        args.supervision_dir,
        args.output_dir / "mis20s1_replay_guard",
        device,
    )
    automatic_gate = _automatic_gate(
        baseline_train_tail,
        candidate_train_tail,
        baseline_holdout,
        candidate_holdout,
        replay_guard,
    )
    video = _render_video(
        holdout_spec,
        holdout_profile,
        holdout_dir / "inference/prediction_stress.npy",
        holdout_prediction,
        args.output_dir / "fixed_isp_video",
        args.fps,
    )
    result = {
        "protocol": "A8 sc450ai blind-spot adaptation evaluation",
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": _sha256(args.candidate_checkpoint),
        "weights": "ema",
        "training_sequence": "sc450ai_804x_lightchanging frames 0-119",
        "train_domain_validation": "sc450ai_804x_lightchanging frames 142-149",
        "held_out_sequence": "sc450ai_1435x_motion_black_object",
        "clean_target_available": False,
        "train_domain_tail": {
            "baseline_a3": baseline_train_tail,
            "candidate": candidate_train_tail,
        },
        "dynamic_holdout": {
            "baseline_a3": baseline_holdout,
            "candidate": candidate_holdout,
        },
        "mis20s1_replay_guard": replay_guard,
        "automatic_gate": automatic_gate,
        "fixed_isp_video": video,
    }
    _save_json(evaluation_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

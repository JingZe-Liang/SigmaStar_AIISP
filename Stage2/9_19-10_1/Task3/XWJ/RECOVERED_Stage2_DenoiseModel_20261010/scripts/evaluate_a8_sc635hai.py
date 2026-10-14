#!/usr/bin/env python3
"""Evaluate the sc635hai A8 candidate and export a fixed-ISP comparison video."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.a4 import (  # noqa: E402
    _read_aligned_video,
    evaluate_outputs,
    infer_video,
    load_checkpoint_model,
)
from raw_fusion.finetune.cross_sensor import (  # noqa: E402
    NativeReference,
    NativeSequenceSpec,
    NativeStressProfile,
    channel_vector,
    contiguous_clip_indices,
    denormalise_native,
    evaluate_disjoint_prediction,
    evaluate_stress_prediction,
    infer_disjoint_clips,
    infer_stress_clips,
    open_sequence,
    read_normalized_clip,
    render_fixed_isp,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _load_specs(path: Path) -> dict[str, NativeSequenceSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: NativeSequenceSpec.from_mapping(name, mapping)
        for name, mapping in payload["sequences"].items()
    }


def _load_reference(directory: Path) -> NativeReference:
    report = _load_json(directory / "report.json")
    return NativeReference(
        mean=np.load(directory / "full_mean_dn.npy", mmap_mode="r"),
        even_mean=np.load(directory / "even_mean_dn.npy", mmap_mode="r"),
        odd_mean=np.load(directory / "odd_mean_dn.npy", mmap_mode="r"),
        confidence=np.load(directory / "confidence.npy", mmap_mode="r"),
        even_confidence=np.load(directory / "even_confidence.npy", mmap_mode="r"),
        odd_confidence=np.load(directory / "odd_confidence.npy", mmap_mode="r"),
        valid_region=np.load(directory / "valid_region.npy", mmap_mode="r"),
        black_levels={
            str(name): float(value) for name, value in report["black_levels"].items()
        },
        frame_means=np.load(directory / "frame_means.npy", mmap_mode="r"),
        motion_fraction_per_frame=np.load(
            directory / "motion_fraction_per_frame.npy", mmap_mode="r"
        ),
        motion_threshold_per_frame=np.load(
            directory / "motion_threshold_per_frame.npy", mmap_mode="r"
        ),
        saturation_fraction=float(report["saturation_fraction"]),
    )


def _load_profile(directory: Path) -> NativeStressProfile:
    report = _load_json(directory / "report.json")
    return NativeStressProfile(
        temporal_mean=np.load(
            directory / "temporal_mean_display_only_dn.npy", mmap_mode="r"
        ),
        black_levels={
            str(name): float(value) for name, value in report["black_levels"].items()
        },
        frame_means=np.load(directory / "frame_means.npy", mmap_mode="r"),
        saturation_fraction=float(report["saturation_fraction"]),
    )


def _candidate_disjoint_metrics(metrics: dict[str, object]) -> dict[str, object]:
    result = dict(metrics)
    result["candidate"] = result.pop("a3")
    for old, new in (
        ("a3_psnr_gain_db", "candidate_psnr_gain_db"),
        ("a3_temporal_mad_dn", "candidate_temporal_mad_dn"),
        ("a3_output_bias_dn", "candidate_output_bias_dn"),
        ("a3_high_frequency_ratio", "candidate_high_frequency_ratio"),
        ("a3_flat_residual_sigma_dn", "candidate_flat_residual_sigma_dn"),
    ):
        result[new] = result.pop(old)
    return result


def _candidate_stress_metrics(metrics: dict[str, object]) -> dict[str, object]:
    result = dict(metrics)
    result["candidate_temporal_mad_dn"] = result.pop("a3_temporal_mad_dn")
    return result


def _static_summary(
    even_metrics: dict[str, object], odd_metrics: dict[str, object]
) -> dict[str, float]:
    directions = (even_metrics, odd_metrics)
    return {
        "mean_candidate_psnr_db": float(
            np.mean([item["candidate"]["psnr_db"] for item in directions])
        ),
        "mean_candidate_psnr_gain_db": float(
            np.mean([item["candidate_psnr_gain_db"] for item in directions])
        ),
        "mean_candidate_high_frequency_ratio": float(
            np.mean([item["candidate_high_frequency_ratio"] for item in directions])
        ),
        "maximum_candidate_output_bias_abs_dn": max(
            abs(float(item["candidate_output_bias_dn"])) for item in directions
        ),
        "minimum_legal_raw_fraction": min(
            float(item["legal_raw_fraction"]) for item in directions
        ),
        "maximum_normalized_clip_fraction": max(
            float(item["normalized_clip_fraction"]) for item in directions
        ),
    }


def _slice_baseline_static(
    source_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for direction in ("even_input", "odd_input"):
        source = np.load(
            source_dir / f"prediction_{direction}.npy", mmap_mode="r"
        )
        path = output_dir / f"prediction_{direction}.npy"
        np.save(path, np.asarray(source[-8:], dtype=np.float16))
        outputs.append(path)
    return outputs[0], outputs[1]


def _mis20s1_guard(
    candidate_checkpoint: Path,
    a3_prediction_path: Path,
    supervision_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    video, overlap = _read_aligned_video(
        supervision_dir, "1000x", 8, start_frame=0
    )
    reference = np.load(
        supervision_dir / "references/04_shdarkroom_10x_clean_reference.npy",
        mmap_mode="r",
    )
    report = _load_json(supervision_dir / "references/a1_reference_report.json")
    black = _channel_vector(report["black_levels"])
    manifest = _load_json(
        supervision_dir / "stream_manifests/04_shdarkroom_1000x.json"
    )
    shift_x = int(manifest["registration"]["integer_shift_x"])
    shift_y = int(manifest["registration"]["integer_shift_y"])
    source_motion = np.load(
        supervision_dir / "motion_masks/04_shdarkroom_1000x.npy", mmap_mode="r"
    )[:8, 0, ::2, ::2]
    motion = np.ones_like(source_motion, dtype=np.float32)
    height, width = source_motion.shape[1:]
    y0, y1 = max(0, -shift_y), min(height, height - shift_y)
    x0, x1 = max(0, -shift_x), min(width, width - shift_x)
    motion[:, y0:y1, x0:x1] = source_motion[
        :, y0 + shift_y : y1 + shift_y, x0 + shift_x : x1 + shift_x
    ]

    model = load_checkpoint_model(candidate_checkpoint, device, ema=True)
    try:
        candidate_prediction = infer_video(model, video, device, tile_batch=1)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "candidate_prediction_1000x_frames_0000_0007.npy",
        candidate_prediction.astype(np.float16),
    )
    a3_prediction = np.asarray(
        np.load(a3_prediction_path, mmap_mode="r")[:8], dtype=np.float32
    )
    candidate = evaluate_outputs(
        candidate_prediction,
        video,
        reference,
        overlap,
        motion,
        black,
    )
    baseline = evaluate_outputs(
        a3_prediction,
        video,
        reference,
        overlap,
        motion,
        black,
    )
    return {
        "frames": [0, 7],
        "baseline_a3": baseline,
        "candidate": candidate,
        "delta": {
            "psnr_db": candidate["finetuned"]["psnr_db"]
            - baseline["finetuned"]["psnr_db"],
            "high_frequency_ratio": candidate["high_frequency_ratio_mean"]
            - baseline["high_frequency_ratio_mean"],
            "temporal_mad_dn": candidate["temporal_mad_dn"]
            - baseline["temporal_mad_dn"],
        },
    }


def _video_encoder() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    available = result.stdout + result.stderr
    if " libx264 " in available:
        return "libx264"
    if " libx265 " in available:
        return "libx265"
    return "mpeg4"


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
    cell_width: int = 640,
) -> np.ndarray:
    overview_height = int(round(image.shape[0] * cell_width / image.shape[1]))
    overview = cv2.resize(image, (cell_width, overview_height), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = crop
    detail = image[y0:y1, x0:x1]
    detail_height = int(round(detail.shape[0] * cell_width / detail.shape[1]))
    detail = cv2.resize(detail, (cell_width, detail_height), interpolation=cv2.INTER_AREA)
    separator = np.full((4, cell_width, 3), 24, dtype=np.uint8)
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
    crop = dict(spec.panel_crops)["center_detail"]
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "sc635hai_16x_input_a3_a8_fixed_isp.mp4"
    preview_path = output_dir / "sc635hai_16x_input_a3_a8_fixed_isp_frame0000.png"
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
                    render_fixed_isp(frame, black, spec.fixed_isp_wb, spec.fixed_isp_exposure)
                    for frame in (input_dn, a3_dn, candidate_dn)
                ]
                labels = (
                    f"Input | clip {clip_index + 1} | frame {source_index}",
                    f"A3 baseline | clip {clip_index + 1} | frame {source_index}",
                    f"A8 sc635hai | clip {clip_index + 1} | frame {source_index}",
                )
                cells = [
                    _comparison_cell(image, label, crop)
                    for image, label in zip(images, labels, strict=True)
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
    if not cv2.imwrite(str(preview_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {preview_path}")
    return {
        "frames": offset,
        "fps": fps,
        "encoder": encoder,
        "video": str(video_path.resolve()),
        "video_sha256": _sha256(video_path),
        "preview": str(preview_path.resolve()),
        "preview_sha256": _sha256(preview_path),
        "layout": "Input | A3 baseline | A8 candidate; overview above center-detail crop",
    }


def _automatic_gate(
    baseline_static: dict[str, object],
    candidate_static: dict[str, object],
    baseline_dynamic: dict[str, object],
    candidate_dynamic: dict[str, object],
    replay_guard: dict[str, object],
) -> dict[str, object]:
    baseline_static_directions = (
        baseline_static["even_input_vs_odd_reference"],
        baseline_static["odd_input_vs_even_reference"],
    )
    baseline_static_psnr = float(
        np.mean([item["a3"]["psnr_db"] for item in baseline_static_directions])
    )
    baseline_static_hf = float(
        np.mean([item["a3_high_frequency_ratio"] for item in baseline_static_directions])
    )
    checks = {
        "static_psnr_not_worse_than_a3_minus_0p10_db": (
            candidate_static["summary"]["mean_candidate_psnr_db"]
            >= baseline_static_psnr - 0.10
        ),
        "static_hf_distance_to_one_not_worse_by_0p02": (
            abs(candidate_static["summary"]["mean_candidate_high_frequency_ratio"] - 1.0)
            <= abs(baseline_static_hf - 1.0) + 0.02
        ),
        "dynamic_legal_raw_fraction": candidate_dynamic["legal_raw_fraction"] == 1.0,
        "dynamic_clip_fraction_guard": (
            candidate_dynamic["normalized_clip_fraction"]
            <= baseline_dynamic["normalized_clip_fraction"] + 0.005
        ),
        "dynamic_high_frequency_guard": (
            candidate_dynamic["spatial_high_frequency_ratio"]
            >= baseline_dynamic["spatial_high_frequency_ratio"] - 0.03
        ),
        "dynamic_motion_guard": (
            candidate_dynamic["motion_temporal_ratio"]
            >= baseline_dynamic["motion_temporal_ratio"] - 0.05
        ),
        "dynamic_static_flicker_guard": (
            candidate_dynamic["static_temporal_ratio"]
            <= baseline_dynamic["static_temporal_ratio"] + 0.05
        ),
        "mis20s1_psnr_guard": replay_guard["delta"]["psnr_db"] >= -0.10,
        "mis20s1_high_frequency_guard": (
            replay_guard["delta"]["high_frequency_ratio"] >= -0.03
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "note": "The dynamic sequence has no clean target; visual review remains mandatory.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a8_sensor_adaptation/sc635hai/training_20261013/checkpoint_step_000500.pth",
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
        / "artifacts/a8_sensor_adaptation/sc635hai/evaluation_step500_20261013",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    if (args.output_dir / "evaluation.json").is_file():
        raise FileExistsError(
            f"completed evaluation already exists: {args.output_dir / 'evaluation.json'}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_specs(args.cross_sensor_config)
    static_source = specs["sc635hai_131x_low_cct"]
    static_spec = NativeSequenceSpec(
        name=static_source.name,
        sensor=static_source.sensor,
        path=static_source.path,
        width=static_source.width,
        height=static_source.height,
        frames_used=static_source.frames_used,
        bit_shift=static_source.bit_shift,
        cfa=static_source.cfa,
        nominal_black_dn=static_source.nominal_black_dn,
        reference_mode=static_source.reference_mode,
        clip_starts=(134,),
        fixed_isp_wb=static_source.fixed_isp_wb,
        fixed_isp_exposure=static_source.fixed_isp_exposure,
        panel_crops=static_source.panel_crops,
    )
    dynamic_spec = specs["sc635hai_16x_motion_color"]
    static_dir = (
        args.baseline_root
        / "sequences/sc635hai/sc635hai_131x_low_cct"
    )
    dynamic_dir = (
        args.baseline_root
        / "sequences/sc635hai/sc635hai_16x_motion_color"
    )
    reference = _load_reference(static_dir / "reference")
    profile = _load_profile(dynamic_dir / "profile")
    device = torch.device(args.device)
    model = load_checkpoint_model(args.candidate_checkpoint, device, ema=True)
    try:
        static_inference_dir = args.output_dir / "static_tail/inference"
        static_even = static_inference_dir / "prediction_even_input.npy"
        static_odd = static_inference_dir / "prediction_odd_input.npy"
        infer_disjoint_clips(
            model,
            static_spec,
            reference,
            0,
            device,
            static_even,
            tile_batch=1,
            resume_key=f"{_sha256(args.candidate_checkpoint)}:static:even",
        )
        infer_disjoint_clips(
            model,
            static_spec,
            reference,
            1,
            device,
            static_odd,
            tile_batch=1,
            resume_key=f"{_sha256(args.candidate_checkpoint)}:static:odd",
        )
        dynamic_prediction = args.output_dir / "dynamic_holdout/inference/prediction_stress.npy"
        infer_stress_clips(
            model,
            dynamic_spec,
            profile,
            device,
            dynamic_prediction,
            tile_batch=1,
            resume_key=f"{_sha256(args.candidate_checkpoint)}:dynamic",
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_even = _candidate_disjoint_metrics(
        evaluate_disjoint_prediction(static_spec, reference, 0, static_even)
    )
    candidate_odd = _candidate_disjoint_metrics(
        evaluate_disjoint_prediction(static_spec, reference, 1, static_odd)
    )
    candidate_static = {
        "protocol": "A8 train-domain tail diagnostic",
        "even_input_vs_odd_reference": candidate_even,
        "odd_input_vs_even_reference": candidate_odd,
        "summary": _static_summary(candidate_even, candidate_odd),
    }
    baseline_cache_dir = args.output_dir / "static_tail/baseline_a3"
    baseline_even_path, baseline_odd_path = _slice_baseline_static(
        static_dir / "inference", baseline_cache_dir
    )
    baseline_static = {
        "protocol": "A3 baseline on the same tail clips",
        "even_input_vs_odd_reference": evaluate_disjoint_prediction(
            static_spec, reference, 0, baseline_even_path
        ),
        "odd_input_vs_even_reference": evaluate_disjoint_prediction(
            static_spec, reference, 1, baseline_odd_path
        ),
    }
    candidate_dynamic = _candidate_stress_metrics(
        evaluate_stress_prediction(dynamic_spec, profile, dynamic_prediction)
    )
    baseline_dynamic = _load_json(dynamic_dir / "metrics.json")
    baseline_dynamic = {
        key: value
        for key, value in baseline_dynamic.items()
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
        baseline_static,
        candidate_static,
        baseline_dynamic,
        candidate_dynamic,
        replay_guard,
    )
    video = _render_video(
        dynamic_spec,
        profile,
        dynamic_dir / "inference/prediction_stress.npy",
        dynamic_prediction,
        args.output_dir / "fixed_isp_video",
        args.fps,
    )
    result = {
        "protocol": "A8 sc635hai adaptation evaluation",
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": _sha256(args.candidate_checkpoint),
        "weights": "ema",
        "training_sequence": "sc635hai_131x_low_cct",
        "held_out_sequence": "sc635hai_16x_motion_color",
        "static_train_domain": {
            "baseline_a3": baseline_static,
            "candidate": candidate_static,
        },
        "dynamic_holdout": {
            "baseline_a3": baseline_dynamic,
            "candidate": candidate_dynamic,
        },
        "mis20s1_replay_guard": replay_guard,
        "automatic_gate": automatic_gate,
        "fixed_isp_video": video,
    }
    _save_json(args.output_dir / "evaluation.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the deterministic A3 zero-shot audit on explicitly described sensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a4 import load_checkpoint_model  # noqa: E402
from raw_fusion.finetune.cross_sensor import (  # noqa: E402
    TEMPORAL_FRAMES,
    NativeReference,
    NativeSequenceSpec,
    NativeStressProfile,
    build_motion_aware_reference,
    build_stress_profile,
    channel_vector,
    contiguous_clip_indices,
    denormalise_native,
    evaluate_disjoint_prediction,
    evaluate_stress_prediction,
    infer_disjoint_clips,
    infer_stress_clips,
    make_fixed_isp_panel,
    open_sequence,
    read_normalized_clip,
    split_clip_indices,
)
from raw_fusion.finetune.reference import _save_array, _save_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, NativeSequenceSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = {
        name: NativeSequenceSpec.from_mapping(name, sequence)
        for name, sequence in payload["sequences"].items()
    }
    return payload, sequences


def _reference_paths(sequence_dir: Path) -> dict[str, Path]:
    reference_dir = sequence_dir / "reference"
    return {
        "mean": reference_dir / "full_mean_dn.npy",
        "even_mean": reference_dir / "even_mean_dn.npy",
        "odd_mean": reference_dir / "odd_mean_dn.npy",
        "confidence": reference_dir / "confidence.npy",
        "even_confidence": reference_dir / "even_confidence.npy",
        "odd_confidence": reference_dir / "odd_confidence.npy",
        "valid_region": reference_dir / "valid_region.npy",
        "frame_means": reference_dir / "frame_means.npy",
        "motion_fraction": reference_dir / "motion_fraction_per_frame.npy",
        "motion_threshold": reference_dir / "motion_threshold_per_frame.npy",
        "report": reference_dir / "report.json",
    }


def _profile_paths(sequence_dir: Path) -> dict[str, Path]:
    profile_dir = sequence_dir / "profile"
    return {
        "temporal_mean": profile_dir / "temporal_mean_display_only_dn.npy",
        "frame_means": profile_dir / "frame_means.npy",
        "report": profile_dir / "report.json",
    }


def _save_reference(
    sequence_dir: Path,
    reference: NativeReference,
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    paths = _reference_paths(sequence_dir)
    arrays = {
        "mean": reference.mean,
        "even_mean": reference.even_mean,
        "odd_mean": reference.odd_mean,
        "confidence": reference.confidence,
        "even_confidence": reference.even_confidence,
        "odd_confidence": reference.odd_confidence,
        "valid_region": reference.valid_region,
        "frame_means": reference.frame_means,
        "motion_fraction": reference.motion_fraction_per_frame,
        "motion_threshold": reference.motion_threshold_per_frame,
    }
    for name, array in arrays.items():
        _save_array(paths[name], array)
    report = {
        **report,
        "protocol": "cross-sensor motion-aware native reference",
        "provenance": provenance,
        "outputs": {
            name: {
                "path": str(paths[name].resolve()),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _sha256(paths[name]),
            }
            for name, array in arrays.items()
        },
    }
    _save_json(paths["report"], report)


def _load_reference(sequence_dir: Path) -> tuple[NativeReference, dict[str, Any]]:
    paths = _reference_paths(sequence_dir)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    black = {str(name): float(value) for name, value in report["black_levels"].items()}
    reference = NativeReference(
        mean=np.load(paths["mean"], mmap_mode="r"),
        even_mean=np.load(paths["even_mean"], mmap_mode="r"),
        odd_mean=np.load(paths["odd_mean"], mmap_mode="r"),
        confidence=np.load(paths["confidence"], mmap_mode="r"),
        even_confidence=np.load(paths["even_confidence"], mmap_mode="r"),
        odd_confidence=np.load(paths["odd_confidence"], mmap_mode="r"),
        valid_region=np.load(paths["valid_region"], mmap_mode="r"),
        black_levels=black,
        frame_means=np.load(paths["frame_means"], mmap_mode="r"),
        motion_fraction_per_frame=np.load(paths["motion_fraction"], mmap_mode="r"),
        motion_threshold_per_frame=np.load(paths["motion_threshold"], mmap_mode="r"),
        saturation_fraction=float(report["saturation_fraction"]),
    )
    return reference, report


def _save_profile(
    sequence_dir: Path,
    profile: NativeStressProfile,
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    paths = _profile_paths(sequence_dir)
    arrays = {
        "temporal_mean": profile.temporal_mean,
        "frame_means": profile.frame_means,
    }
    for name, array in arrays.items():
        _save_array(paths[name], array)
    _save_json(
        paths["report"],
        {
            **report,
            "protocol": "cross-sensor no-reference stress profile",
            "provenance": provenance,
            "outputs": {
                name: {
                    "path": str(paths[name].resolve()),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": _sha256(paths[name]),
                }
                for name, array in arrays.items()
            },
        },
    )


def _load_profile(sequence_dir: Path) -> tuple[NativeStressProfile, dict[str, Any]]:
    paths = _profile_paths(sequence_dir)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    profile = NativeStressProfile(
        temporal_mean=np.load(paths["temporal_mean"], mmap_mode="r"),
        black_levels={
            str(name): float(value) for name, value in report["black_levels"].items()
        },
        frame_means=np.load(paths["frame_means"], mmap_mode="r"),
        saturation_fraction=float(report["saturation_fraction"]),
    )
    return profile, report


def _matching_reference_cache(sequence_dir: Path, provenance: dict[str, Any]) -> bool:
    paths = _reference_paths(sequence_dir)
    if not all(path.is_file() for path in paths.values()):
        return False
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    return report.get("provenance") == provenance


def _matching_profile_cache(sequence_dir: Path, provenance: dict[str, Any]) -> bool:
    paths = _profile_paths(sequence_dir)
    if not all(path.is_file() for path in paths.values()):
        return False
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    return report.get("provenance") == provenance


def _render_panels(
    spec: NativeSequenceSpec,
    sequence_dir: Path,
    reference: NativeReference,
) -> dict[str, Any]:
    reader = open_sequence(spec)
    black = channel_vector(reference.black_levels)
    panel_dir = sequence_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for parity in (0, 1):
        direction = "even_input" if parity == 0 else "odd_input"
        target = reference.odd_mean if parity == 0 else reference.even_mean
        prediction_path = sequence_dir / "inference" / f"prediction_{direction}.npy"
        prediction = np.load(prediction_path, mmap_mode="r")
        rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for clip_index, indices in enumerate(split_clip_indices(spec, parity)):
            frame_in_clip = TEMPORAL_FRAMES // 2
            input_normalized = read_normalized_clip(reader, spec.adapter, indices, black)
            prediction_normalized = np.asarray(
                prediction[clip_index * TEMPORAL_FRAMES + frame_in_clip],
                dtype=np.float32,
            )
            rows.append(
                (
                    f"raw frame {indices[frame_in_clip]}",
                    denormalise_native(input_normalized[frame_in_clip], black),
                    denormalise_native(prediction_normalized, black),
                    target,
                )
            )
        panel_specs: list[tuple[str, tuple[int, int, int, int] | None]] = [
            (f"{direction}_overview", None)
        ]
        panel_specs.extend(
            (f"{direction}_{crop_name}", crop)
            for crop_name, crop in spec.panel_crops
        )
        for panel_name, crop in panel_specs:
            panel = make_fixed_isp_panel(
                rows,
                black,
                spec.fixed_isp_wb,
                spec.fixed_isp_exposure,
                crop=crop,
            )
            path = panel_dir / f"{panel_name}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                raise OSError(f"failed to write {path}")
            outputs[panel_name] = {
                "path": str(path.resolve()),
                "shape": list(panel.shape),
                "sha256": _sha256(path),
                "crop_sensor_xyxy": list(crop) if crop is not None else None,
            }
    manifest = {
        "fixed_isp": {
            "black_levels": reference.black_levels,
            "white_level_dn": 4095.0,
            "wb_packed_R_Gr_Gb_B": list(spec.fixed_isp_wb),
            "exposure": spec.fixed_isp_exposure,
            "transfer": "linear Bayer demosaic then IEC-like sRGB transfer",
        },
        "panels": outputs,
    }
    _save_json(panel_dir / "panel_manifest.json", manifest)
    return manifest


def _render_stress_panels(
    spec: NativeSequenceSpec,
    sequence_dir: Path,
    profile: NativeStressProfile,
) -> dict[str, Any]:
    reader = open_sequence(spec)
    black = channel_vector(profile.black_levels)
    prediction_path = sequence_dir / "inference/prediction_stress.npy"
    prediction = np.load(prediction_path, mmap_mode="r")
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for clip_index, indices in enumerate(contiguous_clip_indices(spec)):
        frame_in_clip = TEMPORAL_FRAMES // 2
        next_frame = min(frame_in_clip + 1, TEMPORAL_FRAMES - 1)
        input_normalized = read_normalized_clip(reader, spec.adapter, indices, black)
        offset = clip_index * TEMPORAL_FRAMES
        prediction_current = np.asarray(
            prediction[offset + frame_in_clip], dtype=np.float32
        )
        prediction_next = np.asarray(prediction[offset + next_frame], dtype=np.float32)
        rows.append(
            (
                f"raw frame {indices[frame_in_clip]}",
                denormalise_native(input_normalized[frame_in_clip], black),
                denormalise_native(prediction_current, black),
                denormalise_native(prediction_next, black),
            )
        )

    panel_dir = sequence_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_specs: list[tuple[str, tuple[int, int, int, int] | None]] = [
        ("stress_overview", None)
    ]
    panel_specs.extend(
        (f"stress_{crop_name}", crop) for crop_name, crop in spec.panel_crops
    )
    outputs: dict[str, dict[str, Any]] = {}
    for panel_name, crop in panel_specs:
        panel = make_fixed_isp_panel(
            rows,
            black,
            spec.fixed_isp_wb,
            spec.fixed_isp_exposure,
            crop=crop,
            column_labels=("Input t", "A3 t", "A3 t+1"),
        )
        path = panel_dir / f"{panel_name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {path}")
        outputs[panel_name] = {
            "path": str(path.resolve()),
            "shape": list(panel.shape),
            "sha256": _sha256(path),
            "crop_sensor_xyxy": list(crop) if crop is not None else None,
        }
    manifest = {
        "fixed_isp": {
            "black_levels": profile.black_levels,
            "white_level_dn": 4095.0,
            "wb_packed_R_Gr_Gb_B": list(spec.fixed_isp_wb),
            "exposure": spec.fixed_isp_exposure,
            "transfer": "linear Bayer demosaic then IEC-like sRGB transfer",
        },
        "clean_target_available": False,
        "columns": ["Input t", "A3 t", "A3 t+1"],
        "panels": outputs,
    }
    _save_json(panel_dir / "panel_manifest.json", manifest)
    return manifest


def _prediction_cache_matches(
    sequence_dir: Path,
    provenance: dict[str, Any],
    spec: NativeSequenceSpec,
) -> bool:
    manifest_path = sequence_dir / "inference" / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("provenance") != provenance:
        return False
    expected = (len(spec.clip_starts) * TEMPORAL_FRAMES, 4, *spec.packed_shape)
    if spec.reference_mode == "motion_aware_split_half":
        names = ("prediction_even_input.npy", "prediction_odd_input.npy")
    else:
        names = ("prediction_stress.npy",)
    for name in names:
        path = sequence_dir / "inference" / name
        if not path.is_file() or np.load(path, mmap_mode="r").shape != expected:
            return False
    return True


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    directions = [metrics["even_input_vs_odd_reference"], metrics["odd_input_vs_even_reference"]]
    return {
        "minimum_a3_psnr_gain_db": min(item["a3_psnr_gain_db"] for item in directions),
        "mean_a3_psnr_gain_db": float(np.mean([item["a3_psnr_gain_db"] for item in directions])),
        "maximum_a3_output_bias_abs_dn": max(abs(item["a3_output_bias_dn"]) for item in directions),
        "minimum_legal_raw_fraction": min(item["legal_raw_fraction"] for item in directions),
        "maximum_normalized_clip_fraction": max(item["normalized_clip_fraction"] for item in directions),
        "a3_temporal_mad_ratio_mean": float(
            np.mean(
                [
                    item["a3_temporal_mad_dn"] / max(item["input_temporal_mad_dn"], 1e-8)
                    for item in directions
                ]
            )
        ),
        "a3_high_frequency_ratio_range": [
            min(item["a3_high_frequency_ratio"] for item in directions),
            max(item["a3_high_frequency_ratio"] for item in directions),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a3_cross_sensor_zero_shot_20261012.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/a3_cross_sensor_zero_shot_20261012",
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument(
        "--phase", choices=("reference", "inference", "metrics", "all"), default="all"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-batch", type=int, default=2)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args()

    config, sequences = _load_config(args.config)
    selected_names = args.sequence or list(sequences)
    unknown = sorted(set(selected_names) - set(sequences))
    if unknown:
        raise ValueError(f"unknown sequences: {unknown}")
    checkpoint = REPO_ROOT / config["checkpoint"]
    config_sha256 = _sha256(args.config)
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != config["checkpoint_sha256"]:
        raise ValueError("A3 checkpoint SHA-256 does not match the protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(
        args.output_dir / "protocol.json",
        {
            "config_path": str(args.config.resolve()),
            "config_sha256": config_sha256,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "config": config,
        },
    )

    model: torch.nn.Module | None = None
    device = torch.device(args.device)
    overall: dict[str, Any] = {"protocol_sha256": config_sha256, "sequences": {}}
    for name in selected_names:
        spec = sequences[name]
        sequence_dir = args.output_dir / "sequences" / spec.sensor / spec.name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        source_sha256 = _sha256(spec.path)
        source_provenance = {
            "protocol_sha256": config_sha256,
            "source_path": str(spec.path.resolve()),
            "source_size_bytes": spec.path.stat().st_size,
            "source_sha256": source_sha256,
            "width": spec.width,
            "height": spec.height,
            "frames_used": spec.frames_used,
            "bit_shift": spec.bit_shift,
            "cfa": spec.cfa,
        }
        _save_json(sequence_dir / "source_manifest.json", source_provenance)

        reference: NativeReference | None = None
        profile: NativeStressProfile | None = None
        if spec.reference_mode == "motion_aware_split_half":
            if args.phase in ("reference", "all") and (
                args.force_reference
                or not _matching_reference_cache(sequence_dir, source_provenance)
            ):
                reference, data_report = build_motion_aware_reference(
                    spec, chunk_frames=args.chunk_frames
                )
                _save_reference(sequence_dir, reference, data_report, source_provenance)
            reference, data_report = _load_reference(sequence_dir)
            data_report_path = _reference_paths(sequence_dir)["report"]
        else:
            if args.phase in ("reference", "all") and (
                args.force_reference
                or not _matching_profile_cache(sequence_dir, source_provenance)
            ):
                profile, data_report = build_stress_profile(
                    spec, chunk_frames=args.chunk_frames
                )
                _save_profile(sequence_dir, profile, data_report, source_provenance)
            profile, data_report = _load_profile(sequence_dir)
            data_report_path = _profile_paths(sequence_dir)["report"]

        inference_provenance = {
            **source_provenance,
            "checkpoint_sha256": checkpoint_sha256,
            "weights": "ema",
            "tile_batch": args.tile_batch,
        }
        resume_key = hashlib.sha256(
            json.dumps(inference_provenance, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if args.phase in ("inference", "all") and (
            args.force_inference
            or not _prediction_cache_matches(sequence_dir, inference_provenance, spec)
        ):
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            if model is None:
                model = load_checkpoint_model(checkpoint, device, ema=True)
            inference_dir = sequence_dir / "inference"
            directions: dict[str, Any] = {}
            if reference is not None:
                for parity, direction in ((0, "even_input"), (1, "odd_input")):
                    path = inference_dir / f"prediction_{direction}.npy"
                    directions[direction] = infer_disjoint_clips(
                        model,
                        spec,
                        reference,
                        parity,
                        device,
                        path,
                        tile_batch=args.tile_batch,
                        resume_key=f"{resume_key}:{direction}",
                    )
                    directions[direction]["path"] = str(path.resolve())
                    directions[direction]["sha256"] = _sha256(path)
            else:
                if profile is None:
                    raise RuntimeError(f"{spec.name}: stress profile was not loaded")
                path = inference_dir / "prediction_stress.npy"
                directions["stress"] = infer_stress_clips(
                    model,
                    spec,
                    profile,
                    device,
                    path,
                    tile_batch=args.tile_batch,
                    resume_key=f"{resume_key}:stress",
                )
                directions["stress"]["path"] = str(path.resolve())
                directions["stress"]["sha256"] = _sha256(path)
            _save_json(
                inference_dir / "manifest.json",
                {"provenance": inference_provenance, "directions": directions},
            )

        metrics: dict[str, Any] | None = None
        if args.phase in ("metrics", "all"):
            if not _prediction_cache_matches(sequence_dir, inference_provenance, spec):
                raise FileNotFoundError(f"{spec.name}: matching inference outputs are absent")
            if reference is not None:
                metrics = {
                    "protocol": "mutually exclusive native split-half zero-shot evaluation",
                    "even_input_vs_odd_reference": evaluate_disjoint_prediction(
                        spec,
                        reference,
                        0,
                        sequence_dir / "inference/prediction_even_input.npy",
                    ),
                    "odd_input_vs_even_reference": evaluate_disjoint_prediction(
                        spec,
                        reference,
                        1,
                        sequence_dir / "inference/prediction_odd_input.npy",
                    ),
                }
                metrics["summary"] = _summary(metrics)
                panel_manifest = _render_panels(spec, sequence_dir, reference)
            else:
                if profile is None:
                    raise RuntimeError(f"{spec.name}: stress profile was not loaded")
                metrics = evaluate_stress_prediction(
                    spec,
                    profile,
                    sequence_dir / "inference/prediction_stress.npy",
                )
                panel_manifest = _render_stress_panels(spec, sequence_dir, profile)
            metrics["panel_manifest"] = str(
                (sequence_dir / "panels/panel_manifest.json").resolve()
            )
            metrics["panel_count"] = len(panel_manifest["panels"])
            _save_json(sequence_dir / "metrics.json", metrics)

        overall["sequences"][name] = {
            "source_manifest": str((sequence_dir / "source_manifest.json").resolve()),
            "data_report": str(data_report_path.resolve()),
            "data_profile": data_report,
            "metrics": metrics,
        }
        _save_json(args.output_dir / "summary.json", overall)

    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

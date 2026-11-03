#!/usr/bin/env python3
"""Compare A3 uniform, weighted-tile, and frame-repeated high-gain inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a4 import load_checkpoint_model  # noqa: E402
from raw_fusion.finetune.all_dataset_preview import (  # noqa: E402
    PreviewSequence,
    discover_sequences,
    load_config,
    normalize_packed,
    save_json,
    sha256_file,
)
from raw_fusion.finetune.high_gain_diagnostic import (  # noqa: E402
    evaluate_variants,
    infer_frame_repeated,
    infer_tiled_clip,
    render_comparison_videos,
)
from raw_fusion.finetune.stream_io import RawStreamReader, pack_frame  # noqa: E402


def _select_sequences(
    catalog: list[PreviewSequence], selected: list[str]
) -> list[PreviewSequence]:
    by_name = {sequence.name: sequence for sequence in catalog}
    unknown = sorted(set(selected) - set(by_name))
    if unknown:
        raise ValueError(f"unknown sequences: {unknown}")
    return [by_name[name] for name in selected]


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array.astype(np.float16))
    temporary.replace(path)


def _load_input_clip(
    sequence: PreviewSequence, source_start: int, black: np.ndarray
) -> np.ndarray:
    reader = RawStreamReader(sequence.path, sequence.width, sequence.height)
    frames = [
        normalize_packed(
            pack_frame(reader.read_frame(source_start + offset), sequence.adapter), black
        )
        for offset in range(8)
    ]
    return np.stack(frames).astype(np.float32)


def _baseline_paths(
    config: dict[str, Any], sequence: PreviewSequence
) -> tuple[Path, Path, Path]:
    baseline_root = REPO_ROOT / config["baseline_preview_root"]
    sequence_dir = (
        baseline_root
        / "sequences"
        / sequence.group
        / sequence.category
        / sequence.name
    )
    return (
        sequence_dir / "runtime.json",
        sequence_dir / "inference/prediction.npy",
        sequence_dir / "inference/manifest.json",
    )


def _load_baseline_clip(
    config: dict[str, Any], sequence: PreviewSequence
) -> tuple[dict[str, Any], np.ndarray, int]:
    runtime_path, prediction_path, manifest_path = _baseline_paths(config, sequence)
    if not runtime_path.is_file() or not prediction_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{sequence.name}: baseline A3 artifacts are incomplete")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip_index = int(config["sampling"]["clip_index"])
    starts = tuple(int(value) for value in runtime["clip_starts"])
    if not 0 <= clip_index < len(starts):
        raise ValueError(f"{sequence.name}: clip_index {clip_index} is invalid")
    prediction = np.load(prediction_path, mmap_mode="r")
    offset = clip_index * 8
    baseline = np.asarray(prediction[offset : offset + 8], dtype=np.float32)
    expected = (8, 4, *sequence.packed_shape)
    if baseline.shape != expected:
        raise ValueError(f"{sequence.name}: expected {expected}, got {baseline.shape}")
    checkpoint_hash = str(manifest["provenance"]["checkpoint_sha256"])
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError(f"{sequence.name}: baseline checkpoint hash mismatch")
    return runtime, baseline, starts[clip_index]


def _resume_repeated_frames(
    model: torch.nn.Module,
    device: torch.device,
    input_clip: np.ndarray,
    output_path: Path,
    progress_path: Path,
    resume_key: str,
    inference: dict[str, Any],
) -> np.ndarray:
    shape = input_clip.shape
    completed = 0
    if output_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        candidate = np.load(output_path, mmap_mode="r+")
        if (
            progress.get("resume_key") == resume_key
            and progress.get("shape") == list(shape)
            and candidate.shape == shape
            and candidate.dtype == np.float16
        ):
            prediction = candidate
            completed = int(progress.get("completed_frames", 0))
        else:
            prediction = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float16, shape=shape
            )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=shape
        )
    for frame_index in range(completed, shape[0]):
        output = infer_frame_repeated(
            model,
            input_clip[frame_index],
            device,
            temporal_output_index=int(inference["temporal_output_index"]),
            tile=int(inference["tile"]),
            stride=int(inference["stride"]),
            tile_batch=int(inference["tile_batch"]),
            blend_mode=str(inference["blend_mode"]),
            blend_floor=float(inference["blend_floor"]),
        )
        prediction[frame_index] = output.astype(np.float16)
        prediction.flush()
        save_json(
            progress_path,
            {
                "resume_key": resume_key,
                "shape": list(shape),
                "completed_frames": frame_index + 1,
            },
        )
        print(
            json.dumps(
                {
                    "phase": "frame_repeated",
                    "completed": frame_index + 1,
                    "total": shape[0],
                }
            ),
            flush=True,
        )
    return np.asarray(prediction, dtype=np.float32)


def _run_sequence(
    model: torch.nn.Module | None,
    device: torch.device,
    config: dict[str, Any],
    config_hash: str,
    sequence: PreviewSequence,
    output_root: Path,
    phase: str,
) -> dict[str, Any]:
    runtime, baseline, source_start = _load_baseline_clip(config, sequence)
    black = np.asarray(runtime["black_levels_R_Gr_Gb_B"], dtype=np.float32)
    input_clip = _load_input_clip(sequence, source_start, black)
    sequence_dir = output_root / sequence.name
    inference_dir = sequence_dir / "inference"
    weighted_path = inference_dir / "a3_weighted.npy"
    repeated_path = inference_dir / "a3_frame_repeated.npy"
    progress_path = inference_dir / "a3_frame_repeated_progress.json"
    inference = config["inference"]
    provenance = {
        "sequence": sequence.to_mapping(),
        "source_start": source_start,
        "frames": 8,
        "checkpoint": str((REPO_ROOT / config["checkpoint"]).resolve()),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "config_sha256": config_hash,
        "weighted_tile": inference,
        "baseline": {"tile": 64, "stride": 56, "blend_mode": "uniform"},
        "dataset_mutated": False,
        "training_performed": False,
    }
    resume_key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if phase in ("inference", "all"):
        if model is None:
            raise RuntimeError("model is required for inference")
        inference_manifest_path = inference_dir / "manifest.json"
        existing_manifest = None
        if inference_manifest_path.is_file():
            existing_manifest = json.loads(
                inference_manifest_path.read_text(encoding="utf-8")
            )
        weighted_valid = (
            weighted_path.is_file()
            and existing_manifest is not None
            and existing_manifest.get("resume_key") == resume_key
            and existing_manifest.get("artifacts", {})
            .get("weighted", {})
            .get("sha256")
            == sha256_file(weighted_path)
        )
        if not weighted_valid:
            weighted = infer_tiled_clip(
                model,
                input_clip,
                device,
                tile=int(inference["tile"]),
                stride=int(inference["stride"]),
                tile_batch=int(inference["tile_batch"]),
                blend_mode=str(inference["blend_mode"]),
                blend_floor=float(inference["blend_floor"]),
            )
            _atomic_save_array(weighted_path, weighted)
            print(json.dumps({"phase": "weighted", "status": "complete"}), flush=True)
        _resume_repeated_frames(
            model,
            device,
            input_clip,
            repeated_path,
            progress_path,
            resume_key,
            inference,
        )
        save_json(
            inference_manifest_path,
            {
                "provenance": provenance,
                "resume_key": resume_key,
                "artifacts": {
                    "weighted": {
                        "path": str(weighted_path.resolve()),
                        "sha256": sha256_file(weighted_path),
                    },
                    "frame_repeated": {
                        "path": str(repeated_path.resolve()),
                        "sha256": sha256_file(repeated_path),
                    },
                },
            },
        )

    if not weighted_path.is_file() or not repeated_path.is_file():
        raise FileNotFoundError(f"{sequence.name}: diagnostic predictions are absent")
    weighted = np.asarray(np.load(weighted_path, mmap_mode="r"), dtype=np.float32)
    repeated = np.asarray(np.load(repeated_path, mmap_mode="r"), dtype=np.float32)
    variants = {
        "input": input_clip,
        "a3_uniform": baseline,
        "a3_weighted": weighted,
        "a3_frame_repeated": repeated,
    }
    evaluation = evaluate_variants(
        variants,
        {"baseline_64_56": (64, 56), "weighted_64_48": (64, 48)},
    )
    evaluation["sequence"] = sequence.to_mapping()
    evaluation["source_start"] = source_start
    save_json(sequence_dir / "evaluation.json", evaluation)

    video_manifest_path = sequence_dir / "traditional_isp_video/video_manifest.json"
    video_manifest = None
    if phase in ("render", "all"):
        video_manifest = render_comparison_videos(
            sequence_dir / "traditional_isp_video",
            sequence,
            source_start,
            black,
            float(runtime["fixed_exposure_ev"]),
            variants,
            {
                "input": "Input",
                "a3_uniform": "A3 uniform 64/56",
                "a3_weighted": "A3 weighted 64/48",
                "a3_frame_repeated": "A3 frame-repeat weighted 64/48",
            },
            config["isp"],
            config["video"],
        )
    elif video_manifest_path.is_file():
        video_manifest = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    return {
        "status": "complete",
        "source_start": source_start,
        "evaluation": str((sequence_dir / "evaluation.json").resolve()),
        "videos": video_manifest["videos"] if video_manifest else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a3_high_gain_diagnostic_20261018.json",
    )
    parser.add_argument(
        "--phase", choices=("inference", "evaluate", "render", "all"), default="all"
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    config_hash = sha256_file(args.config)
    catalog_config = load_config(REPO_ROOT / config["catalog_config"])
    selected_names = args.sequence or list(config["sequences"])
    sequences = _select_sequences(discover_sequences(catalog_config), selected_names)
    output_root = args.output_dir or REPO_ROOT / config["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = REPO_ROOT / config["checkpoint"]
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("A3 checkpoint SHA-256 does not match the frozen protocol")
    device = torch.device(args.device or config["inference"]["preferred_device"])
    model: torch.nn.Module | None = None
    if args.phase in ("inference", "all"):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("diagnostic inference requires an available CUDA device")
        model = load_checkpoint_model(checkpoint, device, ema=True)

    summary_path = output_root / "summary.json"
    existing_summary: dict[str, Any] = {}
    if summary_path.is_file():
        candidate = json.loads(summary_path.read_text(encoding="utf-8"))
        if candidate.get("checkpoint_sha256") == config["checkpoint_sha256"]:
            existing_summary = candidate
    summary: dict[str, Any] = {
        "status": "running",
        "phase": args.phase,
        "configured_device": str(device),
        "inference_device_used": (
            str(device) if args.phase in ("inference", "all") else None
        ),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "dataset_mutated": False,
        "training_performed": False,
        "sequences": dict(existing_summary.get("sequences", {})),
    }
    save_json(output_root / "protocol.json", {"config": config, "config_sha256": config_hash})
    try:
        for sequence in sequences:
            summary["sequences"][sequence.name] = _run_sequence(
                model,
                device,
                config,
                config_hash,
                sequence,
                output_root,
                args.phase,
            )
            save_json(summary_path, summary)
    except torch.cuda.OutOfMemoryError as error:
        summary["status"] = "blocked_cuda_oom"
        summary["error"] = str(error)
        save_json(summary_path, summary)
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
    summary["status"] = "complete"
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

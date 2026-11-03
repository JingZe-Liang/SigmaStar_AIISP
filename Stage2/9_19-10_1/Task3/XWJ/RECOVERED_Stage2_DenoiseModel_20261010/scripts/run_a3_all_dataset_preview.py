#!/usr/bin/env python3
"""Run A3 on every unique company-dataset stream and render ISP videos."""

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
    clip_starts,
    discover_sequences,
    estimate_exposure_ev,
    exposure_target_before_tone_map,
    infer_sequence,
    load_config,
    prediction_provenance,
    render_sequence_video,
    resolve_black_levels,
    save_json,
    sha256_file,
)
from raw_fusion.finetune.stream_io import RawStreamReader  # noqa: E402


def _select_sequences(
    sequences: list[PreviewSequence], selected: list[str]
) -> list[PreviewSequence]:
    if not selected:
        return sequences
    names = {sequence.name for sequence in sequences}
    unknown = sorted(set(selected) - names)
    if unknown:
        raise ValueError(f"unknown sequences: {unknown}")
    selected_names = set(selected)
    return [sequence for sequence in sequences if sequence.name in selected_names]


def _sequence_dir(output_dir: Path, sequence: PreviewSequence) -> Path:
    return output_dir / "sequences" / sequence.group / sequence.category / sequence.name


def _existing_runtime(sequence_dir: Path) -> dict[str, Any] | None:
    path = sequence_dir / "runtime.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a3_all_dataset_traditional_isp_preview_20261017.json",
    )
    parser.add_argument(
        "--phase", choices=("inventory", "inference", "render", "all"), default="all"
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--tile-batch", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    config_sha256 = sha256_file(args.config)
    sequences = _select_sequences(discover_sequences(config), args.sequence)
    output_dir = args.output_dir or REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = REPO_ROOT / config["checkpoint"]
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != config["checkpoint_sha256"]:
        raise ValueError("A3 checkpoint SHA-256 does not match the frozen protocol")
    temporal_frames = int(config["sampling"]["temporal_frames"])
    if temporal_frames != 8:
        raise ValueError("A3 preview protocol requires exactly eight temporal frames")
    clips_per_sequence = int(config["sampling"]["clips_per_sequence"])
    device_name = args.device or config["inference"]["preferred_device"]
    tile_batch = args.tile_batch or int(config["inference"]["tile_batch"])
    if tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    protocol = {
        "config": config,
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_sequences": [sequence.name for sequence in sequences],
        "catalog_count": len(discover_sequences(config)),
        "dataset_mutated": False,
    }
    save_json(output_dir / "protocol.json", protocol)
    save_json(
        output_dir / "inventory.json",
        {
            "count": len(sequences),
            "sequences": [sequence.to_mapping() for sequence in sequences],
        },
    )
    if args.phase == "inventory":
        print(json.dumps({"status": "inventory_complete", "count": len(sequences)}, indent=2))
        return 0

    model: torch.nn.Module | None = None
    device = torch.device(device_name)
    if args.phase in ("inference", "all"):
        if device.type != "cuda":
            raise ValueError("A3 all-dataset inference requires an explicitly selected CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model = load_checkpoint_model(checkpoint, device, ema=True)

    summary: dict[str, Any] = {
        "status": "running",
        "phase": args.phase,
        "completed": 0,
        "failed": 0,
        "sequences": {},
    }
    try:
        for sequence_index, sequence in enumerate(sequences, start=1):
            sequence_dir = _sequence_dir(output_dir, sequence)
            starts = clip_starts(sequence.frames, temporal_frames, clips_per_sequence)
            runtime = _existing_runtime(sequence_dir)
            if runtime is None:
                reader = RawStreamReader(sequence.path, sequence.width, sequence.height)
                black = resolve_black_levels(sequence, reader)
                output_target = float(
                    config["isp"]["exposure_target_after_tone_map_linear"]
                )
                exposure_target = exposure_target_before_tone_map(
                    output_target,
                    str(config["isp"]["tone_map"]),
                )
                exposure_ev, input_p995 = estimate_exposure_ev(
                    reader,
                    sequence,
                    black,
                    starts,
                    exposure_target,
                    tuple(float(value) for value in config["isp"]["exposure_ev_limits"]),
                )
                source_sha256 = sha256_file(sequence.path)
                runtime = {
                    "sequence": sequence.to_mapping(),
                    "source_sha256": source_sha256,
                    "clip_starts": list(starts),
                    "black_levels_R_Gr_Gb_B": [float(value) for value in black],
                    "black_level_source": (
                        sequence.metadata_source
                        if sequence.black_levels is not None
                        else "deterministic 16-frame project pedestal measurement"
                    ),
                    "fixed_exposure_ev": exposure_ev,
                    "input_normalized_p995_after_wb": input_p995,
                    "exposure_target_before_tone_map_linear": exposure_target,
                    "exposure_target_after_tone_map_linear": output_target,
                    "dataset_read_only": True,
                }
                save_json(sequence_dir / "runtime.json", runtime)
            else:
                black = np.asarray(runtime["black_levels_R_Gr_Gb_B"], dtype=np.float32)
                exposure_ev = float(runtime["fixed_exposure_ev"])
                source_sha256 = str(runtime["source_sha256"])
                if tuple(int(value) for value in runtime["clip_starts"]) != starts:
                    raise ValueError(f"{sequence.name}: runtime clip starts do not match config")

            provenance = prediction_provenance(
                sequence,
                source_sha256,
                checkpoint_sha256,
                config_sha256,
                starts,
                black,
                tile_batch,
            )
            if args.phase in ("inference", "all"):
                if model is None:
                    raise RuntimeError("model was not loaded")
                infer_sequence(
                    model,
                    device,
                    sequence,
                    sequence_dir,
                    starts,
                    black,
                    provenance,
                    tile_batch,
                )
            video_manifest = None
            if args.phase in ("render", "all"):
                video_manifest = render_sequence_video(
                    sequence,
                    sequence_dir,
                    starts,
                    black,
                    exposure_ev,
                    config["video"],
                    config["isp"],
                )
            summary["completed"] += 1
            summary["sequences"][sequence.name] = {
                "status": "complete",
                "index": sequence_index,
                "total": len(sequences),
                "runtime": str((sequence_dir / "runtime.json").resolve()),
                "video": video_manifest["video"] if video_manifest else None,
            }
            save_json(output_dir / "summary.json", summary)
            print(
                json.dumps(
                    {
                        "sequence": sequence.name,
                        "status": "complete",
                        "progress": f"{sequence_index}/{len(sequences)}",
                        "video": video_manifest["video"] if video_manifest else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except torch.cuda.OutOfMemoryError as error:
        summary["status"] = "blocked_cuda_oom"
        summary["failed"] += 1
        summary["error"] = str(error)
        save_json(output_dir / "summary.json", summary)
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
    summary["status"] = "complete"
    summary["summary_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()
    save_json(output_dir / "summary.json", summary)
    print(json.dumps({"status": "complete", "completed": summary["completed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

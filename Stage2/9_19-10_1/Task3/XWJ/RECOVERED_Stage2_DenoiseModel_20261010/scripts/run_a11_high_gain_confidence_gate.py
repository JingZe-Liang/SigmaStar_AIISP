#!/usr/bin/env python3
"""Run deterministic A11 gating from existing A3 high-gain predictions."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a11_confidence_gate import (  # noqa: E402
    ConfidenceGateParams,
    build_confidence_weights,
    evaluate_gate,
    fuse_branches,
    render_gate_videos,
)
from raw_fusion.finetune.all_dataset_preview import (  # noqa: E402
    PreviewSequence,
    discover_sequences,
    load_config,
    normalize_packed,
    save_json,
    sha256_file,
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


def _load_input_clip(
    sequence: PreviewSequence, source_start: int, black: np.ndarray
) -> np.ndarray:
    reader = RawStreamReader(sequence.path, sequence.width, sequence.height)
    return np.stack(
        [
            normalize_packed(
                pack_frame(reader.read_frame(source_start + offset), sequence.adapter),
                black,
            )
            for offset in range(8)
        ]
    ).astype(np.float32)


def _baseline_paths(config: dict[str, Any], sequence: PreviewSequence) -> tuple[Path, Path, Path]:
    root = REPO_ROOT / config["baseline_preview_root"]
    sequence_dir = root / "sequences" / sequence.group / sequence.category / sequence.name
    return (
        sequence_dir / "runtime.json",
        sequence_dir / "inference/prediction.npy",
        sequence_dir / "inference/manifest.json",
    )


def _load_existing_variants(
    config: dict[str, Any], sequence: PreviewSequence
) -> tuple[dict[str, np.ndarray], np.ndarray, float, int, dict[str, str]]:
    runtime_path, baseline_path, baseline_manifest_path = _baseline_paths(config, sequence)
    diagnostic_dir = REPO_ROOT / config["diagnostic_root"] / sequence.name / "inference"
    diagnostic_manifest_path = diagnostic_dir / "manifest.json"
    paths = {
        "a3_weighted": diagnostic_dir / "a3_weighted.npy",
        "a3_frame_repeated": diagnostic_dir / "a3_frame_repeated.npy",
    }
    required = (
        runtime_path,
        baseline_path,
        baseline_manifest_path,
        diagnostic_manifest_path,
        *paths.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{sequence.name}: required existing artifacts missing: {missing}")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    diagnostic_manifest = json.loads(diagnostic_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest["provenance"]["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError(f"{sequence.name}: baseline checkpoint mismatch")
    if sha256_file(baseline_path) != baseline_manifest["prediction_sha256"]:
        raise ValueError(f"{sequence.name}: A3 uniform prediction SHA-256 mismatch")
    if diagnostic_manifest["provenance"]["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError(f"{sequence.name}: diagnostic checkpoint mismatch")
    for name, path in paths.items():
        manifest_name = "weighted" if name == "a3_weighted" else "frame_repeated"
        expected_hash = diagnostic_manifest["artifacts"][manifest_name]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{sequence.name}: {name} SHA-256 mismatch")

    clip_index = int(config["sampling"]["clip_index"])
    starts = tuple(int(value) for value in runtime["clip_starts"])
    source_start = starts[clip_index]
    baseline = np.load(baseline_path, mmap_mode="r")
    offset = clip_index * 8
    uniform = np.asarray(baseline[offset : offset + 8], dtype=np.float32)
    weighted = np.asarray(np.load(paths["a3_weighted"], mmap_mode="r"), dtype=np.float32)
    repeated = np.asarray(
        np.load(paths["a3_frame_repeated"], mmap_mode="r"), dtype=np.float32
    )
    black = np.asarray(runtime["black_levels_R_Gr_Gb_B"], dtype=np.float32)
    input_clip = _load_input_clip(sequence, source_start, black)
    expected_shape = (8, 4, *sequence.packed_shape)
    if any(array.shape != expected_shape for array in (input_clip, uniform, weighted, repeated)):
        raise ValueError(f"{sequence.name}: existing array shape mismatch")
    hashes = {
        "baseline_prediction": sha256_file(baseline_path),
        "a3_weighted": sha256_file(paths["a3_weighted"]),
        "a3_frame_repeated": sha256_file(paths["a3_frame_repeated"]),
    }
    variants = {
        "input": input_clip,
        "a3_uniform": uniform,
        "a3_weighted": weighted,
        "a3_frame_repeated": repeated,
    }
    return variants, black, float(runtime["fixed_exposure_ev"]), source_start, hashes


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array.astype(np.float16))
    temporary.replace(path)


def _run_sequence(
    config: dict[str, Any],
    config_hash: str,
    sequence: PreviewSequence,
    output_root: Path,
    phase: str,
) -> dict[str, Any]:
    variants, black, exposure_ev, source_start, source_hashes = _load_existing_variants(
        config, sequence
    )
    params = ConfidenceGateParams.from_mapping(config["mask"])
    weights, masks, records = build_confidence_weights(variants["input"], params)
    candidates = {
        name: fuse_branches(
            variants["a3_weighted"], variants["a3_frame_repeated"], weight
        )
        for name, weight in weights.items()
    }
    variants["a11_gated"] = candidates["full"]
    sequence_dir = output_root / sequence.name
    inference_dir = sequence_dir / "inference"
    if phase in ("evaluate", "all"):
        for name, weight in weights.items():
            _save_array(inference_dir / f"weights_{name}.npy", weight)
            _save_array(inference_dir / f"a11_{name}.npy", candidates[name])
        np.savez_compressed(
            inference_dir / "evidence_masks.npz",
            **{name: mask.astype(np.uint8) for name, mask in masks.items()},
        )
        evaluation = evaluate_gate(variants, weights, masks, black)
        evaluation.update(
            {
                "sequence": sequence.to_mapping(),
                "source_start": source_start,
                "selected_candidate": "full",
                "input_evidence_records": records,
            }
        )
        save_json(sequence_dir / "evaluation.json", evaluation)
        save_json(
            inference_dir / "manifest.json",
            {
                "protocol": config["protocol"],
                "config_sha256": config_hash,
                "checkpoint_sha256": config["checkpoint_sha256"],
                "source_start": source_start,
                "frames": 8,
                "source_artifact_sha256": source_hashes,
                "selected_candidate": "full",
                "artifacts": {
                    "weights": {
                        name: {
                            "path": str((inference_dir / f"weights_{name}.npy").resolve()),
                            "sha256": sha256_file(inference_dir / f"weights_{name}.npy"),
                        }
                        for name in weights
                    },
                    "candidates": {
                        name: {
                            "path": str((inference_dir / f"a11_{name}.npy").resolve()),
                            "sha256": sha256_file(inference_dir / f"a11_{name}.npy"),
                        }
                        for name in candidates
                    },
                    "evidence_masks": {
                        "path": str((inference_dir / "evidence_masks.npz").resolve()),
                        "sha256": sha256_file(inference_dir / "evidence_masks.npz"),
                    },
                },
                "dataset_mutated": False,
                "training_performed": False,
                "old_teacher_used": False,
            },
        )
    evaluation_path = sequence_dir / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"{sequence.name}: evaluation is absent")
    video_manifest = None
    if phase in ("render", "all"):
        video_manifest = render_gate_videos(
            sequence_dir / "traditional_isp_video",
            sequence,
            source_start,
            black,
            exposure_ev,
            variants,
            weights["full"],
            config["isp"],
            config["video"],
        )
    return {
        "status": "complete",
        "source_start": source_start,
        "evaluation": str(evaluation_path.resolve()),
        "videos": video_manifest["videos"] if video_manifest else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a11_high_gain_confidence_gate_20261019.json",
    )
    parser.add_argument(
        "--phase", choices=("evaluate", "render", "all"), default="all"
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    config_hash = sha256_file(args.config)
    checkpoint = REPO_ROOT / config["checkpoint"]
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("A3 checkpoint SHA-256 does not match the frozen protocol")
    catalog_config = load_config(REPO_ROOT / config["catalog_config"])
    selected = args.sequence or list(config["sequences"])
    sequences = _select_sequences(discover_sequences(catalog_config), selected)
    output_root = args.output_dir or REPO_ROOT / config["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "running",
        "phase": args.phase,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "config_sha256": config_hash,
        "cpu_only_postprocess": True,
        "dataset_mutated": False,
        "training_performed": False,
        "old_teacher_used": False,
        "hard_holdout_used": False,
        "sequences": {},
    }
    save_json(output_root / "protocol.json", {"config": config, "config_sha256": config_hash})
    save_json(output_root / "summary.json", summary)
    for sequence in sequences:
        summary["sequences"][sequence.name] = _run_sequence(
            config, config_hash, sequence, output_root, args.phase
        )
        save_json(output_root / "summary.json", summary)
    summary["status"] = "complete"
    save_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebuild missing A0 motion masks from recovered manifests and source RAWs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.adapters import resolve_adapter  # noqa: E402
from raw_fusion.finetune.motion import MotionMaskParams, compute_motion_masks  # noqa: E402
from raw_fusion.finetune.stats import compute_basic_stats  # noqa: E402
from raw_fusion.finetune.stream_io import open_stream  # noqa: E402


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_path(value: str) -> Path:
    path = Path(value)
    target = path if path.is_absolute() else REPO_ROOT / path
    target = target.resolve()
    if not target.is_relative_to(REPO_ROOT):
        raise ValueError(f"motion mask target leaves repository: {target}")
    return target


def _validate_mask(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    expected_static: np.ndarray,
) -> dict[str, float | int | str]:
    mask = np.load(path, mmap_mode="r")
    expected_shape = (frames, 1, height, width)
    if mask.shape != expected_shape:
        raise ValueError(f"{path}: shape {mask.shape} != {expected_shape}")
    if mask.dtype != np.float16:
        raise ValueError(f"{path}: dtype {mask.dtype} != float16")
    minimum = float(mask.min())
    maximum = float(mask.max())
    if minimum != 0.0 or maximum != 1.0:
        raise ValueError(f"{path}: expected binary mask, range is [{minimum}, {maximum}]")
    rebuilt_static = 1.0 - np.asarray(mask[:, 0, ::2, ::2], dtype=np.float32).mean(
        axis=(1, 2), dtype=np.float64
    )
    if rebuilt_static.shape != expected_static.shape:
        raise ValueError(
            f"{path}: static series shape {rebuilt_static.shape} != {expected_static.shape}"
        )
    max_static_error = float(np.max(np.abs(rebuilt_static - expected_static)))
    if max_static_error > 1e-12:
        raise ValueError(
            f"{path}: per-frame static fractions differ from recovered manifest; "
            f"max error={max_static_error}"
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "static_fraction_mean": float(rebuilt_static.mean()),
        "max_static_fraction_error": max_static_error,
    }


def _manifest_records(manifest_dir: Path, selected: set[str]) -> list[tuple[Path, dict[str, object]]]:
    records: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(manifest_dir.glob("*.json")):
        payload = _load_json(path)
        motion = payload.get("motion")
        if not isinstance(motion, dict) or not motion.get("mask_path"):
            continue
        if selected and path.stem not in selected:
            continue
        records.append((path, payload))
    return records


def rebuild(
    *,
    supervision_dir: Path,
    report_path: Path,
    selected: set[str],
    chunk_frames: int,
    overwrite: bool,
) -> dict[str, object]:
    manifest_dir = supervision_dir / "stream_manifests"
    records = _manifest_records(manifest_dir, selected)
    if not records:
        raise ValueError("no motion-bearing manifests selected")

    report: dict[str, object] = {
        "protocol": "stage2_recovered_motion_mask_rebuild_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO_ROOT),
        "supervision_dir": str(supervision_dir.resolve()),
        "chunk_frames": chunk_frames,
        "selected": sorted(selected),
        "expected_masks": len(records),
        "status": "running",
        "masks": [],
    }
    _atomic_json(report_path, report)

    outputs: list[dict[str, object]] = []
    for index, (manifest_path, manifest) in enumerate(records, start=1):
        motion = manifest["motion"]
        if not isinstance(motion, dict):
            raise TypeError(f"{manifest_path}: motion entry is not an object")
        params_payload = motion["params"]
        if not isinstance(params_payload, dict):
            raise TypeError(f"{manifest_path}: motion params entry is not an object")
        expected_static = np.asarray(motion["static_fraction_per_frame"], dtype=np.float64)
        frames = int(manifest["frames_used"])
        height = int(manifest["height"])
        width = int(manifest["width"])
        target = _target_path(str(motion["mask_path"]))
        label = manifest_path.stem

        print(f"[rebuild-mask {index}/{len(records)}] {label}", flush=True)
        if target.exists() and not overwrite:
            validation = _validate_mask(
                target,
                frames=frames,
                height=height,
                width=width,
                expected_static=expected_static,
            )
            outputs.append({
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "mask": str(target.relative_to(REPO_ROOT)),
                "status": "existing_verified",
                **validation,
            })
            report["masks"] = outputs
            _atomic_json(report_path, report)
            continue

        adapter = resolve_adapter(
            str(manifest["group"]), str(manifest["scene"]), str(manifest["gain_label"])
        )
        source = Path(str(manifest["path"]))
        reader, _ = open_stream(source, adapter)
        if reader.frames < frames:
            raise ValueError(f"{source}: {reader.frames} frames, manifest requires {frames}")
        basic = compute_basic_stats(
            reader,
            adapter,
            chunk_frames=chunk_frames,
            max_frames=frames,
            black_sample_frames=16,
        )
        rebuilt_black = basic.black_levels
        recovered_black = manifest["black_levels"]
        if not isinstance(recovered_black, dict):
            raise TypeError(f"{manifest_path}: black_levels entry is not an object")
        black_error = max(
            abs(float(rebuilt_black[name]) - float(recovered_black[name]))
            for name in rebuilt_black
        )
        if black_error > 1e-12:
            raise ValueError(
                f"{manifest_path}: source/basic-stat mismatch; max black-level error={black_error}"
            )

        params = MotionMaskParams(
            k_mad=float(params_payload["k_mad"]),
            signal_margin_dn=float(params_payload["signal_margin_dn"]),
            min_signal_pixels=int(params_payload["min_signal_pixels"]),
        )
        result = compute_motion_masks(
            reader,
            adapter,
            basic,
            params,
            chunk_frames=chunk_frames,
        )
        max_static_error = float(
            np.max(np.abs(result.static_fraction_per_frame - expected_static))
        )
        threshold_error = abs(
            float(result.threshold_per_frame.mean())
            - float(motion["threshold_per_frame_mean_dn"])
        )
        if max_static_error > 1e-12 or threshold_error > 1e-9:
            raise ValueError(
                f"{manifest_path}: rebuilt mask statistics do not match recovered manifest; "
                f"static_error={max_static_error}, threshold_error={threshold_error}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.unlink(missing_ok=True)
        with temporary.open("wb") as handle:
            np.save(handle, result.masks)
        os.replace(temporary, target)
        validation = _validate_mask(
            target,
            frames=frames,
            height=height,
            width=width,
            expected_static=expected_static,
        )
        outputs.append({
            "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "source": str(source),
            "mask": str(target.relative_to(REPO_ROOT)),
            "status": "rebuilt_verified",
            "max_black_level_error": black_error,
            "threshold_mean_error_dn": threshold_error,
            **validation,
        })
        report["masks"] = outputs
        _atomic_json(report_path, report)
        del result, basic

    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["rebuilt"] = sum(item["status"] == "rebuilt_verified" for item in outputs)
    report["existing_verified"] = sum(
        item["status"] == "existing_verified" for item in outputs
    )
    report["total_bytes"] = sum(int(item["bytes"]) for item in outputs)
    _atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/supervision/motion_masks/rebuild_report_20261010.json",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="manifest stem to rebuild; may be repeated",
    )
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    result = rebuild(
        supervision_dir=args.supervision_dir,
        report_path=args.report,
        selected=set(args.only),
        chunk_frames=args.chunk_frames,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "rebuilt": result["rebuilt"],
                "existing_verified": result["existing_verified"],
                "total_bytes": result["total_bytes"],
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate provenance and generated files for A3 cross-sensor audit roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_array(entry: dict[str, Any]) -> None:
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    if list(array.shape) != entry["shape"]:
        raise ValueError(f"{path}: shape {array.shape} != {entry['shape']}")
    if str(array.dtype) != entry["dtype"]:
        raise ValueError(f"{path}: dtype {array.dtype} != {entry['dtype']}")
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch")


def validate_root(
    root: Path,
    *,
    require_complete: bool,
    verify_sources: bool,
) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    config_path = Path(protocol["config_path"])
    checkpoint_path = Path(protocol["checkpoint_path"])
    if _sha256(config_path) != protocol["config_sha256"]:
        raise ValueError(f"{config_path}: config SHA-256 mismatch")
    if _sha256(checkpoint_path) != protocol["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint_path}: checkpoint SHA-256 mismatch")

    checked: dict[str, Any] = {}
    for name, spec in protocol["config"]["sequences"].items():
        sequence_dir = root / "sequences" / spec["sensor"] / name
        source_manifest = _load_json(sequence_dir / "source_manifest.json")
        source_path = Path(source_manifest["source_path"])
        if source_path.stat().st_size != source_manifest["source_size_bytes"]:
            raise ValueError(f"{source_path}: source size mismatch")
        if verify_sources and _sha256(source_path) != source_manifest["source_sha256"]:
            raise ValueError(f"{source_path}: source SHA-256 mismatch")

        data_dir = "reference" if spec["reference_mode"] == "motion_aware_split_half" else "profile"
        data_report_path = sequence_dir / data_dir / "report.json"
        data_report = _load_json(data_report_path)
        if data_report["provenance"] != source_manifest:
            raise ValueError(f"{name}: data provenance does not match source manifest")
        for entry in data_report["outputs"].values():
            _check_array(entry)

        sequence_result: dict[str, Any] = {
            "source_sha256": source_manifest["source_sha256"],
            "data_report": str(data_report_path.resolve()),
            "complete": False,
        }
        inference_manifest_path = sequence_dir / "inference/manifest.json"
        metrics_path = sequence_dir / "metrics.json"
        panel_manifest_path = sequence_dir / "panels/panel_manifest.json"
        complete_files = (
            inference_manifest_path.is_file()
            and metrics_path.is_file()
            and panel_manifest_path.is_file()
        )
        if require_complete and not complete_files:
            raise FileNotFoundError(f"{name}: inference, metrics, or panels are incomplete")
        if complete_files:
            inference = _load_json(inference_manifest_path)
            provenance = inference["provenance"]
            for key, value in source_manifest.items():
                if provenance.get(key) != value:
                    raise ValueError(f"{name}: inference provenance mismatch at {key}")
            if provenance["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
                raise ValueError(f"{name}: inference checkpoint mismatch")
            for direction in inference["directions"].values():
                prediction_path = Path(direction["path"])
                if _sha256(prediction_path) != direction["sha256"]:
                    raise ValueError(f"{prediction_path}: prediction SHA-256 mismatch")
                prediction = np.load(prediction_path, mmap_mode="r")
                if list(prediction.shape) != direction["prediction_shape"]:
                    raise ValueError(f"{prediction_path}: prediction shape mismatch")
                if str(prediction.dtype) != direction["prediction_dtype"]:
                    raise ValueError(f"{prediction_path}: prediction dtype mismatch")

            metrics = _load_json(metrics_path)
            if metrics.get("legal_raw_fraction") is not None:
                fractions = [metrics["legal_raw_fraction"]]
            else:
                fractions = [
                    metrics["even_input_vs_odd_reference"]["legal_raw_fraction"],
                    metrics["odd_input_vs_even_reference"]["legal_raw_fraction"],
                ]
            if any(not 0.0 <= value <= 1.0 for value in fractions):
                raise ValueError(f"{name}: illegal legal_raw_fraction")

            panel_manifest = _load_json(panel_manifest_path)
            if not panel_manifest["panels"]:
                raise ValueError(f"{name}: panel manifest is empty")
            for panel in panel_manifest["panels"].values():
                path = Path(panel["path"])
                if _sha256(path) != panel["sha256"]:
                    raise ValueError(f"{path}: panel SHA-256 mismatch")
            sequence_result["complete"] = True
            sequence_result["metrics"] = str(metrics_path.resolve())
            sequence_result["panels"] = len(panel_manifest["panels"])
        checked[name] = sequence_result

    return {
        "root": str(root.resolve()),
        "protocol_sha256": protocol["config_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "require_complete": require_complete,
        "verify_sources": verify_sources,
        "sequences": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "roots": [
            validate_root(
                root,
                require_complete=args.require_complete,
                verify_sources=args.verify_sources,
            )
            for root in args.root
        ]
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the A8 training-source eligibility gate."""

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from raw_fusion.finetune import a8_data


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_payload(
    path: Path,
    *,
    approved: bool,
    semantics: str,
    evidence: tuple[Path, str] | None = None,
) -> dict:
    payload = {
        "name": path.stem,
        "sensor": "sensor_a",
        "path": str(path),
        "width": 4,
        "height": 2,
        "expected_frames": 3,
        "bit_shift": 0,
        "cfa": "RGGB",
        "target_semantics": semantics,
        "approved_for_backprop": approved,
        "reason": "test source",
    }
    if evidence is not None:
        payload["supervision_manifest"] = str(evidence[0])
        payload["supervision_manifest_sha256"] = evidence[1]
    return payload


def _validation(tmp_path: Path, source: Path) -> Path:
    audit_root = tmp_path / "audit"
    sequence_dir = audit_root / "sequences/sensor_a/frozen"
    manifest = sequence_dir / "source_manifest.json"
    source_provenance = {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "source_size_bytes": source.stat().st_size,
    }
    _write_json(
        manifest,
        source_provenance,
    )
    report = sequence_dir / "reference/report.json"
    _write_json(report, {"provenance": source_provenance})
    validation = tmp_path / "validation.json"
    _write_json(
        validation,
        {
            "roots": [
                {
                    "root": str(audit_root),
                    "require_complete": True,
                    "verify_sources": True,
                    "sequences": {
                        "frozen": {
                            "source_sha256": source_provenance["source_sha256"],
                            "data_report": str(report),
                            "complete": True,
                        }
                    },
                }
            ]
        },
    )
    return validation


def _set_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: Path,
    validation: Path,
) -> None:
    monkeypatch.setattr(a8_data, "_REPO_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        a8_data,
        "_AUTHORITATIVE_DATASET_ROOT",
        dataset.resolve(),
    )
    monkeypatch.setattr(
        a8_data,
        "_AUTHORITATIVE_VALIDATION_PATH",
        validation.resolve(),
    )
    monkeypatch.setattr(
        a8_data,
        "_AUTHORITATIVE_VALIDATION_SHA256",
        _sha256(validation),
    )
    source_manifest = json.loads(
        (
            tmp_path
            / "audit/sequences/sensor_a/frozen/source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        a8_data,
        "_AUTHORITATIVE_FROZEN_SOURCES",
        {
            "frozen": (
                source_manifest["source_path"],
                source_manifest["source_size_bytes"],
                source_manifest["source_sha256"],
            )
        },
    )
    monkeypatch.setattr(a8_data, "_AUTHORITATIVE_REQUIRED_SENSORS", ("sensor_a",))
    monkeypatch.setattr(
        a8_data,
        "_AUTHORITATIVE_CANDIDATE_SENSORS",
        {str(path.resolve()): "sensor_a" for path in dataset.glob("*.raw")},
    )
    monkeypatch.setattr(a8_data, "_AUTHORITATIVE_APPROVED_TRAINING_SOURCES", {})


def _evidence(
    tmp_path: Path,
    source: Path,
    semantics: str,
    *,
    input_ranges: list[list[int]] | None = None,
    target_ranges: list[list[int]] | None = None,
) -> tuple[Path, str]:
    target = tmp_path / "evidence/target.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    target_array = np.full((4, 2, 2), 0.25, dtype=np.float32)
    np.save(target, target_array)
    label_origin = (
        "direct_clean_capture"
        if semantics == "clean_reference"
        else "disjoint_same_sensor_frames"
    )
    payload = {
        "schema_version": "a8-supervision-v1",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "target_semantics": semantics,
        "independent_target": True,
        "label_origin": label_origin,
        "old_teacher_used": False,
        "temporal_average_used_as_dynamic_truth": False,
        "target_artifacts": [
            {
                "path": str(target),
                "sha256": _sha256(target),
                "role": "clean_target",
                "format": "npy",
                "dtype": "float32",
                "shape": list(target_array.shape),
            }
        ],
    }
    if input_ranges is not None:
        payload["input_frame_ranges"] = input_ranges
    if target_ranges is not None:
        payload["target_frame_ranges"] = target_ranges
    manifest = tmp_path / "evidence/supervision.json"
    _write_json(manifest, payload)
    return manifest, _sha256(manifest)


def _approve_source(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    semantics: str,
    evidence: tuple[Path, str],
) -> None:
    monkeypatch.setattr(
        a8_data,
        "_AUTHORITATIVE_APPROVED_TRAINING_SOURCES",
        {
            str(source.resolve()): (
                "sensor_a",
                semantics,
                str(evidence[0].resolve()),
                evidence[1],
            )
        },
    )


def _config(
    path: Path,
    dataset: Path,
    validation: Path,
    sources: list[dict],
) -> Path:
    _write_json(
        path,
        {
            "dataset_root": str(dataset),
            "frozen_validation": str(validation),
            "required_sensors": ["sensor_a"],
            "candidate_sources": sources,
        },
    )
    return path


def test_audit_blocks_frozen_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [_source_payload(frozen, approved=False, semantics="frozen")],
    )

    result = a8_data.audit_training_sources(config)

    assert result["status"] == "blocked_no_compliant_training_source"
    assert result["sources"][0]["status"] == "blocked_frozen_audit_source"


def test_audit_allows_independent_approved_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    trainable = dataset / "trainable.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    trainable.write_bytes(b"\x01" * (4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    evidence = _evidence(
        tmp_path,
        trainable,
        "independent_noise2noise",
        input_ranges=[[0, 1]],
        target_ranges=[[1, 3]],
    )
    _approve_source(monkeypatch, trainable, "independent_noise2noise", evidence)
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [
            _source_payload(
                trainable,
                approved=True,
                semantics="independent_noise2noise",
                evidence=evidence,
            )
        ],
    )

    result = a8_data.audit_training_sources(config)

    assert result["status"] == "ready"
    assert result["sources"][0]["status"] == "trainable"


def test_audit_rejects_frame_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    candidate = dataset / "candidate.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    candidate.write_bytes(bytes(4 * 2 * 2 * 2))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [_source_payload(candidate, approved=False, semantics="dynamic")],
    )

    with pytest.raises(ValueError, match="expected 3 frames, found 2"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_non_boolean_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    candidate = dataset / "candidate.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    candidate.write_bytes(bytes(4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    source = _source_payload(
        candidate,
        approved=False,
        semantics="dynamic_no_clean_target",
    )
    source["approved_for_backprop"] = "false"
    config = _config(tmp_path / "config.json", dataset, validation, [source])

    with pytest.raises(TypeError, match="must be a JSON boolean"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_duplicate_candidate_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    candidate = dataset / "candidate.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    candidate.write_bytes(bytes(4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    first = _source_payload(
        candidate,
        approved=False,
        semantics="dynamic_no_clean_target",
    )
    second = {**first, "name": "duplicate"}
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [first, second],
    )

    with pytest.raises(ValueError, match="duplicate candidate source path"):
        a8_data.audit_training_sources(config)


def test_audit_blocks_hardlink_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    alias = dataset / "alias.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    os.link(frozen, alias)
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [_source_payload(alias, approved=False, semantics="alias")],
    )

    result = a8_data.audit_training_sources(config)

    assert result["sources"][0]["status"] == "blocked_frozen_audit_source"
    assert result["sources"][0]["frozen_inode_alias"] is True


def test_audit_blocks_frozen_content_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    copied = dataset / "copied.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    shutil.copyfile(frozen, copied)
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    source = _source_payload(
        copied,
        approved=True,
        semantics="clean_reference",
        evidence=(tmp_path / "unused.json", "0" * 64),
    )
    config = _config(tmp_path / "config.json", dataset, validation, [source])

    result = a8_data.audit_training_sources(config)

    assert result["sources"][0]["status"] == "blocked_frozen_content_alias"


def test_audit_rejects_tampered_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    validation.write_text(validation.read_text() + " ", encoding="utf-8")
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [_source_payload(frozen, approved=False, semantics="frozen")],
    )

    with pytest.raises(ValueError, match="validation SHA-256 mismatch"):
        a8_data.audit_training_sources(config)


def test_source_spec_rejects_non_integer_numeric_fields(tmp_path: Path) -> None:
    source = _source_payload(
        tmp_path / "candidate.raw",
        approved=False,
        semantics="dynamic",
    )
    source["width"] = 4.0

    with pytest.raises(TypeError, match="width must be a JSON integer"):
        a8_data.A8SourceSpec.from_mapping(source)


def test_audit_rejects_overlapping_supervision_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    candidate = dataset / "candidate.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    candidate.write_bytes(b"\x01" * (4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    evidence = _evidence(
        tmp_path,
        candidate,
        "static_disjoint_reference",
        input_ranges=[[0, 2]],
        target_ranges=[[1, 3]],
    )
    _approve_source(monkeypatch, candidate, "static_disjoint_reference", evidence)
    source = _source_payload(
        candidate,
        approved=True,
        semantics="static_disjoint_reference",
        evidence=evidence,
    )
    config = _config(tmp_path / "config.json", dataset, validation, [source])

    with pytest.raises(ValueError, match="frame pools overlap"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_non_authoritative_dataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    other = tmp_path / "other"
    frozen = dataset / "frozen.raw"
    dataset.mkdir()
    other.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    config = _config(
        tmp_path / "config.json",
        other,
        validation,
        [_source_payload(frozen, approved=False, semantics="frozen")],
    )

    with pytest.raises(ValueError, match="dataset_root must be"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_replaced_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    replacement = dataset / "replacement.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    replacement.write_bytes(b"\x01" * (4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    provenance = {
        "source_path": str(replacement),
        "source_sha256": _sha256(replacement),
        "source_size_bytes": replacement.stat().st_size,
    }
    sequence_dir = tmp_path / "audit/sequences/sensor_a/frozen"
    _write_json(sequence_dir / "source_manifest.json", provenance)
    _write_json(sequence_dir / "reference/report.json", {"provenance": provenance})
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [_source_payload(frozen, approved=False, semantics="frozen")],
    )

    with pytest.raises(ValueError, match="source path differs from frozen policy"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_invalid_target_artifact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    candidate = dataset / "candidate.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    candidate.write_bytes(b"\x01" * (4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    manifest = tmp_path / "evidence/supervision.json"
    _write_json(
        manifest,
        {
            "schema_version": "a8-supervision-v1",
            "source_path": str(candidate),
            "source_sha256": _sha256(candidate),
            "target_semantics": "clean_reference",
            "independent_target": True,
            "label_origin": "direct_clean_capture",
            "old_teacher_used": False,
            "temporal_average_used_as_dynamic_truth": False,
            "target_artifacts": [
                {
                    "path": str(candidate),
                    "sha256": _sha256(candidate),
                    "role": "clean_target",
                    "format": "npy",
                    "dtype": "float32",
                    "shape": [4, 2, 3],
                }
            ],
        },
    )
    evidence = (manifest, _sha256(manifest))
    _approve_source(monkeypatch, candidate, "clean_reference", evidence)
    source = _source_payload(
        candidate,
        approved=True,
        semantics="clean_reference",
        evidence=evidence,
    )
    config = _config(tmp_path / "config.json", dataset, validation, [source])

    with pytest.raises(ValueError, match="must be NPY"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_sensor_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    candidate = dataset / "candidate.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    candidate.write_bytes(b"\x01" * (4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    source = _source_payload(candidate, approved=False, semantics="dynamic")
    source["sensor"] = "sensor_b"
    config = _config(tmp_path / "config.json", dataset, validation, [source])

    with pytest.raises(ValueError, match="sensor differs from candidate policy"):
        a8_data.audit_training_sources(config)


def test_audit_rejects_required_sensor_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    frozen = dataset / "frozen.raw"
    dataset.mkdir()
    frozen.write_bytes(bytes(4 * 2 * 2 * 3))
    validation = _validation(tmp_path, frozen)
    _set_policy(monkeypatch, tmp_path, dataset, validation)
    config = _config(
        tmp_path / "config.json",
        dataset,
        validation,
        [_source_payload(frozen, approved=False, semantics="frozen")],
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["required_sensors"] = ["sensor_b"]
    _write_json(config, payload)

    with pytest.raises(ValueError, match="required_sensors must be"):
        a8_data.audit_training_sources(config)

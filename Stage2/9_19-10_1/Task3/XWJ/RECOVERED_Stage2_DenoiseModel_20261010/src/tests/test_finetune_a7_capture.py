import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from raw_fusion.finetune import a7_capture
from raw_fusion.finetune.a7_capture import (
    CAPTURE_SCHEMA_VERSION,
    CaptureClip,
    CaptureManifest,
    CaptureValidationThresholds,
    REQUIRED_MOTION_TYPES,
    analyze_capture_clip,
    load_capture_manifest,
    validate_capture_inventory,
    validate_capture_dataset,
)


def _write_motion_raw(path: Path, frames: int, low_aligned: bool = False, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    output = np.empty((frames, 32, 48), dtype="<u2")
    for index in range(frames):
        frame = np.full((32, 48), 520.0, dtype=np.float32)
        left = 2 + index % 20
        frame[8:24, left : left + 12] = 1100.0
        frame += rng.normal(0.0, 2.0, frame.shape)
        dn = np.clip(np.rint(frame), 0, 4095).astype(np.uint16)
        output[index] = dn if low_aligned else dn << 4
    output.tofile(path)


def _manifest(clips: tuple[CaptureClip, ...]) -> CaptureManifest:
    return CaptureManifest(
        sensor="MIS20S1",
        width=48,
        height=32,
        dtype="uint16_le",
        bit_alignment="high12",
        cfa="RGGB",
        black_level_dn=256.0,
        white_level_dn=4095,
        clips=clips,
    )


def _clip(path: str = "clip.raw") -> CaptureClip:
    return CaptureClip(
        clip_id="train_s01_lateral_walk",
        scene_id="s01",
        split="train",
        motion_type="lateral_walk",
        path=path,
        fps=12.5,
        exposure_us=4000,
        sensor_gain=1024,
        isp_gain=1024,
        wb_r=2048,
        wb_g=1024,
        wb_b=2048,
        nr_disabled=True,
        auto_exposure=False,
        auto_gain=False,
        auto_white_balance=False,
        tripod_fixed=True,
    )


def _thresholds() -> CaptureValidationThresholds:
    return CaptureValidationThresholds(
        min_frames_per_clip=12,
        min_train_scenes=1,
        min_validation_scenes=0,
        max_target_noise_dn=8.0,
        max_background_cv=0.03,
        max_background_deviation=0.08,
        min_active_pair_fraction=0.1,
        min_motion_fraction=0.005,
        sample_pairs=8,
        noise_block_size=4,
        motion_block_size=2,
    )


def test_analyze_clip_accepts_clean_high12_motion(tmp_path: Path) -> None:
    raw = tmp_path / "clip.raw"
    _write_motion_raw(raw, frames=16)
    result = analyze_capture_clip(tmp_path, _manifest((_clip(),)), _clip(), _thresholds())
    assert result["passed"]
    assert result["metrics"]["high12_alignment_fraction"] == 1.0
    assert result["metrics"]["target_noise_dn"] < 8.0
    assert result["metrics"]["mean_motion_fraction"] > 0.005


def test_analyze_clip_rejects_low_alignment(tmp_path: Path) -> None:
    raw = tmp_path / "clip.raw"
    _write_motion_raw(raw, frames=16, low_aligned=True)
    result = analyze_capture_clip(tmp_path, _manifest((_clip(),)), _clip(), _thresholds())
    assert not result["passed"]
    assert "high12_alignment" in result["failures"]


def test_inventory_rejects_split_leakage_and_missing_motion_types() -> None:
    first = _clip()
    second = replace(
        first,
        clip_id="validation_s01_depth_walk",
        split="validation",
        motion_type="depth_walk",
        path="validation.raw",
    )
    result = validate_capture_inventory(_manifest((first, second)), _thresholds())
    assert not result["passed"]
    assert any("scene leakage" in failure for failure in result["failures"])
    assert any("missing motion types" in failure for failure in result["failures"])


def test_inventory_rejects_mixed_gain_and_fps() -> None:
    clips = tuple(
        replace(
            _clip(),
            clip_id=f"s01_{motion_type}",
            motion_type=motion_type,
            path=f"{motion_type}.raw",
            sensor_gain=2048 if index == 0 else 1024,
            fps=25.0 if index == 1 else 12.5,
        )
        for index, motion_type in enumerate(sorted(REQUIRED_MOTION_TYPES))
    )
    result = validate_capture_inventory(_manifest(clips), _thresholds())
    assert not result["passed"]
    assert any("one low-gain setting" in failure for failure in result["failures"])
    assert any("one FPS" in failure for failure in result["failures"])


def test_load_manifest_requires_raw_capture_controls(tmp_path: Path) -> None:
    clips = []
    for motion_type in sorted(REQUIRED_MOTION_TYPES):
        clips.append(
            {
                "clip_id": f"s01_{motion_type}",
                "scene_id": "s01",
                "split": "train",
                "motion_type": motion_type,
                "path": f"{motion_type}.raw",
                "fps": 12.5,
                "exposure_us": 4000,
                "sensor_gain": 1024,
                "isp_gain": 1024,
                "wb_r": 2048,
                "wb_g": 1024,
                "wb_b": 2048,
                "nr_disabled": "yes",
                "auto_exposure": False,
                "auto_gain": False,
                "auto_white_balance": False,
                "tripod_fixed": True,
            }
        )
    payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "sensor": "MIS20S1",
        "width": 1920,
        "height": 1080,
        "dtype": "uint16_le",
        "bit_alignment": "high12",
        "cfa": "RGGB",
        "black_level_dn": 256.0,
        "white_level_dn": 4095,
        "clips": clips,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_capture_manifest(path)
    except ValueError as error:
        assert "nr_disabled must be a boolean" in str(error)
    else:
        raise AssertionError("expected strict boolean validation")


def _complete_capture(tmp_path: Path, duplicate_kind: str | None = None) -> Path:
    clips: list[CaptureClip] = []
    first = tmp_path / "clip_0.raw"
    for scene in range(6):
        for motion_type in sorted(REQUIRED_MOTION_TYPES):
            index = len(clips)
            path = tmp_path / f"clip_{index}.raw"
            relative = path.name
            if index == 0 or duplicate_kind is None:
                _write_motion_raw(path, frames=16, seed=index)
            elif duplicate_kind == "alias":
                relative = f"alias_{index}/../{first.name}"
            elif duplicate_kind == "symlink":
                path.symlink_to(first.name)
            elif duplicate_kind == "hardlink":
                path.hardlink_to(first)
            elif duplicate_kind == "copy":
                path.write_bytes(first.read_bytes())
            else:
                raise ValueError(f"unknown duplicate fixture {duplicate_kind}")
            clips.append(replace(
                _clip(relative), clip_id=f"clip_{index}", scene_id=f"s{scene}",
                split="train" if scene < 4 else "validation", motion_type=motion_type,
            ))
    payload = asdict(_manifest(tuple(clips))) | {"schema_version": CAPTURE_SCHEMA_VERSION}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("duplicate_kind", ["alias", "symlink", "hardlink", "copy"])
def test_dataset_rejects_raw_reuse_across_declared_scenes(tmp_path: Path, duplicate_kind: str) -> None:
    manifest_path = _complete_capture(tmp_path, duplicate_kind)
    limits = replace(_thresholds(), min_train_scenes=4, min_validation_scenes=2)
    report = validate_capture_dataset(manifest_path, tmp_path, limits)
    assert len(report["inventory"]["train_scenes"]) == 4
    assert len(report["inventory"]["validation_scenes"]) == 2
    assert all(clip["passed"] for clip in report["clips"])
    assert not report["inventory"]["passed"]
    assert not report["overall_pass"]
    assert not report["training_ready"]
    assert report["admission_status"] == "automatic_checks_failed"
    assert any(group["kind"] == "identical_content" for group in report["inventory"]["duplicate_files"])
    assert len({record["sha256"] for record in report["source_files"]}) == 1


def test_independent_files_pass_automatic_checks_but_require_manual_review(tmp_path: Path) -> None:
    manifest_path = _complete_capture(tmp_path)
    limits = replace(_thresholds(), min_train_scenes=4, min_validation_scenes=2)
    report = validate_capture_dataset(manifest_path, tmp_path, limits)
    assert report["overall_pass"]
    assert report["inventory"]["duplicate_files"] == []
    assert len({record["sha256"] for record in report["source_files"]}) == 30
    assert not report["training_ready"]
    assert report["admission_status"] == "manual_review_required"
    assert len(report["manual_review_required"]) == 4


@pytest.mark.parametrize("field", ["fps", "black_level_dn"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_manifest_rejects_nonfinite_numbers(tmp_path: Path, field: str, value: float) -> None:
    payload = asdict(_manifest((_clip(),))) | {"schema_version": CAPTURE_SCHEMA_VERSION}
    if field == "fps":
        payload["clips"][0][field] = value
    else:
        payload[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{field} must be a finite positive number"):
        load_capture_manifest(path)


@pytest.mark.parametrize("black", [4095, 5000])
def test_manifest_rejects_black_level_at_or_above_white(tmp_path: Path, black: int) -> None:
    payload = asdict(_manifest((_clip(),))) | {
        "schema_version": CAPTURE_SCHEMA_VERSION, "black_level_dn": black,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="black_level_dn must be below white_level_dn"):
        load_capture_manifest(path)


def test_clip_rejects_absolute_path_and_symlink_escape(tmp_path: Path) -> None:
    raw = tmp_path / "clip.raw"
    _write_motion_raw(raw, frames=16)
    with pytest.raises(ValueError, match="RAW path must be relative"):
        analyze_capture_clip(tmp_path, _manifest((_clip(),)), _clip(str(raw)), _thresholds())
    contained = tmp_path / "contained"
    contained.mkdir()
    (contained / "clip.raw").symlink_to(raw)
    with pytest.raises(ValueError, match="path escapes dataset root"):
        analyze_capture_clip(contained, _manifest((_clip(),)), _clip(), _thresholds())


@pytest.mark.parametrize("frames", [0, 1])
def test_clip_rejects_missing_frame_pair_clearly(tmp_path: Path, frames: int) -> None:
    _write_motion_raw(tmp_path / "clip.raw", frames=frames)
    with pytest.raises(ValueError, match="at least two complete RAW frames"):
        analyze_capture_clip(tmp_path, _manifest((_clip(),)), _clip(), _thresholds())


@pytest.mark.parametrize("stage", ["parsing", "quality checks"])
def test_dataset_rejects_manifest_changes_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    manifest_path = _complete_capture(tmp_path)
    if stage == "parsing":
        original_load = a7_capture.load_capture_manifest

        def changing_load(path: Path) -> CaptureManifest:
            result = original_load(path)
            payload = json.loads(path.read_text())
            payload["clips"][0]["fps"] = 25.0
            path.write_text(json.dumps(payload))
            return result

        monkeypatch.setattr(a7_capture, "load_capture_manifest", changing_load)
    else:
        original_analyze = a7_capture.analyze_capture_clip

        def changing_analyze(
            root: Path, manifest: CaptureManifest, clip: CaptureClip, limits: CaptureValidationThresholds,
        ) -> dict:
            result = original_analyze(root, manifest, clip, limits)
            payload = json.loads(manifest_path.read_text())
            payload["clips"][0]["fps"] = 25.0
            manifest_path.write_text(json.dumps(payload))
            return result

        monkeypatch.setattr(a7_capture, "analyze_capture_clip", changing_analyze)
    with pytest.raises(ValueError, match=f"capture manifest changed during {stage}"):
        validate_capture_dataset(manifest_path, tmp_path, _thresholds())


def test_dataset_rejects_retargeted_symlink_after_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _complete_capture(tmp_path)
    link = tmp_path / "clip_0.raw"
    link.rename(tmp_path / "original_0.raw")
    link.symlink_to("original_0.raw")
    _write_motion_raw(tmp_path / "replacement.raw", frames=16, seed=100)
    original_analyze = a7_capture.analyze_capture_clip

    def changing_analyze(
        root: Path, manifest: CaptureManifest, clip: CaptureClip, limits: CaptureValidationThresholds,
    ) -> dict:
        result = original_analyze(root, manifest, clip, limits)
        if clip.clip_id == "clip_0":
            link.rename(tmp_path / "saved_original_link.raw")
            link.symlink_to("replacement.raw")
        else:
            pass
        return result

    monkeypatch.setattr(a7_capture, "analyze_capture_clip", changing_analyze)
    with pytest.raises(ValueError, match="RAW path target changed during validation"):
        validate_capture_dataset(manifest_path, tmp_path, _thresholds())

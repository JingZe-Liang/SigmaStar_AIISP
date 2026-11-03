import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from raw_fusion.finetune.a7_capture import (
    CAPTURE_SCHEMA_VERSION,
    CaptureClip,
    CaptureManifest,
    CaptureValidationThresholds,
    REQUIRED_MOTION_TYPES,
    analyze_capture_clip,
    load_capture_manifest,
    validate_capture_inventory,
)


def _write_motion_raw(path: Path, frames: int, low_aligned: bool = False) -> None:
    rng = np.random.default_rng(7)
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

"""Tests for exact constructed targets and region-specific restoration errors."""

import cv2
import json
import numpy as np
from pathlib import Path
import pytest

from raw_fusion.finetune.a12_dynamic_probe import (
    CHANNELS,
    add_measured_noise,
    make_motion_probe,
    measure_regions,
    preprocess_model_input,
    rescore_cached_probes,
    sha256_file,
)


def _background(size: int = 16) -> np.ndarray:
    y, x = np.indices((size, size))
    return np.stack([300.0 + channel * 100.0 + y * 3.0 + x for channel in range(4)]).astype(np.float32)


def _noise_model() -> dict:
    return {
        "gain_factors": {"100": 1.0},
        "empirical_shot_coefficients": {channel: {"100": 0.5} for channel in CHANNELS},
        "read_variance_dn2": {channel: {"100": 4.0} for channel in CHANNELS},
        "channel_correlation": {"aggregate": np.eye(4).tolist()},
    }


def test_integer_translation_keeps_channels_and_excludes_new_border() -> None:
    background = _background()
    valid = np.ones((16, 16), dtype=bool)
    valid[8, 8] = False
    probe = make_motion_probe(background, valid, mode="global_translation", frames=3, velocity_yx=(1, -2))
    np.testing.assert_array_equal(probe.target_dn[2, :, 2:, :-4], background[:, :-2, 4:])
    assert not probe.valid[2, :2].any()
    assert not probe.valid[2, :, -4:].any()
    assert not probe.valid[2, 10, 4]
    assert not probe.static.any()
    np.testing.assert_array_equal(probe.motion, probe.valid)


def test_moving_foreground_has_exact_targets_and_vacated_background_motion_mask() -> None:
    background = _background()
    foreground = np.full((4, 4, 4), 900.0, dtype=np.float32)
    valid = np.ones((16, 16), dtype=bool)
    fg_valid = np.ones((4, 4), dtype=bool)
    fg_valid[0, 0] = False
    probe = make_motion_probe(background, valid, mode="moving_patch", frames=3, velocity_yx=(0, 2),
                              foreground=foreground, foreground_valid=fg_valid, origin_yx=(4, 2))
    np.testing.assert_array_equal(probe.target_dn[0, :, 4:8, 2:6], foreground)
    np.testing.assert_array_equal(probe.target_dn[2, :, 4:8, 6:10], foreground)
    np.testing.assert_array_equal(probe.target_dn[2, :, 4:8, 2:6], background[:, 4:8, 2:6])
    assert probe.motion[2, 5, 3]
    assert probe.motion[0, 5, 8]
    assert not probe.valid[0, 4, 2]
    assert not np.any(probe.static & probe.motion)
    np.testing.assert_array_equal(probe.static | probe.motion, probe.valid)


def test_noise_is_seed_reproducible_independent_across_frames_and_not_constant_fpn() -> None:
    target = np.repeat(_background()[None], 8, axis=0)
    fpn = np.full_like(target[0], 3.0)
    black = np.full(4, 256.0, dtype=np.float32)
    first = add_measured_noise(target, _noise_model(), "100", black, fpn, 9)
    second = add_measured_noise(target, _noise_model(), "100", black, fpn, 9)
    different = add_measured_noise(target, _noise_model(), "100", black, fpn, 10)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert not np.array_equal(first[0], first[1])
    assert first.dtype == np.uint16


def test_exact_target_has_zero_error_and_empty_region_is_explicit() -> None:
    probe = make_motion_probe(_background(), np.ones((16, 16), dtype=bool), mode="static")
    metrics = measure_regions(probe.target_dn, probe, black_dn=np.zeros(4))
    assert metrics["motion_swept"] is None
    for region in ("all", "static"):
        assert metrics[region]["rmse_dn"] == 0.0
        assert metrics[region]["highpass_rmse_dn"] == 0.0
        assert metrics[region]["gradient_rmse_dn"] == 0.0
        assert metrics[region]["temporal_difference_error_rmse_dn"] == 0.0


def test_invalid_boundary_cannot_pollute_highpass_metric() -> None:
    valid = np.ones((16, 16), dtype=bool)
    valid[:, 0] = False
    probe = make_motion_probe(_background(), valid, mode="static")
    prediction = probe.target_dn.copy()
    prediction[..., 0] = 4000.0
    metrics = measure_regions(prediction, probe, black_dn=np.zeros(4))["all"]
    assert metrics["rmse_dn"] == 0.0
    assert metrics["highpass_rmse_dn"] == 0.0


def test_temporal_average_erases_motion_despite_lower_frame_variation() -> None:
    background = np.full((4, 32, 32), 400.0, dtype=np.float32)
    foreground = np.full((4, 8, 8), 800.0, dtype=np.float32)
    probe = make_motion_probe(background, np.ones((32, 32), dtype=bool), mode="moving_patch", frames=8,
                              velocity_yx=(0, 2), foreground=foreground,
                              foreground_valid=np.ones((8, 8), dtype=bool), origin_yx=(8, 4))
    average = np.broadcast_to(probe.target_dn.mean(axis=0, keepdims=True), probe.target_dn.shape)
    assert np.mean(np.abs(np.diff(average, axis=0))) == 0.0
    metrics = measure_regions(average, probe, black_dn=np.zeros(4))
    assert metrics["motion_swept"]["temporal_difference_error_rmse_dn"] > 0.0
    assert metrics["motion_swept"]["highpass_rmse_dn"] > 0.0
    assert metrics["static"]["rmse_dn"] == 0.0


def test_blur_reduces_flat_noise_but_loses_structured_detail() -> None:
    background = np.full((4, 32, 32), 400.0, dtype=np.float32)
    background[:, :, 16:] += (np.arange(16) % 2)[None, None] * 100.0
    probe = make_motion_probe(background, np.ones((32, 32), dtype=bool), mode="static")
    rng = np.random.default_rng(99)
    noisy = probe.target_dn + rng.normal(0.0, 3.0, probe.target_dn.shape)
    blurred = np.stack([np.stack([cv2.blur(channel, (3, 3)) for channel in frame]) for frame in noisy])
    flat_mask = np.zeros((8, 32, 32), dtype=bool)
    flat_mask[:, :, :12] = True
    detail_mask = np.zeros_like(flat_mask)
    detail_mask[:, :, 20:] = True
    flat_probe = type(probe)(probe.target_dn, flat_mask, np.zeros_like(flat_mask), flat_mask)
    detail_probe = type(probe)(probe.target_dn, detail_mask, np.zeros_like(detail_mask), detail_mask)
    assert measure_regions(blurred, flat_probe, black_dn=np.zeros(4))["all"]["rmse_dn"] < measure_regions(noisy, flat_probe, black_dn=np.zeros(4))["all"]["rmse_dn"]
    assert measure_regions(blurred, detail_probe, black_dn=np.zeros(4))["all"]["highpass_rmse_dn"] > measure_regions(noisy, detail_probe, black_dn=np.zeros(4))["all"]["highpass_rmse_dn"]


@pytest.mark.parametrize("mode", ["invalid", "moving_patch"])
def test_invalid_motion_configuration_fails_explicitly(mode: str) -> None:
    with pytest.raises(ValueError):
        make_motion_probe(_background(), np.ones((16, 16), dtype=bool), mode=mode)


def test_storage_zero_and_normalized_black_endpoints_are_distinct() -> None:
    black = np.asarray([250.0, 252.0, 255.0, 258.0], dtype=np.float32)
    probe = make_motion_probe(np.full((4, 4, 4), 300.0, dtype=np.float32), np.ones((4, 4), dtype=bool), mode="static", frames=2)
    raw = probe.target_dn.copy()
    raw[:, :, 0, 0] = 0.0
    raw[:, :, 0, 1] = black
    raw[:, :, 0, 2] = black - 1.0
    normalized, model_input = preprocess_model_input(raw, black)
    assert np.all(normalized[:, :, 0, :3] == 0.0)
    np.testing.assert_array_equal(model_input[:, :, 0, 0], np.broadcast_to(black, (2, 4)))
    raw_metrics = measure_regions(raw, probe, black_dn=black)["all"]
    model_metrics = measure_regions(model_input, probe, black_dn=black)["all"]
    assert raw_metrics["storage_code_endpoints"]["zero_fraction"] == pytest.approx(1 / 16)
    assert raw_metrics["normalized_domain_limits"]["black_endpoint_fraction"] == pytest.approx(1 / 16)
    assert raw_metrics["normalized_domain_limits"]["below_black_fraction"] == pytest.approx(2 / 16)
    assert model_metrics["storage_code_endpoints"]["zero_fraction"] == 0.0
    assert model_metrics["normalized_domain_limits"]["black_endpoint_fraction"] == pytest.approx(3 / 16)
    assert model_metrics["normalized_domain_limits"]["below_black_fraction"] == 0.0
    assert model_metrics["rmse_dn"] < raw_metrics["rmse_dn"]


def _cached_case(tmp_path: Path) -> tuple[Path, Path]:
    prepared = tmp_path / "prepared"
    source = tmp_path / "baseline"
    prepared.mkdir()
    source.mkdir()
    probe = make_motion_probe(_background(), np.ones((16, 16), dtype=bool), mode="static")
    case_path = prepared / "case.npz"
    np.savez(case_path, noisy_dn=probe.target_dn.astype(np.uint16), target_dn=probe.target_dn,
             valid=probe.valid, motion=probe.motion, static=probe.static, black_dn=np.full(4, 256.0, dtype=np.float32))
    prediction_path = source / "prediction.npy"
    np.save(prediction_path, probe.target_dn)
    manifest = {
        "config": {"protocol": "test", "scope": "test fixture"},
        "cases": [{"name": "case", "mode": "static", "gain": "100", "file": case_path.name, "sha256": sha256_file(case_path)}],
    }
    manifest_path = prepared / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evaluation = {
        "prepared_manifest_sha256": sha256_file(manifest_path), "weights": "ema",
        "checkpoint": "unused_for_cpu_rescore.pth", "checkpoint_sha256": "historical_hash",
        "cases": [{"name": "case", "a3_ema": {}, "prediction": {"file": prediction_path.name, "sha256": sha256_file(prediction_path)}}],
    }
    (source / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    return prepared, source


def test_rescore_uses_cache_without_mutating_originals_or_loading_checkpoint(tmp_path: Path) -> None:
    prepared, source = _cached_case(tmp_path)
    source_hashes = {path.name: sha256_file(path) for path in source.iterdir()}
    output = tmp_path / "rescored"
    result = rescore_cached_probes(prepared, source, output)
    assert result["cases"][0]["a3_ema"]["all"]["rmse_dn"] == 0.0
    assert "raw_input" in result["cases"][0] and "model_input" in result["cases"][0]
    assert result["metric_protocol"]["version"] == "a12_region_errors_v2"
    assert source_hashes == {path.name: sha256_file(path) for path in source.iterdir()}
    with pytest.raises(FileExistsError):
        rescore_cached_probes(prepared, source, output)


def test_rescore_rejects_changed_cached_prediction(tmp_path: Path) -> None:
    prepared, source = _cached_case(tmp_path)
    np.save(source / "prediction.npy", np.zeros((8, 4, 16, 16)))
    with pytest.raises(ValueError, match="cached prediction SHA-256"):
        rescore_cached_probes(prepared, source, tmp_path / "rescored")
    assert not (tmp_path / "rescored").exists()

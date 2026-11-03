from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from raw_fusion.finetune.a10_sc450_audit import (
    A10AuditConfig,
    A10Source,
    _correlation,
    _correlation_authenticity,
    _correlation_report,
    _window_report,
)
from raw_fusion.finetune.cross_sensor import NativeSequenceSpec


def _source(tmp_path: Path, role: str) -> A10Source:
    path = tmp_path / f"{role}.raw"
    path.touch()
    spec = NativeSequenceSpec(
        name=role,
        sensor="sc450ai",
        path=path,
        width=64,
        height=64,
        frames_used=8,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=258.0,
        reference_mode="stress_only",
        clip_starts=(0,),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )
    return A10Source(role, role, "0" * 64, (0, 0, 64, 64), spec)


def _config(tmp_path: Path) -> A10AuditConfig:
    return A10AuditConfig(
        dataset_root=str(tmp_path),
        noise_profile_path=str(tmp_path / "profile.json"),
        noise_profile_sha256="0" * 64,
        output=str(tmp_path / "audit.json"),
        sources=(
            _source(tmp_path, "noise_statistics_source"),
            _source(tmp_path, "hard_holdout"),
        ),
        window_length=8,
        window_stride=4,
        pair_stride=4,
        spatial_sample_stride=1,
        semantic_constraints={
            "dataset_read_only": True,
            "old_teacher_used_as_clean_target": False,
            "dynamic_temporal_mean_used_as_clean_target": False,
            "hard_holdout_used_for_training": False,
            "training_launched": False,
        },
    )


def test_a10_config_requires_one_noise_source_and_one_holdout(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.validate()
    invalid = replace(config, sources=(config.sources[0], config.sources[0]))
    with pytest.raises(ValueError, match="source names must be unique"):
        invalid.validate()


def test_a10_config_rejects_training_semantics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    constraints = dict(config.semantic_constraints or {})
    constraints["training_launched"] = True
    with pytest.raises(ValueError, match="semantic constraints"):
        replace(config, semantic_constraints=constraints).validate()


def test_correlation_detects_matching_signal() -> None:
    values = np.arange(512, dtype=np.float32)
    assert _correlation(values, values) == pytest.approx(1.0)
    assert _correlation(values, -values) == pytest.approx(-1.0)


def test_window_report_ranks_stable_clip_above_dynamic_clip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    base = np.full((4, 24, 24), 400.0, dtype=np.float32)
    stable = np.stack(
        [base + rng.normal(0.0, 4.0, base.shape) for _ in range(8)]
    )
    dynamic = stable.copy()
    dynamic[:, :, 8:16, 8:16] += np.arange(8, dtype=np.float32)[:, None, None, None] * 30.0
    frames = np.concatenate((stable, dynamic), axis=0).astype(np.uint16)
    config = replace(_config(tmp_path), window_stride=8)
    report = _window_report(
        frames,
        np.full(4, 258.0, dtype=np.float32),
        np.full(4, 1.0, dtype=np.float32),
        np.full(4, 16.0, dtype=np.float32),
        config,
    )
    assert report["best_windows"][0]["start"] == 0
    assert report["clean_reference_eligible"] is False


def test_correlation_report_does_not_induce_highpass_neighbor_correlation(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    frames = rng.normal(500.0, 20.0, (16, 4, 64, 64)).astype(np.uint16)
    config = replace(
        _config(tmp_path),
        pair_stride=2,
        flat_percentile=100.0,
        temporal_outlier_percentile=99.9,
    )
    report = _correlation_report(
        frames,
        (0, 16),
        np.full(4, 258.0, dtype=np.float32),
        config,
    )
    for channel in report["channels"].values():
        assert abs(channel["horizontal_neighbor_correlation"]) < 0.05
        assert abs(channel["vertical_neighbor_correlation"]) < 0.05


def test_correlation_authenticity_separates_spatial_and_cross_channel_gates(
    tmp_path: Path,
) -> None:
    channel = {
        "horizontal_neighbor_correlation": 0.03,
        "vertical_neighbor_correlation": 0.02,
        "median_row_common_sigma_ratio": 0.20,
        "median_column_common_sigma_ratio": 0.18,
    }
    train = {
        "channels": {name: dict(channel) for name in ("R", "Gr", "Gb", "B")},
        "cross_channel_correlation": np.eye(4).tolist(),
    }
    validation = {
        "channels": {name: dict(channel) for name in ("R", "Gr", "Gb", "B")},
        "cross_channel_correlation": np.eye(4).tolist(),
    }
    validation["cross_channel_correlation"][0][1] = 0.10
    validation["cross_channel_correlation"][1][0] = 0.10
    report = _correlation_authenticity(train, validation, _config(tmp_path))
    assert report["per_channel_spatial_structure_passed"] is True
    assert report["cross_channel_coupling_passed"] is False
    assert report["passed"] is False

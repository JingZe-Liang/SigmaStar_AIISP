from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from raw_fusion.finetune.a10b_degradation import (
    A10bConfig,
    _PairMoments,
    _fit_spatial_model,
    generate_degraded_frame,
    generate_unit_spatial_noise,
)


def _config(tmp_path: Path) -> A10bConfig:
    return A10bConfig(
        dataset_root="/HardDisk/jingzeliang/Data/Company_Dataset/dataset",
        a10_config_path=str(tmp_path / "a10_config.json"),
        a10_config_sha256="0" * 64,
        a10_audit_path=str(tmp_path / "audit.json"),
        a10_audit_sha256="0" * 64,
        noise_profile_path=str(tmp_path / "profile.json"),
        noise_profile_sha256="0" * 64,
        output_dir=str(tmp_path / "output"),
        noise_source_name="noise",
        hard_holdout_name="holdout",
        train_frame_range=(0, 120),
        validation_frame_range=(120, 150),
        hard_holdout_frame_range=(172, 180),
        pair_stride=4,
        metric_sample_stride=4,
        correlation_lags=(1, 2, 3, 4, 5, 6, 7, 8),
        gate_correlation_lags=(1, 2, 4, 8),
        flat_percentile=35.0,
        temporal_outlier_percentile=99.5,
        signal_bin_edges_dn=(0.0, 16.0, 32.0, 64.0),
        min_bin_pixels=32,
        validation_replicas=2,
        hard_holdout_replicas=1,
        seed=7,
        spectral_floor=0.25,
        quantization_step_dn=4.0,
        max_median_relative_sigma_error=0.10,
        max_p90_relative_sigma_error=0.25,
        max_aggregate_median_relative_sigma_error=0.075,
        max_aggregate_p90_relative_sigma_error=0.20,
        max_weighted_correlation_mae=0.01,
        max_lag1_correlation_absolute_error=0.015,
        max_row_column_ratio_error=0.08,
        max_normalized_quantile_error=0.25,
        max_cross_channel_correlation=0.05,
        min_spatial_rmse_improvement_over_q4_iid=0.20,
        min_validation_pair_wins=2,
        semantic_constraints={
            "dataset_read_only": True,
            "old_teacher_used_as_clean_target": False,
            "dynamic_temporal_mean_used_as_clean_target": False,
            "adjacent_pair_mean_used_as_clean_target": False,
            "hard_holdout_used_for_fitting": False,
            "model_training_launched": False,
        },
    )


def test_a10b_config_rejects_overlapping_fit_and_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, validation_frame_range=(119, 150)).validate()


def test_a10b_config_rejects_training_semantics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    constraints = dict(config.semantic_constraints)
    constraints["model_training_launched"] = True
    with pytest.raises(ValueError, match="semantic constraints"):
        replace(config, semantic_constraints=constraints).validate()


def test_pair_moments_recovers_linear_correlation() -> None:
    values = np.arange(1024, dtype=np.float32)
    moments = _PairMoments()
    moments.update(values, 2.0 * values + 3.0)
    assert moments.correlation() == pytest.approx(1.0)


def test_spatial_model_uses_one_shared_bayer_spectrum(tmp_path: Path) -> None:
    config = _config(tmp_path)
    channels = {}
    for channel_index, name in enumerate(("R", "Gr", "Gb", "B")):
        channels[name] = {
            "horizontal_correlation": {
                str(lag): 0.01 * (channel_index + 1) / lag
                for lag in config.correlation_lags
            },
            "vertical_correlation": {
                str(lag): 0.005 * (channel_index + 1) / lag
                for lag in config.correlation_lags
            },
        }
    model = _fit_spatial_model({"channels": channels}, config)
    first = model["channels"]["R"]
    for name in ("Gr", "Gb", "B"):
        assert model["channels"][name] == first
    assert first["horizontal_autocorrelation"][0] == pytest.approx(0.025)


def test_spatial_generator_is_deterministic_and_unit_variance() -> None:
    first = generate_unit_spatial_noise(
        (96, 128),
        [0.04, 0.02, 0.01, 0.0],
        [0.03, 0.01, 0.0, 0.0],
        0.25,
        np.random.default_rng(17),
    )
    second = generate_unit_spatial_noise(
        (96, 128),
        [0.04, 0.02, 0.01, 0.0],
        [0.03, 0.01, 0.0, 0.0],
        0.25,
        np.random.default_rng(17),
    )
    assert np.array_equal(first, second)
    assert float(first.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(first.std()) == pytest.approx(1.0, abs=1e-6)


def test_degraded_frame_is_q4_and_cross_channel_independent() -> None:
    conditioning = np.full((4, 64, 64), 800.0, dtype=np.float32)
    black = np.full(4, 256.0, dtype=np.float32)
    shot = np.full(4, 10.0, dtype=np.float32)
    read = np.full(4, 100.0, dtype=np.float32)
    output = generate_degraded_frame(
        conditioning,
        black,
        shot,
        read,
        np.random.default_rng(23),
        spatial_model=None,
        quantization_step_dn=4.0,
        spectral_floor=0.25,
    )
    assert output.shape == conditioning.shape
    assert np.all(output >= 0.0)
    assert np.all(output <= 4092.0)
    assert np.all(np.mod(output, 4.0) == 0.0)
    matrix = np.corrcoef(output.reshape(4, -1))
    off_diagonal = matrix[~np.eye(4, dtype=bool)]
    assert float(np.max(np.abs(off_diagonal))) < 0.08

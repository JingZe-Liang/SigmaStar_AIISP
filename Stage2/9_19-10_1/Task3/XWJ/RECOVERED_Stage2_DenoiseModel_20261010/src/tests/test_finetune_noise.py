"""Unit tests for A2 model interpolation and synthetic quantization."""

import numpy as np
import pytest

from raw_fusion.finetune.noise import (
    _interpolate_log_gain,
    generate_noisy_frame,
)


def _model() -> dict:
    gains = {"100": 1.0, "200": 2.0, "400": 4.0}
    return {
        "gain_factors": gains,
        "gain_labels": ["100", "200", "400"],
        "empirical_shot_coefficients": {name: {"100": 1.0, "200": 2.0, "400": 4.0} for name in ("R", "Gr", "Gb", "B")},
        "read_variance_dn2": {name: {"100": 1.0, "200": 4.0, "400": 16.0} for name in ("R", "Gr", "Gb", "B")},
        "channel_correlation": {"aggregate": np.eye(4).tolist()},
    }


def test_log_gain_interpolation_preserves_exact_non_monotonic_points() -> None:
    table = {"100": 1.0, "200": 0.25, "400": 2.0}
    gains = {"100": 1.0, "200": 2.0, "400": 4.0}
    assert _interpolate_log_gain(table, 2.0, gains) == pytest.approx(0.25)
    assert _interpolate_log_gain(table, 4.0, gains) == pytest.approx(2.0)


def test_generate_noisy_frame_is_quantized_and_clipped() -> None:
    clean = np.full((4, 540, 960), 300.0, dtype=np.float32)
    result = generate_noisy_frame(
        clean,
        _model(),
        "200",
        {"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0},
        None,
        np.random.default_rng(3),
    )
    assert result.dtype == np.uint16
    assert result.shape == clean.shape
    assert int(result.min()) >= 0 and int(result.max()) <= 4095

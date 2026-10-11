import numpy as np
import pytest

from raw_fusion.finetune.a7_motion_fusion import (
    fuse_motion_regions,
    fusion_shift_summary,
    motion_temporal_summary,
)


def test_fusion_preserves_static_pixels_exactly_and_limits_motion_residual() -> None:
    baseline = np.full((3, 4, 4, 5), 0.25, dtype=np.float32)
    teacher = np.full_like(baseline, 0.75)
    weights = np.zeros((3, 4, 5), dtype=np.float32)
    weights[:, 1:3, 2:4] = 1.0
    black = np.full(4, 256.0, dtype=np.float32)
    candidate = fuse_motion_regions(
        baseline,
        teacher,
        weights,
        black,
        alpha=0.5,
        residual_clip_dn=40.0,
    )
    np.testing.assert_array_equal(candidate[:, :, weights[0] == 0], baseline[:, :, weights[0] == 0])
    scale = 4095.0 - 256.0
    expected = 0.25 + 20.0 / scale
    np.testing.assert_allclose(candidate[:, :, 1:3, 2:4], expected, atol=1e-7)


def test_fusion_shift_summary_reports_zero_static_shift() -> None:
    baseline = np.zeros((2, 4, 3, 3), dtype=np.float32)
    candidate = baseline.copy()
    candidate[:, :, 1, 1] = 0.01
    motion = np.zeros((2, 3, 3), dtype=np.float32)
    motion[:, 1, 1] = 1.0
    summary = fusion_shift_summary(
        candidate,
        baseline,
        motion,
        np.full(4, 256.0, dtype=np.float32),
    )
    assert summary["motion_mean_abs_dn"] > 0.0
    assert summary["static_mean_abs_dn"] == 0.0
    assert summary["static_max_abs_dn"] == 0.0


def test_motion_temporal_summary_increases_when_motion_is_restored() -> None:
    input_video = np.zeros((3, 4, 2, 2), dtype=np.float32)
    input_video[1:, :, 0, 0] = 1.0
    baseline = input_video * 0.2
    candidate = input_video * 0.5
    motion = np.zeros((3, 2, 2), dtype=np.float32)
    motion[:, 0, 0] = 1.0
    baseline_summary = motion_temporal_summary(baseline, input_video, motion)
    candidate_summary = motion_temporal_summary(candidate, input_video, motion)
    assert candidate_summary["motion_to_input_ratio"] > baseline_summary["motion_to_input_ratio"]


def test_fusion_rejects_invalid_alpha() -> None:
    array = np.zeros((2, 4, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="alpha"):
        fuse_motion_regions(
            array,
            array,
            np.zeros((2, 2, 2), dtype=np.float32),
            np.full(4, 256.0, dtype=np.float32),
            alpha=1.1,
            residual_clip_dn=32.0,
        )

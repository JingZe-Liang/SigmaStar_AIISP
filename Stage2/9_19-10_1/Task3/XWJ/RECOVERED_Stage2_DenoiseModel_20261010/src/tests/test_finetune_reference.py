"""Unit tests for A1 reference math on small synthetic arrays."""

import numpy as np

from raw_fusion.finetune.reference import (
    FpnMaps,
    TemporalReference,
    apply_fpn_correction,
    estimate_fpn_maps,
    measure_split_half_sigma,
)


def _temporal_reference(mean: np.ndarray) -> TemporalReference:
    height, width = mean.shape[-2:]
    return TemporalReference(
        mean=mean.astype(np.float32),
        even_mean=mean.astype(np.float32),
        odd_mean=mean.astype(np.float32),
        confidence=np.ones((height, width), dtype=np.float32),
        valid_region=np.ones((height, width), dtype=bool),
        black_levels={"R": 10.0, "Gr": 10.0, "Gb": 10.0, "B": 10.0},
        frames_total=20,
        exposure_scales=np.ones((20, 4), dtype=np.float32),
    )


def test_apply_fpn_correction_subtracts_offset() -> None:
    reference = np.full((4, 4, 5), 100.0, dtype=np.float32)
    offset = np.full_like(reference, 1.5)
    corrected = apply_fpn_correction(reference, FpnMaps(offset, np.zeros_like(offset), offset))
    np.testing.assert_allclose(corrected, 98.5)


def test_split_half_sigma_uses_half_difference() -> None:
    mean = np.full((4, 540, 960), 100.0, dtype=np.float32)
    reference = _temporal_reference(mean)
    even = mean + 0.5
    odd = mean - 0.5
    sigma = measure_split_half_sigma(reference, even, odd)
    assert all(value == 0.0 for value in sigma["per_channel_dn"].values())


def test_fpn_estimate_keeps_channel_shapes() -> None:
    black = _temporal_reference(np.full((4, 20, 20), 11.0, dtype=np.float32))
    flat = _temporal_reference(np.full((4, 20, 20), 211.0, dtype=np.float32))
    target = _temporal_reference(np.full((4, 20, 20), 111.0, dtype=np.float32))
    maps = estimate_fpn_maps(black, flat, target)
    assert maps.dsnu_dn.shape == (4, 20, 20)
    assert maps.prnu_fraction.shape == (4, 20, 20)
    assert maps.reference_offset_dn.shape == (4, 20, 20)

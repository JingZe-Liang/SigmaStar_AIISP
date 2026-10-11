"""Registration tests: synthetic sign/semantics checks + real cross-gain shifts.

Real-file tests read the dataset strictly read-only.
"""

from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.registration import (
    overlap_crops,
    register_streams,
)
from raw_fusion.finetune.stats import compute_basic_stats
from raw_fusion.finetune.stream_io import open_stream

DATASET = Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset")
SHDARKROOM = DATASET / "04_Sigmastar_7_30" / "shdarkroom"


def _means(gain: str, max_frames: int = 40):
    adapter = resolve_adapter("04", "shdarkroom", gain)
    raw = sorted((SHDARKROOM / gain).glob("*.raw"))[0]
    reader, _meta = open_stream(raw, adapter)
    return compute_basic_stats(reader, adapter, chunk_frames=10, max_frames=max_frames, black_sample_frames=4)


def test_synthetic_registration_semantics() -> None:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 4095, size=(120, 160)).astype(np.float32)
    # Moving content displaced +5 px right, +3 px down in packed coordinates
    # (= +10/+6 sensor px in the reported result).
    moving = np.roll(base, shift=(3, 5), axis=(0, 1))
    black = {"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0}
    mean4 = np.stack([base] * 4)
    moving4 = np.stack([moving] * 4)
    result = register_streams(mean4, black, moving4, black, baseline_label="base")
    assert abs(result.shift_x - 10.0) < 1.0
    assert abs(result.shift_y - 6.0) < 1.0
    assert (result.integer_shift_x, result.integer_shift_y) == (5, 3)
    assert result.paired and result.ncc > 0.9


def test_overlap_crops_consistency() -> None:
    base = np.arange(200, dtype=np.float32).reshape(10, 20)
    moving = np.roll(base, shift=(2, 3), axis=(0, 1))  # content shift (+3, +2)
    base_crop, moving_crop = overlap_crops(base, moving, 3, 2)
    np.testing.assert_array_equal(base_crop, moving_crop)


def test_real_632x_shift_vs_645x() -> None:
    # Survey: 632x needs a +62.7 px alignment shift (sensor px) onto 645x, i.e.
    # its content sits 62.7 px left and ~1.7 px up: content displacement is
    # negative here.
    baseline = _means("645x")
    moving = _means("632x")
    result = register_streams(baseline.temporal_mean, baseline.black_levels,
                              moving.temporal_mean, moving.black_levels, "645x")
    assert result.paired
    assert abs(result.shift_x + 62.7) <= 4.0
    assert abs(result.shift_y + 1.7) <= 4.0
    assert result.integer_shift_x == round(result.shift_x / 2)


def test_real_10x_aligned_with_645x() -> None:
    baseline = _means("645x")
    moving = _means("10x")
    result = register_streams(baseline.temporal_mean, baseline.black_levels,
                              moving.temporal_mean, moving.black_levels, "645x")
    assert result.paired
    assert abs(result.shift_x) <= 2.0
    assert abs(result.shift_y) <= 2.0

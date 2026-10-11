from pathlib import Path

import numpy as np
import pytest

from raw_fusion.finetune.a4 import _read_aligned_video


SUPERVISION = Path("artifacts/finetune_mis20s1/supervision")


def test_read_aligned_video_supports_disjoint_holdout() -> None:
    first, first_overlap = _read_aligned_video(
        SUPERVISION, "645x", 2, start_frame=0
    )
    holdout, holdout_overlap = _read_aligned_video(
        SUPERVISION, "645x", 2, start_frame=160
    )

    assert first.shape == holdout.shape == (2, 4, 540, 960)
    assert first_overlap.shape == holdout_overlap.shape == (2, 1, 540, 960)
    assert not np.array_equal(first, holdout)


def test_read_aligned_video_rejects_out_of_range_start() -> None:
    with pytest.raises(ValueError, match="start_frame"):
        _read_aligned_video(SUPERVISION, "645x", 1, start_frame=200)

from __future__ import annotations

import numpy as np


def test_mog2_cache_round_trip_and_progress_are_resumable(tmp_path) -> None:
    from raw_fusion.v2.data_first_mog2_cache import MOG2MaskCache

    cache = MOG2MaskCache.create(
        tmp_path,
        source_sha256={"128x": "a" * 64, "645x": "b" * 64},
        target_frames={"128x": (58,), "645x": (58,)},
        mog2_config={"history": 50, "var_threshold": 64.0, "detect_shadows": False},
    )
    mask = np.zeros((540, 960), dtype=np.uint8)
    mask[10, 20] = 255
    cache.write_mask("128x", 58, mask)
    cache.write_progress(completed=1, total=2, elapsed_seconds=2.0)

    restored = MOG2MaskCache.open(tmp_path)
    assert np.array_equal(restored.read_mask("128x", 58), mask)
    assert restored.is_complete("128x", 58)
    assert restored.progress_path.read_text().count("\n") == 1

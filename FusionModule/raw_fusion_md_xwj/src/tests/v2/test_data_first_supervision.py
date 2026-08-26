from __future__ import annotations

import numpy as np


def _frames(height: int = 32, width: int = 32) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for index in range(200):
        frame = np.full((4, height, width), 512, dtype=np.uint16)
        frame[1] += np.uint16(index % 3)
        frame[3] += np.uint16(index % 3)
        if index == 58:
            frame[:, 8:24, 8:24] += np.uint16(120)
        result[index] = frame
    return result


def test_mog2_supervision_returns_finite_labels_without_formal_artifacts() -> None:
    from raw_fusion.v2.data_first_supervision import (
        DataFirstSupervision,
        MOG2SupervisionConfig,
        MOG2SupervisionGenerator,
    )
    from raw_fusion.v2.md import Mog2ConfigV2

    frames = _frames()
    generator = MOG2SupervisionGenerator(
        lambda condition, frame: frames[frame],
        config=MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)),
    )
    result = generator.supervise("128x", 58)
    assert isinstance(result, DataFirstSupervision)
    assert result.pixel_state.shape == (32, 32)
    assert result.cell_state.shape == (1, 1, 1)
    assert result.policy_alpha_target.shape == (1, 1, 1)
    assert result.hf_target.shape == (4, 32, 32)
    assert np.isfinite(result.hf_target).all()
    assert result.policy_alpha_valid.dtype == np.uint8
    assert result.md_mask.dtype == np.uint8
    assert not hasattr(result, "model_inputs")


def test_supervision_is_deterministic_for_same_frame_sequence() -> None:
    from raw_fusion.v2.data_first_supervision import MOG2SupervisionConfig, MOG2SupervisionGenerator
    from raw_fusion.v2.md import Mog2ConfigV2

    frames = _frames()
    config = MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50))
    first = MOG2SupervisionGenerator(lambda _condition, frame: frames[frame], config=config).supervise("645x", 58)
    second = MOG2SupervisionGenerator(lambda _condition, frame: frames[frame], config=config).supervise("645x", 58)
    assert np.array_equal(first.pixel_state, second.pixel_state)
    assert np.array_equal(first.policy_alpha_class, second.policy_alpha_class)


def test_supervision_rebuilds_labels_from_cached_mog2_mask() -> None:
    from raw_fusion.v2.data_first_supervision import MOG2SupervisionConfig, MOG2SupervisionGenerator
    from raw_fusion.v2.md import Mog2ConfigV2

    frames = _frames()
    generator = MOG2SupervisionGenerator(lambda _condition, frame: frames[frame], config=MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)))
    online = generator.supervise("128x", 58)
    cached = generator.supervise_from_mask("128x", 58, online.md_mask)

    assert np.array_equal(cached.md_mask, online.md_mask)
    assert np.array_equal(cached.policy_alpha_class, online.policy_alpha_class)
    assert np.array_equal(cached.valid_bits, online.valid_bits)

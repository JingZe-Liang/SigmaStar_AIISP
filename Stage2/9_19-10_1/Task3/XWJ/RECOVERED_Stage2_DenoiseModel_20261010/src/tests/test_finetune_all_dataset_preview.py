import numpy as np

from raw_fusion.finetune.all_dataset_preview import (
    canonical_to_mosaic,
    clip_starts,
    exposure_target_before_tone_map,
    render_traditional_isp,
)


def test_clip_starts_cover_first_middle_and_last() -> None:
    assert clip_starts(200, 8, 3) == (0, 96, 192)
    assert clip_starts(50, 8, 3) == (0, 21, 42)


def test_reinhard_exposure_target_is_inverted() -> None:
    assert np.isclose(exposure_target_before_tone_map(0.8, "reinhard"), 4.0)
    assert np.isclose(exposure_target_before_tone_map(0.8, "hard-clip"), 0.8)


def test_canonical_to_mosaic_respects_project_channel_order() -> None:
    packed = np.stack(
        [np.full((2, 3), value, dtype=np.float32) for value in (1, 2, 3, 4)]
    )
    rggb = canonical_to_mosaic(packed, "RGGB")
    assert np.all(rggb[0::2, 0::2] == 1)
    assert np.all(rggb[0::2, 1::2] == 2)
    assert np.all(rggb[1::2, 0::2] == 3)
    assert np.all(rggb[1::2, 1::2] == 4)

    grbg = canonical_to_mosaic(packed, "GRBG")
    assert np.all(grbg[0::2, 0::2] == 2)
    assert np.all(grbg[0::2, 1::2] == 1)
    assert np.all(grbg[1::2, 0::2] == 4)
    assert np.all(grbg[1::2, 1::2] == 3)


def test_traditional_isp_render_has_expected_shape_and_dtype() -> None:
    frame = np.full((4, 64, 80), 800.0, dtype=np.float32)
    image = render_traditional_isp(
        frame,
        np.full(4, 256.0, dtype=np.float32),
        (1.8, 1.0, 1.7),
        1.0,
        "RGGB",
        highlight_recovery=True,
        tone_map="reinhard",
    )
    assert image.shape == (128, 160, 3)
    assert image.dtype == np.uint8
    assert np.isfinite(image).all()

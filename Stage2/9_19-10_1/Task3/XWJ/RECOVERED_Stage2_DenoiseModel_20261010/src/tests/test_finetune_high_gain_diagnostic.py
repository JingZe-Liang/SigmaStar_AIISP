import numpy as np
import torch

from raw_fusion.finetune.high_gain_diagnostic import (
    evaluate_variants,
    infer_frame_repeated,
    infer_tiled_clip,
    spatial_blend_window,
    spatial_starts,
)


class IdentityVideoModel(torch.nn.Module):
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return video


def test_spatial_starts_cover_trailing_edge() -> None:
    assert spatial_starts(11, 4, 3) == (0, 3, 6, 7)
    assert spatial_starts(4, 4, 4) == (0,)


def test_hann_window_is_positive_and_center_weighted() -> None:
    window = spatial_blend_window(8, "hann", 0.05)
    assert window.shape == (8, 8)
    assert float(window.min()) > 0.0
    assert np.isclose(float(window.max()), 1.0)
    assert window[4, 4] > window[0, 0]


def test_weighted_tiled_identity_inference_preserves_clip() -> None:
    rng = np.random.default_rng(7)
    clip = rng.random((8, 4, 9, 11), dtype=np.float32)
    prediction = infer_tiled_clip(
        IdentityVideoModel(),
        clip,
        torch.device("cpu"),
        tile=4,
        stride=3,
        tile_batch=5,
        blend_mode="hann",
        blend_floor=0.05,
    )
    assert np.allclose(prediction, clip, atol=1e-6)


def test_frame_repeated_identity_inference_preserves_frame() -> None:
    rng = np.random.default_rng(9)
    frame = rng.random((4, 9, 11), dtype=np.float32)
    prediction = infer_frame_repeated(
        IdentityVideoModel(),
        frame,
        torch.device("cpu"),
        temporal_output_index=4,
        tile=4,
        stride=3,
        tile_batch=5,
        blend_mode="hann",
        blend_floor=0.05,
    )
    assert np.allclose(prediction, frame, atol=1e-6)


def test_evaluation_reports_temporal_and_grid_metrics() -> None:
    rng = np.random.default_rng(11)
    input_video = rng.random((8, 4, 16, 16), dtype=np.float32)
    variants = {
        "input": input_video,
        "a3_uniform": input_video * 0.9,
        "a3_weighted": input_video * 0.8,
        "a3_frame_repeated": input_video * 0.7,
    }
    result = evaluate_variants(variants, {"grid": (8, 4)})
    assert result["shape"] == [8, 4, 16, 16]
    assert result["variants"]["input"]["legal_raw_fraction"] == 1.0
    assert result["variants"]["input"]["highpass_rms"] > 0.0
    assert result["variants"]["a3_uniform"]["mae_vs_a3_uniform"] == 0.0
    assert "amplification_vs_input" in result["variants"]["a3_weighted"][
        "tile_discontinuity"
    ]["grid"]

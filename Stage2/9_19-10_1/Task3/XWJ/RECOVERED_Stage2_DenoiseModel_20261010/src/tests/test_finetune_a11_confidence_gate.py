import numpy as np

from raw_fusion.finetune.a11_confidence_gate import (
    ConfidenceGateParams,
    build_confidence_weights,
    evaluate_gate,
    fuse_branches,
)


def _params() -> ConfidenceGateParams:
    return ConfidenceGateParams(
        temporal_gap=2,
        residual_low_z=1.5,
        residual_high_z=4.0,
        gradient_low=0.1,
        gradient_high=0.5,
        structure_low=0.001,
        structure_high=0.01,
        flow_low_px=0.5,
        flow_high_px=2.0,
        fb_low_px=0.5,
        fb_high_px=2.0,
        photometric_low_z=3.0,
        photometric_high_z=8.0,
        seed_threshold=0.2,
        min_component_area=8,
        dilation_radius=2,
        feather_sigma=1.0,
        min_weight=0.01,
        max_weight_residual=0.35,
        max_weight_gradient=0.5,
        max_weight_full=0.65,
    )


def test_confidence_gate_targets_moving_structure_and_protects_static_core() -> None:
    video = np.full((8, 4, 48, 64), 0.1, dtype=np.float32)
    for frame_index in range(8):
        x = 8 + frame_index * 3
        video[frame_index, :, 16:32, x : x + 10] = 0.7
    weights, masks, _ = build_confidence_weights(video, _params())
    full = weights["full"]
    assert float(full.max()) > 0.1
    assert np.all(full[masks["static_core"]] == 0.0)
    assert float((full > 0.0).mean()) < 0.5


def test_fuse_branches_is_bounded_and_respects_zero_weight() -> None:
    baseline = np.full((3, 4, 8, 9), 0.2, dtype=np.float32)
    repeated = np.full_like(baseline, 0.8)
    weights = np.zeros((3, 8, 9), dtype=np.float32)
    weights[:, 2:6, 3:7] = 0.5
    candidate = fuse_branches(baseline, repeated, weights)
    assert np.all(candidate[:, :, :2] == baseline[:, :, :2])
    assert np.allclose(candidate[:, :, 2:6, 3:7], 0.5)
    assert float(candidate.min()) >= 0.0
    assert float(candidate.max()) <= 1.0


def test_evaluate_gate_reports_contamination_and_boundary_metrics() -> None:
    rng = np.random.default_rng(17)
    input_video = rng.random((8, 4, 96, 96), dtype=np.float32)
    weighted = input_video * 0.8
    repeated = input_video * 0.9
    weights = np.zeros((8, 96, 96), dtype=np.float32)
    weights[:, 24:72, 24:72] = 0.5
    gated = fuse_branches(weighted, repeated, weights)
    masks = {
        "motion_core": weights >= 0.5,
        "static_core": weights == 0.0,
        "residual_seed": weights >= 0.5,
        "gradient_seed": weights >= 0.5,
    }
    result = evaluate_gate(
        {
            "input": input_video,
            "a3_uniform": weighted,
            "a3_weighted": weighted,
            "a3_frame_repeated": repeated,
            "a11_gated": gated,
        },
        {"full": weights},
        masks,
        np.full(4, 256.0, dtype=np.float32),
    )
    metrics = result["mask_ablations"]["full"]
    assert metrics["coverage_w_gt_0"] == 0.25
    assert metrics["static_nonzero_weight_fraction"] == 0.0
    assert metrics["shift_dn_static"]["max"] == 0.0
    assert metrics["boundary"]["boundary_fraction"] > 0.0

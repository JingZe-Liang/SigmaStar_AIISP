from pathlib import Path

import numpy as np
import torch

from raw_fusion.finetune.cross_sensor import (
    NativeSequenceSpec,
    build_motion_aware_reference,
    build_stress_profile,
    channel_vector,
    contiguous_clip_indices,
    denormalise_native,
    evaluate_stress_prediction,
    infer_stress_clips,
    normalise_native_dn,
    open_sequence,
    read_normalized_clip,
    render_fixed_isp,
    split_clip_indices,
)


def _spec(path: Path) -> NativeSequenceSpec:
    return NativeSequenceSpec(
        name="fixture",
        sensor="fixture",
        path=path,
        width=128,
        height=128,
        frames_used=16,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="motion_aware_split_half",
        clip_starts=(0,),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )


def test_explicit_dimensions_and_disjoint_clip_indices(tmp_path: Path) -> None:
    path = tmp_path / "stream_without_dimensions.raw"
    frames = np.full((16, 128, 128), 512, dtype="<u2")
    frames.tofile(path)
    spec = _spec(path)

    assert open_sequence(spec).frames == 16
    assert split_clip_indices(spec, 0) == [[0, 2, 4, 6, 8, 10, 12, 14]]
    assert split_clip_indices(spec, 1) == [[1, 3, 5, 7, 9, 11, 13, 15]]
    assert set(split_clip_indices(spec, 0)[0]).isdisjoint(split_clip_indices(spec, 1)[0])


def test_stress_clip_indices_are_contiguous(tmp_path: Path) -> None:
    spec = NativeSequenceSpec(
        name="stress",
        sensor="fixture",
        path=tmp_path / "stress.raw",
        width=128,
        height=128,
        frames_used=12,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="stress_only",
        clip_starts=(3,),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )
    spec.validate()
    assert contiguous_clip_indices(spec) == [[3, 4, 5, 6, 7, 8, 9, 10]]


def test_identity_stress_prediction_has_unit_diagnostic_ratios(tmp_path: Path) -> None:
    path = tmp_path / "stress.raw"
    yy, xx = np.mgrid[:128, :128]
    texture = ((xx // 4 + yy // 4) % 2) * 200
    frames = np.stack(
        [600 + xx + yy + texture + index * 3 for index in range(16)], axis=0
    ).astype("<u2")
    frames.tofile(path)
    spec = NativeSequenceSpec(
        name="stress",
        sensor="fixture",
        path=path,
        width=128,
        height=128,
        frames_used=16,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="stress_only",
        clip_starts=(0, 8),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )
    profile, _ = build_stress_profile(spec, chunk_frames=4)
    reader = open_sequence(spec)
    black = channel_vector(profile.black_levels)
    prediction = np.concatenate(
        [
            read_normalized_clip(reader, spec.adapter, indices, black)
            for indices in contiguous_clip_indices(spec)
        ],
        axis=0,
    )
    prediction_path = tmp_path / "prediction.npy"
    np.save(prediction_path, prediction.astype(np.float16))

    metrics = evaluate_stress_prediction(spec, profile, prediction_path)

    assert abs(metrics["temporal_mad_ratio"] - 1.0) < 0.01
    assert abs(metrics["motion_temporal_ratio"] - 1.0) < 0.01
    assert abs(metrics["spatial_high_frequency_ratio"] - 1.0) < 0.01
    assert metrics["legal_raw_fraction"] == 1.0


def test_stress_inference_resumes_completed_clips(tmp_path: Path) -> None:
    path = tmp_path / "stress.raw"
    frames = np.full((16, 128, 128), 800, dtype="<u2")
    frames.tofile(path)
    spec = NativeSequenceSpec(
        name="stress",
        sensor="fixture",
        path=path,
        width=128,
        height=128,
        frames_used=16,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="stress_only",
        clip_starts=(0, 8),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )
    profile, _ = build_stress_profile(spec, chunk_frames=4)
    output_path = tmp_path / "prediction.npy"

    class Identity(torch.nn.Module):
        def forward(self, values):
            return values

    class MustNotRun(torch.nn.Module):
        def forward(self, values):
            raise AssertionError("completed clips should be resumed")

    infer_stress_clips(
        Identity(),
        spec,
        profile,
        torch.device("cpu"),
        output_path,
        tile_batch=1,
        resume_key="same-provenance",
    )
    first = np.load(output_path).copy()
    infer_stress_clips(
        MustNotRun(),
        spec,
        profile,
        torch.device("cpu"),
        output_path,
        tile_batch=1,
        resume_key="same-provenance",
    )
    np.testing.assert_array_equal(np.load(output_path), first)


def test_native_normalization_round_trip() -> None:
    black = np.asarray([250.0, 252.0, 254.0, 256.0], dtype=np.float32)
    values = np.linspace(300.0, 3500.0, 4 * 8 * 10, dtype=np.float32).reshape(4, 8, 10)
    restored = denormalise_native(normalise_native_dn(values, black), black)
    np.testing.assert_allclose(restored, values, rtol=0, atol=5e-4)


def test_motion_aware_reference_excludes_moving_region(tmp_path: Path) -> None:
    path = tmp_path / "moving.raw"
    frames = np.full((16, 128, 128), 700, dtype=np.uint16)
    for index in range(16):
        x0 = 8 + index * 4
        frames[index, 32:64, x0 : x0 + 16] = 2200
    frames.astype("<u2").tofile(path)

    reference, report = build_motion_aware_reference(_spec(path), chunk_frames=4)

    assert reference.mean.shape == (4, 64, 64)
    assert report["valid_fraction"] < 1.0
    assert report["motion_fraction_mean"] > 0.0
    assert reference.valid_region[0, 0]


def test_fixed_isp_is_deterministic() -> None:
    frame = np.full((4, 64, 64), 1200.0, dtype=np.float32)
    black = channel_vector({"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0})
    first = render_fixed_isp(frame, black, (1.2, 1.0, 1.0, 1.4), 2.0)
    second = render_fixed_isp(frame, black, (1.2, 1.0, 1.0, 1.4), 2.0)
    assert first.shape == (128, 128, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)

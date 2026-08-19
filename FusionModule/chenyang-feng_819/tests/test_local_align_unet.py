import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import raw_fusion_local_align as local_align
import render_local_align_comparison as local_align_render


def test_dual_branch_model_has_four_plane_gate_and_identity_start() -> None:
    model = local_align.LocalAlignResidualUNet(base_ch=4)
    inputs = torch.rand(2, 26, 16, 16)
    base = torch.rand(2, 4, 16, 16)

    output, gate = model(inputs, base)

    assert output.shape == base.shape
    assert gate.shape == base.shape
    torch.testing.assert_close(output, base, rtol=0.0, atol=0.0)


def test_zero_confidence_forces_identity_even_with_nonzero_residual() -> None:
    model = local_align.LocalAlignResidualUNet(base_ch=4)
    inputs = torch.rand(1, 26, 16, 16)
    inputs[:, 24:26] = 0.0
    base = torch.rand(1, 4, 16, 16)
    with torch.no_grad():
        model.residual.bias.fill_(1.0)

    output, gate = model(inputs, base)

    torch.testing.assert_close(output, base, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gate, torch.zeros_like(gate), rtol=0.0, atol=0.0)


def test_temporal_channels_reach_the_confidence_gated_fusion_module() -> None:
    torch.manual_seed(0)
    model = local_align.LocalAlignResidualUNet(base_ch=4)
    inputs = torch.rand(1, 26, 16, 16)
    inputs[:, 24:26] = 1.0
    base = torch.full((1, 4, 16, 16), 0.5)
    fusion_inputs: list[torch.Tensor] = []
    hook = model.fuse1.register_forward_hook(
        lambda module, args, output: fusion_inputs.append(args[0].detach().clone())
    )

    model(inputs, base)
    inputs[:, 8:24] += 5.0
    model(inputs, base)
    hook.remove()

    assert len(fusion_inputs) == 2
    torch.testing.assert_close(fusion_inputs[0][:, :4], fusion_inputs[1][:, :4])
    assert not torch.equal(fusion_inputs[0][:, 4:], fusion_inputs[1][:, 4:])


def test_fixed_split_contexts_are_disjoint() -> None:
    train_range, selection_range, report_range = local_align._la_fixed_splits(200)

    assert (train_range.start, train_range.stop - 1) == (2, 119)
    assert (selection_range.start, selection_range.stop - 1) == (135, 155)
    assert (report_range.start, report_range.stop - 1) == (169, 198)

    contexts = [
        {frame for current in frame_range for frame in range(current - 2, current + 2)}
        for frame_range in (train_range, selection_range, report_range)
    ]
    assert contexts[0].isdisjoint(contexts[1])
    assert contexts[0].isdisjoint(contexts[2])
    assert contexts[1].isdisjoint(contexts[2])
    with pytest.raises(ValueError, match="200-frame"):
        local_align._la_fixed_splits(199)


def test_patch_keeps_future_confidence_out_of_model_input(monkeypatch: pytest.MonkeyPatch) -> None:
    history_trust = np.full((1, 4, 4), 0.8, dtype=np.float32)
    input_frame = np.zeros((26, 4, 4), dtype=np.float32)
    base_frame = np.zeros((4, 4, 4), dtype=np.float32)
    future_target = np.ones((4, 4, 4), dtype=np.float32)
    future_trust = np.full((4, 4), 0.3, dtype=np.float32)

    monkeypatch.setattr(
        local_align,
        "_la_input_full",
        lambda sequence, current: (input_frame, base_frame, history_trust),
    )

    class Sequence:
        frames = 10

        def aligned_future_target(self, current: int,
                                  delta: int = 1) -> tuple[np.ndarray, np.ndarray]:
            return future_target, future_trust

    inp, base, target, returned_history, supervision = local_align._la_patch(
        Sequence(), 2, 0, 0, 4
    )

    assert inp.shape == (26, 4, 4)
    np.testing.assert_array_equal(inp[24:26], 0.0)
    np.testing.assert_array_equal(base, base_frame)
    np.testing.assert_array_equal(target, future_target)
    np.testing.assert_array_equal(returned_history, history_trust)
    np.testing.assert_allclose(supervision, 0.3, rtol=0.0, atol=1e-7)


def test_future_target_bundle_uses_both_future_observations() -> None:
    class Sequence:
        frames = 8

        def aligned_future_target(self, current: int,
                                  delta: int = 1) -> tuple[np.ndarray, np.ndarray]:
            value = np.full((4, 4, 4), float(delta), dtype=np.float32)
            trust = np.ones((4, 4), dtype=np.float32)
            return value, trust

    targets, trusts, pseudo, target_trust, variance = local_align._la_future_target_bundle(Sequence(), 2)

    assert targets.shape == (2, 4, 4, 4)
    assert trusts.shape == (2, 4, 4)
    assert variance.shape == (4, 4)
    np.testing.assert_allclose(pseudo, 1.5, rtol=0.0, atol=1e-6)
    assert float(target_trust.mean()) > 0.0


def test_rejected_checkpoint_copies_2dnr_raw_exactly(tmp_path: Path) -> None:
    frames, height, width = 3, 8, 8
    source = np.full((frames, height, width), 16 * 252, dtype=np.uint16)
    denoised = np.full((frames, height, width), 300, dtype=np.uint16)
    denoised[1, 0, 0] = 255
    denoised[2, 0, 1] = 299
    source_path = tmp_path / "source.raw"
    denoised_path = tmp_path / "denoised.raw"
    output_path = tmp_path / "output.raw"
    checkpoint_path = tmp_path / "rejected.pth"
    source.tofile(source_path)
    denoised.tofile(denoised_path)
    torch.save(
        {
            "architecture": "dual_branch_26ch_v1",
            "config": {"accepted_against_2dnr": False},
            "base_ch": 4,
            "stride": 1,
        },
        checkpoint_path,
    )

    local_align.infer_local_align_unet(
        source=str(source_path),
        denoised=str(denoised_path),
        model_path=str(checkpoint_path),
        output=str(output_path),
        frames=frames,
        height=height,
        width=width,
        stride=1,
        device="cpu",
    )

    restored = np.fromfile(output_path, dtype=np.uint16).reshape(frames, height, width)
    np.testing.assert_array_equal(restored, denoised)


def test_accepted_zero_residual_checkpoint_keeps_2dnr_raw_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, height, width = 3, 8, 8
    source = np.full((frames, height, width), 16 * 252, dtype=np.uint16)
    denoised = np.full((frames, height, width), 300, dtype=np.uint16)
    denoised[1, 0, 0] = 255
    denoised[2, 0, 1] = 299
    source_path = tmp_path / "source.raw"
    denoised_path = tmp_path / "denoised.raw"
    output_path = tmp_path / "output.raw"
    checkpoint_path = tmp_path / "accepted_identity.pth"
    source.tofile(source_path)
    denoised.tofile(denoised_path)
    model = local_align.LocalAlignResidualUNet(base_ch=4)
    torch.save(
        {
            "architecture": "dual_branch_26ch_v1",
            "config": {"accepted_against_2dnr": True},
            "model_state": model.state_dict(),
            "base_ch": 4,
            "stride": 1,
        },
        checkpoint_path,
    )
    input_frame = np.zeros((26, 4, 4), dtype=np.float32)
    base_frame = np.zeros((4, 4, 4), dtype=np.float32)
    monkeypatch.setattr(
        local_align,
        "_la_input_full",
        lambda sequence, current: (input_frame, base_frame, np.zeros((1, 4, 4), dtype=np.float32)),
    )

    local_align.infer_local_align_unet(
        source=str(source_path),
        denoised=str(denoised_path),
        model_path=str(checkpoint_path),
        output=str(output_path),
        frames=frames,
        height=height,
        width=width,
        stride=1,
        tile=4,
        overlap=0,
        device="cpu",
    )

    restored = np.fromfile(output_path, dtype=np.uint16).reshape(frames, height, width)
    np.testing.assert_array_equal(restored, denoised)


def test_comparison_renderer_creates_video_and_manifest(tmp_path: Path) -> None:
    frames, height, width = 2, 64, 64
    stream = np.full((frames, height, width), 800, dtype=np.uint16)
    paths = [tmp_path / name for name in ("two_dnr.raw", "three_dnr.raw", "ai.raw")]
    for path in paths:
        stream.tofile(path)
    metadata_source = tmp_path / (
        "raw_stream_1920x1080_16bit@RG_"
        "[Shutter=79999,SenserG=131072,IspG=5167,R=2120,G=1024,B=1956].raw"
    )
    output = tmp_path / "comparison.mp4"
    output.write_bytes(b"stale MP4")
    output.with_suffix(".mp4.json").write_text("stale manifest", encoding="utf-8")

    manifest = local_align_render.render_comparison_video(
        two_dnr=paths[0],
        three_dnr=paths[1],
        ai_output=paths[2],
        metadata_source=metadata_source,
        output=output,
        width=width,
        height=height,
        fps=1.0,
        scale=1.0,
        overwrite=True,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.with_suffix(".mp4.json").is_file()
    assert manifest["panel_order"] == ["2DNR", "3DNR", "local-align AI"]

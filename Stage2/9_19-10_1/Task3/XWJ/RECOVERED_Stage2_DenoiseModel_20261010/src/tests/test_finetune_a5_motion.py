import json

import numpy as np
import pytest
import torch

from raw_fusion.finetune.a5_motion import (
    DynamicWindowDataset,
    dilate_motion_mask,
    dynamic_charbonnier_loss,
    set_trainable_scope,
    teacher_anchor_loss,
)


def _write_probe(tmp_path, *, allowed=(0, 160), min_flow=1.0):
    source = np.zeros((8, 4, 80, 96), dtype=np.float32)
    target = np.ones_like(source)
    confidence = np.zeros((8, 80, 96), dtype=np.uint8)
    confidence[:, 30:40, 50:60] = 1
    sample = tmp_path / "sample.npz"
    np.savez_compressed(sample, source=source, target=target, confidence=confidence)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "temporal_frames": 8,
        "allowed_frame_range": list(allowed),
        "confidence_definition": {"flow_magnitude_px_gte": min_flow},
        "windows": [{"sample": str(sample)}],
    }))
    return manifest


def test_dynamic_window_dataset_returns_nonempty_64_patch(tmp_path) -> None:
    dataset = DynamicWindowDataset(_write_probe(tmp_path), seed=7)
    source, target, mask = dataset[0]
    assert source.shape == target.shape == (8, 4, 64, 64)
    assert mask.shape == (8, 1, 64, 64)
    assert torch.any(mask > 0)


def test_dynamic_window_dataset_rejects_unapproved_flow_rule(tmp_path) -> None:
    with pytest.raises(ValueError, match="flow >= 1 px"):
        DynamicWindowDataset(_write_probe(tmp_path, min_flow=0.0))


def test_dynamic_charbonnier_is_masked_and_finite() -> None:
    prediction = torch.zeros((1, 8, 4, 8, 8), requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.zeros((1, 8, 1, 8, 8))
    mask[..., 2:4, 2:4] = 1
    loss, metrics = dynamic_charbonnier_loss(prediction, target, mask)
    assert torch.isfinite(loss)
    assert metrics["dynamic_mask_fraction"] > 0
    loss.backward()
    assert prediction.grad is not None


def test_motion_mask_dilation_is_spatial_and_frame_local() -> None:
    mask = torch.zeros((1, 2, 1, 7, 7))
    mask[0, 0, 0, 3, 3] = 1.0
    mask[0, 1, 0, 1, 5] = 1.0
    dilated = dilate_motion_mask(mask, radius=1)
    assert dilated.shape == mask.shape
    assert torch.all(dilated[0, 0, 0, 2:5, 2:5] == 1)
    assert torch.all(dilated[0, 1, 0, 0:3, 4:7] == 1)
    assert dilated[0, 0, 0, 1, 5] == 0


def test_teacher_anchor_excludes_dilated_motion_pixels() -> None:
    prediction = torch.zeros((1, 1, 1, 7, 7), requires_grad=True)
    target = torch.ones_like(prediction)
    motion = torch.zeros((1, 1, 1, 7, 7))
    motion[0, 0, 0, 3, 3] = 1.0
    static = 1.0 - dilate_motion_mask(motion, radius=1)
    loss, metrics = teacher_anchor_loss(prediction, target, static)
    assert torch.isfinite(loss)
    assert metrics["static_mask_fraction"] == pytest.approx(40 / 49)
    assert static[0, 0, 0, 3, 3] == 0
    loss.backward()
    assert prediction.grad is not None


def test_teacher_anchor_empty_static_mask_is_finite() -> None:
    prediction = torch.zeros((1, 1, 1, 4, 4), requires_grad=True)
    target = torch.ones_like(prediction)
    static = torch.zeros_like(prediction)
    loss, metrics = teacher_anchor_loss(prediction, target, static)
    assert loss.item() == 0.0
    assert metrics["static_mask_empty"] == 1.0
    loss.backward()
    assert prediction.grad is not None


def test_mutual_attention_scope_freezes_other_parameters() -> None:
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.hi_qkv_mut = torch.nn.Linear(2, 2)
    model.attn.hi_qkv_self = torch.nn.Linear(2, 2)
    result = set_trainable_scope(model, "mutual_attention")
    assert result["tensors"] == 2
    assert model.attn.hi_qkv_mut.weight.requires_grad
    assert not model.attn.hi_qkv_self.weight.requires_grad


def test_teacher_static_target_mode_is_validated(tmp_path) -> None:
    from raw_fusion.finetune.a5_motion import A5MotionConfig

    config = A5MotionConfig(
        supervision_dir="supervision",
        noise_dir="noise",
        a3_checkpoint="checkpoint",
        dynamic_train_manifest="train",
        dynamic_holdout_manifest="holdout",
        output_dir=str(tmp_path),
        static_target_mode="a3_teacher",
    )
    assert config.static_target_mode == "a3_teacher"
    assert config.static_anchor_weight == 1.0


def test_same_window_teacher_mode_and_radius_are_configurable(tmp_path) -> None:
    from raw_fusion.finetune.a5_motion import A5MotionConfig

    config = A5MotionConfig(
        supervision_dir="supervision",
        noise_dir="noise",
        a3_checkpoint="checkpoint",
        dynamic_train_manifest="train",
        dynamic_holdout_manifest="holdout",
        output_dir=str(tmp_path),
        static_target_mode="same_window_a3_teacher",
        motion_boundary_radius=1,
    )
    assert config.static_target_mode == "same_window_a3_teacher"
    assert config.motion_boundary_radius == 1

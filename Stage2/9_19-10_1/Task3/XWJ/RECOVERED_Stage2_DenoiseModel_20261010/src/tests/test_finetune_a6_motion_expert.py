import numpy as np
import pytest
import torch

from raw_fusion.finetune.a6_motion_expert import (
    FrozenA3MotionModel,
    MotionResidualExpert,
    _balanced_gate_loss,
    dilate_dynamic_mask,
)
from raw_fusion.finetune.noise import generate_noisy_frame


class IdentityBackbone(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def _noise_model() -> dict:
    gains = {"100": 1.0, "200": 2.0}
    return {
        "gain_factors": gains,
        "gain_labels": list(gains),
        "empirical_shot_coefficients": {
            name: {"100": 1.0, "200": 2.0}
            for name in ("R", "Gr", "Gb", "B")
        },
        "read_variance_dn2": {
            name: {"100": 1.0, "200": 4.0}
            for name in ("R", "Gr", "Gb", "B")
        },
        "channel_correlation": {"aggregate": np.eye(4).tolist()},
    }


def test_noise_generator_accepts_packed_patches() -> None:
    clean = np.full((4, 64, 64), 600.0, dtype=np.float32)
    noisy = generate_noisy_frame(
        clean,
        _noise_model(),
        "200",
        {"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0},
        None,
        np.random.default_rng(4),
    )
    assert noisy.shape == clean.shape
    assert noisy.dtype == np.uint16


def test_motion_expert_is_exact_a3_identity_at_initialization() -> None:
    inputs = torch.rand((1, 8, 4, 16, 16))
    model = FrozenA3MotionModel(
        IdentityBackbone(), MotionResidualExpert(channels=8, max_residual=0.25)
    )
    candidate, baseline, gate, residual = model.forward_with_aux(inputs)
    torch.testing.assert_close(candidate, baseline)
    assert torch.count_nonzero(residual) == 0
    assert 0.0 < float(gate.mean()) < 0.2


def test_motion_expert_uses_spatial_only_convolutions() -> None:
    expert = MotionResidualExpert(channels=8, max_residual=0.25)
    convolutions = [
        module for module in expert.modules() if isinstance(module, torch.nn.Conv3d)
    ]
    assert convolutions
    assert all(module.kernel_size[0] == 1 for module in convolutions)


def test_frozen_backbone_stays_eval_and_has_no_gradients() -> None:
    backbone = torch.nn.Conv3d(4, 4, kernel_size=1)

    class ConvBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = backbone

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.conv(inputs.transpose(1, 2)).transpose(1, 2)

    model = FrozenA3MotionModel(
        ConvBackbone(), MotionResidualExpert(channels=8, max_residual=0.25)
    )
    model.train()
    output = model(torch.rand((1, 8, 4, 16, 16)))
    output.mean().backward()
    assert not model.backbone.training
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.expert.parameters())


def test_dynamic_mask_dilation_does_not_mix_frames() -> None:
    mask = torch.zeros((1, 2, 1, 7, 7))
    mask[0, 0, 0, 3, 3] = 1.0
    dilated = dilate_dynamic_mask(mask, radius=1)
    assert torch.all(dilated[0, 0, 0, 2:5, 2:5] == 1.0)
    assert torch.count_nonzero(dilated[0, 1]) == 0


def test_balanced_gate_loss_penalizes_missed_motion() -> None:
    target = torch.zeros((1, 1, 1, 8, 8))
    target[..., 3:5, 3:5] = 1.0
    valid = torch.ones_like(target)
    missed = _balanced_gate_loss(torch.full_like(target, 0.01), target, valid)
    detected = _balanced_gate_loss(
        torch.where(target > 0, torch.full_like(target, 0.99), torch.full_like(target, 0.01)),
        target,
        valid,
    )
    assert detected < missed


def test_invalid_noise_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="packed clean shape"):
        generate_noisy_frame(
            np.zeros((3, 64, 64), dtype=np.float32),
            _noise_model(),
            "100",
            {"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0},
            None,
            np.random.default_rng(4),
        )

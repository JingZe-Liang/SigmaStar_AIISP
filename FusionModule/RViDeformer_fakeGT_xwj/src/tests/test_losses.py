from __future__ import annotations

import pytest
import torch

from raw_fusion.config import LossConfig
from raw_fusion.losses import FusionLoss, valid_target_mask
from raw_fusion.model import FusionOutput


def make_output(
    *,
    prediction: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
) -> FusionOutput:
    if prediction is None:
        prediction = torch.full((1, 4, 2, 2), 0.4)
    if gate is None:
        gate = torch.full((prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]), 0.5)
    if correction is None:
        correction = torch.zeros_like(prediction)
    base = prediction - correction
    return FusionOutput(prediction=prediction, base=base, gate=gate, correction=correction)


def make_loss(**overrides: float | int) -> FusionLoss:
    values: dict[str, float | int] = {
        "gradient_weight": 0.05,
        "gate_weight": 0.02,
        "residual_weight": 0.01,
        "range_weight": 0.01,
        "charbonnier_epsilon": 0.001,
        "gate_temperature": 0.02,
        "gate_margin": 0.005,
        "saturation_margin_dn": 4,
    }
    values.update(overrides)
    return FusionLoss(LossConfig(**values), white_level=4095, target_black_level=252)


def test_valid_target_mask_excludes_saturated_threshold() -> None:
    target = torch.tensor([0.0, 0.999, 1.0])
    mask = valid_target_mask(target, threshold=0.999)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(mask, torch.tensor([True, False, False]))


def test_saturated_values_do_not_change_reconstruction_loss() -> None:
    target = torch.tensor([[[[0.2, 1.0]]]]).repeat(1, 4, 1, 1)
    output_a = make_output(prediction=target.clone())
    changed = target.clone()
    changed[..., 1] = -10.0
    output_b = make_output(prediction=changed)
    loss_fn = make_loss(saturation_margin_dn=4)
    first = loss_fn(output_a, target, target, target).reconstruction
    second = loss_fn(output_b, target, target, target).reconstruction
    torch.testing.assert_close(first, second)


def test_gate_target_prefers_lower_error_candidate() -> None:
    target = torch.full((1, 4, 2, 2), 0.4)
    denoised = torch.full_like(target, 0.41)
    fused = torch.full_like(target, 0.8)
    output = make_output(gate=torch.full((1, 1, 2, 2), 0.1))
    breakdown = make_loss().forward(output, denoised, fused, target)
    assert breakdown.gate > 0


def test_all_loss_terms_and_total_are_scalar() -> None:
    target = torch.full((1, 4, 4, 4), 0.4)
    output = make_output(
        prediction=torch.full_like(target, 0.7),
        gate=torch.full((1, 1, 4, 4), 0.25),
        correction=torch.full_like(target, 0.03),
    )
    breakdown = make_loss().forward(output, target, target, target)
    for value in (
        breakdown.total,
        breakdown.reconstruction,
        breakdown.gradient,
        breakdown.gate,
        breakdown.residual,
        breakdown.range,
    ):
        assert value.ndim == 0
        assert torch.isfinite(value)
    expected = (
        breakdown.reconstruction
        + 0.05 * breakdown.gradient
        + 0.02 * breakdown.gate
        + 0.01 * breakdown.residual
        + 0.01 * breakdown.range
    )
    torch.testing.assert_close(breakdown.total, expected)


def test_empty_auxiliary_masks_are_device_zero() -> None:
    target = torch.full((1, 4, 2, 2), 0.4)
    output = make_output(prediction=torch.zeros_like(target))
    breakdown = make_loss().forward(output, target, target, target)
    for value in (breakdown.gradient, breakdown.gate):
        assert value.ndim == 0
        assert value.device == target.device
        assert value.item() == 0.0


def test_empty_reconstruction_mask_raises_value_error() -> None:
    target = torch.ones(1, 4, 2, 2)
    with pytest.raises(ValueError, match="reconstruction.*valid"):
        make_loss().forward(make_output(prediction=torch.zeros_like(target)), target, target, target)


def test_loss_rejects_auxiliary_output_dtype_mismatch_at_entry() -> None:
    target = torch.full((1, 4, 2, 2), 0.4)
    output = make_output(prediction=target)
    invalid = FusionOutput(
        prediction=output.prediction,
        base=output.base,
        gate=output.gate,
        correction=output.correction.double(),
    )
    with pytest.raises(ValueError, match="correction.*dtype"):
        make_loss().forward(invalid, target, target, target)


def test_loss_rejects_invalid_base_shape_at_entry() -> None:
    target = torch.full((1, 4, 2, 2), 0.4)
    output = make_output(prediction=target)
    invalid = FusionOutput(
        prediction=output.prediction,
        base=output.base[..., :1],
        gate=output.gate,
        correction=output.correction,
    )
    with pytest.raises(ValueError, match="base.*shape"):
        make_loss().forward(invalid, target, target, target)

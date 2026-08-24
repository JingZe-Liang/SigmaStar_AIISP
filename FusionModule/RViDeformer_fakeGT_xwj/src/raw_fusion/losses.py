"""Distillation losses for packed RAW fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from raw_fusion.config import LossConfig
from raw_fusion.model import FusionOutput


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: torch.Tensor
    reconstruction: torch.Tensor
    gradient: torch.Tensor
    gate: torch.Tensor
    residual: torch.Tensor
    range: torch.Tensor


def valid_target_mask(target: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    """Return the strict below-saturation supervision mask."""
    return target < threshold


class FusionLoss:
    """Compute reconstruction and candidate-selection auxiliary losses."""

    def __init__(
        self,
        config: LossConfig,
        white_level: float = 4095.0,
        target_black_level: float = 252.0,
    ) -> None:
        self.config = config
        self.white_level = float(white_level)
        self.target_black_level = float(target_black_level)
        denominator = self.white_level - self.target_black_level
        if denominator <= 0:
            raise ValueError("white_level must be greater than target_black_level")
        self.saturation_threshold = (
            self.white_level - config.saturation_margin_dn - self.target_black_level
        ) / denominator

    def __call__(
        self,
        output: FusionOutput,
        denoised: torch.Tensor,
        fused: torch.Tensor,
        target: torch.Tensor,
    ) -> LossBreakdown:
        return self.forward(output, denoised, fused, target)

    def forward(
        self,
        output: FusionOutput,
        denoised: torch.Tensor,
        fused: torch.Tensor,
        target: torch.Tensor,
    ) -> LossBreakdown:
        self._validate(output, denoised, fused, target)
        valid = valid_target_mask(target, self.saturation_threshold)
        if not bool(valid.any()):
            raise ValueError("reconstruction valid mask is empty")

        residual_error = output.prediction - target
        epsilon = float(self.config.charbonnier_epsilon)
        reconstruction = torch.sqrt(residual_error.square() + epsilon * epsilon)[valid].mean()
        gradient = self._gradient_loss(output.prediction, target, valid)
        gate = self._gate_loss(output.gate, denoised, fused, target, valid)
        residual = output.correction.abs().mean()
        range_penalty = (F.relu(-output.prediction) + F.relu(output.prediction - 1.0)).mean()
        total = (
            reconstruction
            + self.config.gradient_weight * gradient
            + self.config.gate_weight * gate
            + self.config.residual_weight * residual
            + self.config.range_weight * range_penalty
        )
        return LossBreakdown(
            total=total,
            reconstruction=reconstruction,
            gradient=gradient,
            gate=gate,
            residual=residual,
            range=range_penalty,
        )

    @staticmethod
    def _gradient_loss(
        prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
        vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
        terms: list[torch.Tensor] = []
        if bool(horizontal_valid.any()):
            prediction_gradient = prediction[..., :, 1:] - prediction[..., :, :-1]
            target_gradient = target[..., :, 1:] - target[..., :, :-1]
            terms.append((prediction_gradient - target_gradient).abs()[horizontal_valid])
        if bool(vertical_valid.any()):
            prediction_gradient = prediction[..., 1:, :] - prediction[..., :-1, :]
            target_gradient = target[..., 1:, :] - target[..., :-1, :]
            terms.append((prediction_gradient - target_gradient).abs()[vertical_valid])
        if not terms:
            return prediction.new_zeros(())
        return torch.cat(terms).mean()

    def _gate_loss(
        self,
        gate: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        denoised_error = (denoised - target).abs().mean(dim=1, keepdim=True)
        fused_error = (fused - target).abs().mean(dim=1, keepdim=True)
        error_difference = fused_error - denoised_error
        gate_mask = valid.all(dim=1, keepdim=True) & (
            error_difference.abs() > self.config.gate_margin
        )
        if not bool(gate_mask.any()):
            return gate.new_zeros(())
        gate_target = torch.sigmoid(error_difference / self.config.gate_temperature)
        # Disable autocast for binary_cross_entropy as it's not AMP-safe
        with torch.amp.autocast(enabled=False, device_type=gate.device.type):
            return F.binary_cross_entropy(
                gate[gate_mask].float(), gate_target[gate_mask].float()
            )

    @staticmethod
    def _validate(
        output: FusionOutput,
        denoised: torch.Tensor,
        fused: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if not isinstance(output, FusionOutput):
            raise TypeError("output must be a FusionOutput")
        reference = output.prediction
        if not isinstance(reference, torch.Tensor):
            raise TypeError("prediction must be a torch.Tensor")
        if reference.ndim != 4 or reference.shape[1] != 4:
            raise ValueError("prediction must have shape [B, 4, H, W]")
        if not reference.is_floating_point():
            raise TypeError("prediction must be floating-point")
        expected_gate_shape = (
            reference.shape[0],
            1,
            reference.shape[2],
            reference.shape[3],
        )
        for name, value, expected_shape in (
            ("base", output.base, reference.shape),
            ("gate", output.gate, expected_gate_shape),
            ("correction", output.correction, reference.shape),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.shape != expected_shape:
                raise ValueError(f"{name} has an invalid shape")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating-point")
            if value.device != reference.device:
                raise ValueError(f"{name} must be on the prediction device")
            if value.dtype != reference.dtype:
                raise ValueError(f"{name} must have the prediction dtype")
        for name, value in (
            ("denoised", denoised),
            ("fused", fused),
            ("target", target),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.shape != reference.shape:
                raise ValueError(f"{name} must have the same shape as prediction")
            if value.device != reference.device:
                raise ValueError(f"{name} must be on the same device as prediction")
            if value.dtype != reference.dtype:
                raise ValueError(f"{name} must have the same dtype as prediction")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating-point")

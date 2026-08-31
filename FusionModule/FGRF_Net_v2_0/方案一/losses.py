"""Three-term training objective for FGRF-Net v2.0."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import fuse


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _charbonnier(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


def _highpass(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    lowpass = F.avg_pool2d(value, kernel_size, stride=1, padding=kernel_size // 2)
    return value - lowpass


@dataclass(frozen=True)
class LossWeights:
    gate: float = 1.0
    texture: float = 0.35
    motion: float = 0.25


class TextureFusionLoss(torch.nn.Module):
    """Static gate + texture supervision and motion-only D2 fallback supervision."""

    def __init__(
        self,
        weights: LossWeights,
        oracle_kernel: int = 5,
        texture_kernels: tuple[int, ...] = (3, 7),
        texture_threshold: float = 0.003,
        candidate_min_norm: float = 0.001,
        candidate_range: float = 0.01,
    ) -> None:
        super().__init__()
        if oracle_kernel <= 0 or oracle_kernel % 2 == 0:
            raise ValueError("oracle_kernel must be a positive odd integer")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in texture_kernels):
            raise ValueError("texture kernels must be positive odd integers")
        self.weights = weights
        self.oracle_kernel = oracle_kernel
        self.texture_kernels = texture_kernels
        self.texture_threshold = texture_threshold
        self.candidate_min_norm = candidate_min_norm
        self.candidate_range = candidate_range

    def forward(self, alpha: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = batch["base"]
        temporal = batch["temporal"]
        proxy = batch["proxy"]
        static = batch["static_mask"]
        motion = batch["motion_mask"]
        delta = temporal - base
        output = fuse(base, temporal, alpha)

        candidate_norm = torch.sqrt(delta.square().mean(dim=1, keepdim=True) + 1e-8)
        candidate_confidence = ((candidate_norm - self.candidate_min_norm) / self.candidate_range).clamp(0.0, 1.0)

        numerator = ((proxy - base) * delta).sum(dim=1, keepdim=True)
        denominator = delta.square().sum(dim=1, keepdim=True)
        padding = self.oracle_kernel // 2
        numerator = F.avg_pool2d(numerator, self.oracle_kernel, stride=1, padding=padding)
        denominator = F.avg_pool2d(denominator, self.oracle_kernel, stride=1, padding=padding)
        oracle_alpha = (numerator / denominator.clamp_min(1e-8)).clamp(0.0, 1.0).detach()

        gate_weight = static * candidate_confidence
        gate_loss = _masked_mean(F.smooth_l1_loss(alpha, oracle_alpha, reduction="none"), gate_weight)

        texture_loss = torch.zeros((), dtype=alpha.dtype, device=alpha.device)
        texture_support = torch.zeros_like(static)
        for kernel in self.texture_kernels:
            proxy_high = _highpass(proxy, kernel)
            output_high = _highpass(output, kernel)
            support = (proxy_high.abs().mean(dim=1, keepdim=True) / self.texture_threshold).clamp(0.0, 1.0)
            texture_support = torch.maximum(texture_support, support)
            texture_error = _charbonnier(output_high - proxy_high).mean(dim=1, keepdim=True)
            texture_loss = texture_loss + _masked_mean(texture_error, static * candidate_confidence * support)
        texture_loss = texture_loss / len(self.texture_kernels)

        motion_weight = motion * candidate_confidence
        motion_loss = _masked_mean(F.smooth_l1_loss(alpha, torch.zeros_like(alpha), reduction="none"), motion_weight)

        total = (
            self.weights.gate * gate_loss
            + self.weights.texture * texture_loss
            + self.weights.motion * motion_loss
        )
        with torch.no_grad():
            metrics = {
                "total": total.detach(),
                "gate": gate_loss.detach(),
                "texture": texture_loss.detach(),
                "motion": motion_loss.detach(),
                "alpha": alpha.mean().detach(),
                "alpha_static": _masked_mean(alpha, static).detach(),
                "alpha_motion": _masked_mean(alpha, motion).detach(),
                "oracle_static": _masked_mean(oracle_alpha, static).detach(),
                "static_fraction": static.mean().detach(),
                "motion_fraction": motion.mean().detach(),
                "texture_fraction": _masked_mean(texture_support, static).detach(),
            }
        return total, metrics

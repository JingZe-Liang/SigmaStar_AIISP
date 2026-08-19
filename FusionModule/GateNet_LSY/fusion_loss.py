from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / torch.clamp(weights.sum(), min=1.0)


@dataclass(frozen=True)
class LossWeights:
    static_gate: float = 1.0
    motion_gate: float = 0.5
    proxy_reconstruction: float = 0.2
    smoothness: float = 0.01


class WeakFusionLoss(nn.Module):
    def __init__(
        self,
        *,
        oracle_window: int = 9,
        static_temporal_threshold: float = 1.5,
        static_range_threshold: float = 3.0,
        motion_temporal_threshold: float = 1.0,
        weights: LossWeights = LossWeights(),
    ):
        super().__init__()
        if oracle_window <= 0 or oracle_window % 2 == 0:
            raise ValueError("oracle_window must be a positive odd integer")
        self.oracle_window = oracle_window
        self.static_temporal_threshold = static_temporal_threshold
        self.static_range_threshold = static_range_threshold
        self.motion_temporal_threshold = motion_temporal_threshold
        self.weights = weights

    def forward(
        self, alpha: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        denoised = batch["denoised"]
        fused = batch["fused"]
        proxy = batch["proxy"]
        source = batch["source"]
        source_prev = batch["source_prev"]
        source_next = batch["source_next"]
        motion = batch["motion"]
        valid_signal = batch["valid_signal"]
        sigma = torch.clamp(batch["noise_sigma"], min=1.0)
        sigma_mean = sigma.mean(dim=1, keepdim=True)

        delta = fused - denoised
        output = denoised + alpha * delta
        temporal_score = torch.maximum(
            torch.abs(source - source_prev), torch.abs(source - source_next)
        ).mean(dim=1, keepdim=True) / (math.sqrt(2.0) * sigma_mean)
        range_score = batch["temporal_range"] / sigma_mean

        static_mask = (
            (motion < 0.5)
            & (temporal_score < self.static_temporal_threshold)
            & (range_score < self.static_range_threshold)
            & (valid_signal > 0.5)
        ).float()
        motion_mask = (
            (motion > 0.5)
            & (temporal_score > self.motion_temporal_threshold)
            & (valid_signal > 0.5)
        ).float()

        candidate_distance = torch.sqrt(delta.square().mean(dim=1, keepdim=True) + 1e-6)
        candidate_distance = candidate_distance / sigma_mean
        candidate_confidence = torch.clamp((candidate_distance - 0.1) / 1.9, 0.0, 1.0)

        numerator = ((proxy - denoised) * delta).sum(dim=1, keepdim=True)
        denominator = delta.square().sum(dim=1, keepdim=True)
        padding = self.oracle_window // 2
        numerator = F.avg_pool2d(
            numerator, self.oracle_window, stride=1, padding=padding
        )
        denominator = F.avg_pool2d(
            denominator, self.oracle_window, stride=1, padding=padding
        )
        oracle_alpha = torch.clamp(numerator / (denominator + 1e-6), 0.0, 1.0).detach()

        static_weight = static_mask * candidate_confidence
        motion_weight = motion_mask * candidate_confidence
        static_gate_loss = _masked_mean(
            F.smooth_l1_loss(alpha, oracle_alpha, reduction="none"), static_weight
        )
        motion_gate_loss = _masked_mean(
            F.smooth_l1_loss(alpha, torch.zeros_like(alpha), reduction="none"),
            motion_weight,
        )

        normalized_error = (output - proxy) / sigma
        charbonnier = torch.sqrt(normalized_error.square() + 1e-4).mean(
            dim=1, keepdim=True
        )
        proxy_loss = _masked_mean(charbonnier, static_mask)

        alpha_dx = torch.abs(alpha[..., 1:] - alpha[..., :-1])
        alpha_dy = torch.abs(alpha[..., 1:, :] - alpha[..., :-1, :])
        source_dx = torch.abs(source[..., 1:] - source[..., :-1]).mean(
            dim=1, keepdim=True
        )
        source_dy = torch.abs(source[..., 1:, :] - source[..., :-1, :]).mean(
            dim=1, keepdim=True
        )
        edge_weight_x = torch.exp(-source_dx / (2.0 * sigma_mean))
        edge_weight_y = torch.exp(-source_dy / (2.0 * sigma_mean))
        smoothness_loss = 0.5 * (
            (alpha_dx * edge_weight_x).mean() + (alpha_dy * edge_weight_y).mean()
        )

        total = (
            self.weights.static_gate * static_gate_loss
            + self.weights.motion_gate * motion_gate_loss
            + self.weights.proxy_reconstruction * proxy_loss
            + self.weights.smoothness * smoothness_loss
        )

        with torch.no_grad():
            d2_proxy = torch.sqrt(((denoised - proxy) / sigma).square() + 1e-4).mean(
                dim=1, keepdim=True
            )
            d3_proxy = torch.sqrt(((fused - proxy) / sigma).square() + 1e-4).mean(
                dim=1, keepdim=True
            )
            metrics = {
                "total": total.detach(),
                "static_gate": static_gate_loss.detach(),
                "motion_gate": motion_gate_loss.detach(),
                "proxy": proxy_loss.detach(),
                "smooth": smoothness_loss.detach(),
                "alpha_mean": alpha.mean().detach(),
                "alpha_static": _masked_mean(alpha, static_mask).detach(),
                "alpha_motion": _masked_mean(alpha, motion_mask).detach(),
                "oracle_static": _masked_mean(oracle_alpha, static_mask).detach(),
                "static_fraction": static_mask.mean().detach(),
                "motion_fraction": motion_mask.mean().detach(),
                "output_proxy": _masked_mean(charbonnier, static_mask).detach(),
                "d2_proxy": _masked_mean(d2_proxy, static_mask).detach(),
                "d3_proxy": _masked_mean(d3_proxy, static_mask).detach(),
            }
        return total, metrics

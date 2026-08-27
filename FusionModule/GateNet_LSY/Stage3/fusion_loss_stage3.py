from __future__ import annotations

import math

import torch
import torch.nn as nn

from fusion_loss_stage2 import Stage2FusionLoss


class Stage3FusionLoss(nn.Module):
    """Stage2 loss plus an explicit static-region 2DNR preference."""

    def __init__(self, base_criterion: Stage2FusionLoss, *, static_alpha_weight=0.3, static_d2_weight=0.3):
        super().__init__()
        if static_alpha_weight < 0 or static_d2_weight < 0:
            raise ValueError("static loss weights must be non-negative")
        self.base_criterion = base_criterion
        self.static_alpha_weight = static_alpha_weight
        self.static_d2_weight = static_d2_weight

    def forward(self, alpha, motion_logit, batch):
        total, metrics = self.base_criterion(alpha, motion_logit, batch)
        sigma = torch.clamp(batch["noise_sigma"], min=1.0)
        sigma_mean = sigma.mean(dim=1, keepdim=True)
        temporal_score = torch.maximum(
            torch.abs(batch["source"] - batch["source_prev"]),
            torch.abs(batch["source"] - batch["source_next"]),
        ).mean(dim=1, keepdim=True) / (math.sqrt(2.0) * sigma_mean)
        range_score = batch["temporal_range"] / sigma_mean
        static_mask = (
            (batch["motion"] < 0.5)
            & (temporal_score < 1.5)
            & (range_score < 3.0)
            & (batch["valid_signal"] > 0.5)
        ).float()
        valid = torch.clamp(static_mask.sum(), min=1.0)
        static_alpha = (alpha * static_mask).sum() / valid
        output_delta = alpha * (batch["fused"] - batch["denoised"])
        static_d2 = (
            torch.sqrt((output_delta / sigma).square() + 1e-4).mean(dim=1, keepdim=True)
            * static_mask
        ).sum() / valid
        total = total + self.static_alpha_weight * static_alpha + self.static_d2_weight * static_d2
        metrics = dict(metrics)
        metrics["static_alpha_penalty"] = static_alpha.detach()
        metrics["static_d2_penalty"] = static_d2.detach()
        metrics["total"] = total.detach()
        return total, metrics

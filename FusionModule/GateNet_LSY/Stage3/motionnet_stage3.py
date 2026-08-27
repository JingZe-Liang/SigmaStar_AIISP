from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MOTION_FEATURE_CHANNELS = 10


def _gradient(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(torch.abs(x[..., 1:] - x[..., :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(x[..., 1:, :] - x[..., :-1, :]), (0, 0, 0, 1))
    return dx + dy


def build_motion_features(
    source: torch.Tensor,
    source_prev: torch.Tensor,
    source_next: torch.Tensor,
    noise_sigma: torch.Tensor,
) -> torch.Tensor:
    """Temporal-only features; deliberately independent of the fusion backbone."""
    sigma = torch.clamp(noise_sigma, min=1.0)
    sigma_mean = sigma.mean(dim=1, keepdim=True)
    scale = math.sqrt(2.0) * sigma
    prev_delta = torch.clamp((source - source_prev) / scale, -8.0, 8.0) / 8.0
    next_delta = torch.clamp((source_next - source) / scale, -8.0, 8.0) / 8.0
    temporal = torch.maximum(prev_delta.abs(), next_delta.abs()).mean(dim=1, keepdim=True)
    gradient = _gradient(source).mean(dim=1, keepdim=True) / sigma_mean
    return torch.cat(
        [prev_delta, next_delta, torch.clamp(temporal, 0.0, 1.0), torch.clamp(gradient, 0.0, 8.0) / 8.0],
        dim=1,
    )


class TemporalMotionNet(nn.Module):
    def __init__(self, base_channels: int = 12):
        super().__init__()
        self.base_channels = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(MOTION_FEATURE_CHANNELS, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def model_config(self) -> dict[str, int]:
        return {"base_channels": self.base_channels}


class MotionFocalLoss(nn.Module):
    def __init__(self, *, positive_weight: float = 4.0, gamma: float = 2.0):
        super().__init__()
        self.positive_weight = positive_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = batch["motion"]
        valid = batch["valid_signal"]
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=torch.tensor(self.positive_weight, device=logits.device)
        )
        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        loss = (bce * (1.0 - p_t).pow(self.gamma) * valid).sum() / torch.clamp(valid.sum(), min=1.0)
        with torch.no_grad():
            prediction = probability >= 0.7
            truth = target >= 0.5
            valid_bool = valid >= 0.5
            tp = (prediction & truth & valid_bool).sum().float()
            fp = (prediction & ~truth & valid_bool).sum().float()
            fn = (~prediction & truth & valid_bool).sum().float()
            precision = tp / torch.clamp(tp + fp, min=1.0)
            recall = tp / torch.clamp(tp + fn, min=1.0)
        return loss, {
            "motion_loss": loss.detach(),
            "motion_precision": precision.detach(),
            "motion_recall": recall.detach(),
            "predicted_motion_fraction": probability.mean().detach(),
        }

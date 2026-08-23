from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


QUALITY_CHANNELS = 13
MD_CHANNELS = 2
FEATURE_CHANNELS = QUALITY_CHANNELS + MD_CHANNELS


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(torch.abs(x[..., 1:] - x[..., :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(x[..., 1:, :] - x[..., :-1, :]), (0, 0, 0, 1))
    return dx + dy


def _local_std(x: torch.Tensor) -> torch.Tensor:
    mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mean_square = F.avg_pool2d(x.square(), kernel_size=3, stride=1, padding=1)
    return torch.sqrt(torch.clamp(mean_square - mean.square(), min=0.0) + 1e-6)


def build_gate_features(
    denoised: torch.Tensor,
    fused: torch.Tensor,
    source: torch.Tensor,
    source_prev: torch.Tensor,
    source_next: torch.Tensor,
    motion: torch.Tensor,
    noise_sigma: torch.Tensor,
) -> torch.Tensor:
    """Build bounded, noise-normalized features in the packed RAW domain."""
    sigma = torch.clamp(noise_sigma, min=1.0)
    sigma_mean = sigma.mean(dim=1, keepdim=True)
    delta = (fused - denoised) / sigma
    signed_delta = torch.clamp(delta, -8.0, 8.0) / 8.0
    absolute_delta = torch.clamp(torch.abs(delta), 0.0, 8.0) / 8.0

    denoised_std = torch.clamp(
        _local_std(denoised).mean(dim=1, keepdim=True) / sigma_mean,
        0.0,
        8.0,
    ) / 8.0
    fused_std = torch.clamp(
        _local_std(fused).mean(dim=1, keepdim=True) / sigma_mean,
        0.0,
        8.0,
    ) / 8.0
    temporal = torch.maximum(
        torch.abs(source - source_prev), torch.abs(source - source_next)
    )
    temporal = torch.clamp(
        temporal.mean(dim=1, keepdim=True) / (math.sqrt(2.0) * sigma_mean),
        0.0,
        8.0,
    ) / 8.0
    brightness = torch.clamp(
        denoised.mean(dim=1, keepdim=True) / (4095.0 - 300.0), 0.0, 1.0
    )
    gradient = torch.clamp(
        _gradient_magnitude(source).mean(dim=1, keepdim=True) / sigma_mean,
        0.0,
        8.0,
    ) / 8.0
    motion = torch.clamp(motion, 0.0, 1.0)
    motion_min = -F.max_pool2d(-motion, kernel_size=3, stride=1, padding=1)
    motion_boundary = F.max_pool2d(
        motion, kernel_size=3, stride=1, padding=1
    ) - motion_min
    return torch.cat(
        [
            signed_delta,
            absolute_delta,
            denoised_std,
            fused_std,
            temporal,
            brightness,
            gradient,
            motion,
            motion_boundary,
        ],
        dim=1,
    )


class GateNet(nn.Module):
    """Small convex-fusion gate with a capped MD contribution."""

    def __init__(self):
        super().__init__()
        self.quality = nn.Sequential(
            nn.Conv2d(QUALITY_CHANNELS, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.md = nn.Sequential(
            nn.Conv2d(MD_CHANNELS, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)
        nn.init.zeros_(self.md[-1].weight)
        nn.init.zeros_(self.md[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != FEATURE_CHANNELS:
            raise ValueError(
                f"Expected {FEATURE_CHANNELS} features, got {features.shape[1]}"
            )
        quality_logit = self.quality(features[:, :QUALITY_CHANNELS])
        md_logit = self.md(features[:, QUALITY_CHANNELS:])
        return torch.sigmoid(quality_logit + 0.5 * torch.tanh(md_logit))

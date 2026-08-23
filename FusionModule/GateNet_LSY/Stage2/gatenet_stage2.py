from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_CHANNELS = 9


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
    noise_sigma: torch.Tensor,
) -> torch.Tensor:
    """Build compact D2/D3/noisy features without an MD input."""
    sigma = torch.clamp(noise_sigma, min=1.0)
    sigma_mean = sigma.mean(dim=1, keepdim=True)

    normalized_delta = (fused - denoised) / sigma
    signed_delta = torch.clamp(normalized_delta, -8.0, 8.0) / 8.0
    absolute_delta = torch.clamp(
        normalized_delta.abs().mean(dim=1, keepdim=True), 0.0, 8.0
    ) / 8.0

    std_difference = (
        _local_std(fused).mean(dim=1, keepdim=True)
        - _local_std(denoised).mean(dim=1, keepdim=True)
    ) / sigma_mean
    std_difference = torch.clamp(std_difference, -8.0, 8.0) / 8.0

    temporal = torch.maximum(
        torch.abs(source - source_prev), torch.abs(source - source_next)
    ).mean(dim=1, keepdim=True)
    temporal = torch.clamp(
        temporal / (math.sqrt(2.0) * sigma_mean), 0.0, 8.0
    ) / 8.0

    brightness = torch.clamp(
        denoised.mean(dim=1, keepdim=True) / (4095.0 - 300.0), 0.0, 1.0
    )
    gradient = torch.clamp(
        _gradient_magnitude(source).mean(dim=1, keepdim=True) / sigma_mean,
        0.0,
        8.0,
    ) / 8.0

    return torch.cat(
        [
            signed_delta,
            absolute_delta,
            std_difference,
            temporal,
            brightness,
            gradient,
        ],
        dim=1,
    )


class DepthwisePointwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GateNetStage2(nn.Module):
    """Compact fusion gate with an auxiliary, internally predicted motion map."""

    def __init__(self, base_channels: int = 12):
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        expanded_channels = base_channels + 4
        self.base_channels = base_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            DepthwisePointwiseBlock(
                base_channels, expanded_channels, dilation=2
            ),
            DepthwisePointwiseBlock(expanded_channels, base_channels),
        )
        self.fusion_head = nn.Conv2d(base_channels, 1, 1)
        self.motion_head = nn.Conv2d(base_channels, 1, 1)
        nn.init.zeros_(self.fusion_head.weight)
        nn.init.zeros_(self.fusion_head.bias)
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)

    def forward(
        self, features: torch.Tensor, *, return_motion: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if features.shape[1] != FEATURE_CHANNELS:
            raise ValueError(
                f"Expected {FEATURE_CHANNELS} features, got {features.shape[1]}"
            )
        shared = self.backbone(features)
        alpha = torch.sigmoid(self.fusion_head(shared))
        if not return_motion:
            return alpha
        return alpha, self.motion_head(shared)

    def model_config(self) -> dict[str, int]:
        return {"base_channels": self.base_channels}

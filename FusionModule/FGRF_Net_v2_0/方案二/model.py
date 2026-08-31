"""FGRF-Net v2.0 lightweight variant: separable 3x3 blocks."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return next((groups for groups in (8, 4, 2, 1) if channels % groups == 0), 1)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride=stride, padding=1, groups=input_channels, bias=False),
            nn.Conv2d(input_channels, output_channels, 1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False),
            nn.Conv2d(output_channels, output_channels, 1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TextureGateNet(nn.Module):
    """Predict alpha from only current noisy RAW, 2DNR, and 3DNR tensors."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(12, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.decoder2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 3, base_channels)
        self.alpha_head = nn.Conv2d(base_channels, 1, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)

    def forward(self, noisy: torch.Tensor, base: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        x = torch.cat((noisy, base, temporal), dim=1)
        first = self.encoder1(x)
        second = self.encoder2(first)
        third = self.encoder3(second)
        decoded_second = self.decoder2(
            torch.cat((F.interpolate(third, size=second.shape[-2:], mode="bilinear", align_corners=False), second), dim=1)
        )
        decoded_first = self.decoder1(
            torch.cat((F.interpolate(decoded_second, size=first.shape[-2:], mode="bilinear", align_corners=False), first), dim=1)
        )
        return torch.sigmoid(self.alpha_head(decoded_first))


def fuse(base: torch.Tensor, temporal: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Convex 2DNR/3DNR fusion; alpha is unconstrained only by its sigmoid head."""
    return base + alpha * (temporal - base)

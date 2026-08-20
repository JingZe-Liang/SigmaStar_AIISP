from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 6, 4, 3, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise(self.norm(inputs))
        left, right = self.expand(features).chunk(2, dim=1)
        features = self.project(F.silu(left) * right)
        return inputs + self.scale * features


class SafeGateUNet(nn.Module):
    """Small causal candidate-aware network that predicts one shared CFA gate."""

    def __init__(self, input_channels: int = 24, width: int = 24):
        super().__init__()
        self.input_channels = input_channels
        self.width = width

        self.stem = ConvNormAct(input_channels, width)
        self.enc0 = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.down1 = ConvNormAct(width, width * 2, stride=2)
        self.enc1 = nn.Sequential(ResidualBlock(width * 2), ResidualBlock(width * 2))
        self.down2 = ConvNormAct(width * 2, width * 3, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(width * 3), ResidualBlock(width * 3), ResidualBlock(width * 3)
        )
        self.up1 = ConvNormAct(width * 3 + width * 2, width * 2)
        self.dec1 = ResidualBlock(width * 2)
        self.up0 = ConvNormAct(width * 2 + width, width)
        self.dec0 = ResidualBlock(width)
        self.head = nn.Conv2d(width, 1, 3, padding=1)

        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected Bx{self.input_channels}xHxW input, got {inputs.shape}"
            )
        skip0 = self.enc0(self.stem(inputs))
        skip1 = self.enc1(self.down1(skip0))
        features = self.bottleneck(self.down2(skip1))
        features = F.interpolate(
            features, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.dec1(self.up1(torch.cat((features, skip1), dim=1)))
        features = F.interpolate(
            features, size=skip0.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.dec0(self.up0(torch.cat((features, skip0), dim=1)))
        return self.head(features)

    def gate(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(inputs))


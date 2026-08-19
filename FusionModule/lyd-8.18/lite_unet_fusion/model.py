from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(value))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, value), dim=1))


class LiteFusionUNet(nn.Module):
    """Predicts four Bayer-plane 3DNR correction confidences in [0, 1].

    Inputs are 14 channels: 2DNR (4), 3DNR (4), source RAW (4), motion (1),
    and flatness (1). The network predicts a confidence, never a replacement
    RAW image; the caller applies the constrained 2DNR-prior fusion formula.
    """

    def __init__(self, in_channels: int = 14, base_channels: int = 24) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, 64, 80
        self.config = {"in_channels": in_channels, "base_channels": base_channels}
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.bridge = DownBlock(c3, c4)
        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.enc1(value)
        second = self.enc2(first)
        third = self.enc3(second)
        bridge = self.bridge(third)
        value = self.dec3(bridge, third)
        value = self.dec2(value, second)
        value = self.dec1(value, first)
        return torch.sigmoid(self.head(value))

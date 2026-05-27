from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, groups: int = 8) -> nn.GroupNorm:
    g = min(groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            _gn(channels, groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            _gn(channels, groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block1 = ResidualBlock(out_ch, groups)
        self.block2 = ResidualBlock(out_ch, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(self.proj(x)))


class SEGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class MotionAwareFusionUNet(nn.Module):
    """
    Lightweight controller for RAW 2DNR/3DNR adaptive fusion.

    Input channels, strong mode:
      noisy_prev, noisy_curr, |curr-prev|, 2dnr, 3dnr, |2dnr-3dnr|, edge(curr)
    Output:
      alpha_3d in [0,1]. alpha_3d=1 means prefer 3DNR; alpha_3d=0 means prefer 2DNR.
    """

    def __init__(self, in_ch: int = 7, base: int = 24, groups: int = 8, init_alpha3d: float = 0.80):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base = int(base)
        self.enc1 = ConvStem(in_ch, base, groups)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvStem(base * 2, base * 2, groups)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(
            ConvStem(base * 4, base * 4, groups),
            SEGate(base * 4),
            ResidualBlock(base * 4, groups),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvStem(base * 4, base * 2, groups)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvStem(base * 2, base, groups)
        self.head = nn.Sequential(
            _gn(base, groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )
        self._init_head_bias(init_alpha3d)

    def _init_head_bias(self, init_alpha3d: float) -> None:
        init_alpha3d = float(min(max(init_alpha3d, 1e-4), 1.0 - 1e-4))
        bias = torch.logit(torch.tensor(init_alpha3d)).item()
        last = self.head[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.up2(m)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.head(d1))


def fuse_alpha3d(alpha_3d: torch.Tensor, dnr2: torch.Tensor, dnr3: torch.Tensor) -> torch.Tensor:
    """fused = alpha_3d * 3DNR + (1-alpha_3d) * 2DNR."""
    if alpha_3d.shape[1] == 1 and dnr2.shape[1] > 1:
        alpha_3d = alpha_3d.repeat(1, dnr2.shape[1], 1, 1)
    return alpha_3d * dnr3 + (1.0 - alpha_3d) * dnr2


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

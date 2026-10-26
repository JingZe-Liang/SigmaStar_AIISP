from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class NAFBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int | None = 64, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, hidden * 2, 1)
        self.conv4 = nn.Conv2d(hidden, channels, 1)
        self.time1 = nn.Linear(time_dim, channels * 2) if time_dim else None
        self.time2 = nn.Linear(time_dim, channels * 2) if time_dim else None
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _gate(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second

    @staticmethod
    def _film(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        gamma, beta = params.chunk(2, dim=1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        if self.time1 is not None:
            h = self._film(h, self.time1(time_embedding))
        h = self._gate(self.dwconv(self.conv1(h)))
        h = h * self.sca(h)
        h = self.conv2(h)
        x = x + h * self.beta
        h = self.norm2(x)
        if self.time2 is not None:
            h = self._film(h, self.time2(time_embedding))
        h = self._gate(self.conv3(h))
        h = self.conv4(h)
        return x + h * self.gamma

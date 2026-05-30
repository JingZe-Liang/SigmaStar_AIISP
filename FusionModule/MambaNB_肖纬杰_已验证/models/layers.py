from __future__ import annotations

"""Reusable low-level layers for the lightweight RAW fusion backbone."""

from typing import Literal

import torch.nn as nn


ActivationType = Literal["gelu", "silu"]


def build_activation(name: ActivationType) -> nn.Module:
    """Create the activation module used by lightweight stem blocks."""
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ConvDepthwisePointwise(nn.Module):
    """Lightweight conv block used by both frame stem and motion stem.

    结构与设计文档保持一致：
    `Conv 3x3 -> DWConv 3x3 -> Pointwise Conv 1x1 -> Activation`
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: ActivationType,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        self.in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.depthwise = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )
        self.out_proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.activation = build_activation(activation)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.depthwise(x)
        x = self.out_proj(x)
        return self.activation(x)

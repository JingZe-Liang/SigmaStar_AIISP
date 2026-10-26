from __future__ import annotations

import math

import torch
from torch import nn

from .naf_block import NAFBlock


class BottleneckAttention(nn.Module):
    """Low-resolution spatial attention used only at the NAF U-Net bottleneck."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError("bottleneck channels must be divisible by attention_heads")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float()[:, None] * frequencies[None]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class NAFDiffusion(nn.Module):
    """Multi-scale NAF U-Net that predicts epsilon for VP diffusion."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int | None = None,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_level: int = 2,
        time_dim: int = 64,
        bottleneck_attention_blocks: int = 0,
        attention_heads: int = 8,
    ):
        super().__init__()
        if len(widths) < 2 or any(width <= 0 for width in widths):
            raise ValueError("widths must contain at least two positive levels")
        if blocks_per_level <= 0:
            raise ValueError("blocks_per_level must be positive")
        if bottleneck_attention_blocks < 0:
            raise ValueError("bottleneck_attention_blocks must be nonnegative")
        self.time = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.intro = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*[NAFBlock(width, time_dim) for _ in range(blocks_per_level)])
            for width in widths
        )
        self.downs = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
            for i in range(len(widths) - 1)
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
            for i in range(len(widths) - 2, -1, -1)
        )
        self.decoders = nn.ModuleList(
            nn.Sequential(*[NAFBlock(widths[i], time_dim) for _ in range(blocks_per_level)])
            for i in range(len(widths) - 2, -1, -1)
        )
        self.bottleneck_attention = nn.Sequential(*[
            BottleneckAttention(widths[-1], attention_heads)
            for _ in range(bottleneck_attention_blocks)
        ])
        self.outro = nn.Conv2d(widths[0], out_channels or in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time(timestep)
        current = self.intro(x)
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            for block in encoder:
                current = block(current, time_embedding)
            if level < len(self.downs):
                skips.append(current)
                current = self.downs[level](current)
        current = self.bottleneck_attention(current)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            current = up(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = torch.nn.functional.interpolate(
                    current, size=skip.shape[-2:], mode="nearest"
                )
            current = current + skip
            for block in decoder:
                current = block(current, time_embedding)
        return self.outro(current)


class NAFRestorer(nn.Module):
    """Deterministic NAF baseline with the same multi-scale capacity."""

    def __init__(
        self,
        in_channels: int = 4,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_level: int = 2,
        bottleneck_attention_blocks: int = 0,
        attention_heads: int = 8,
    ):
        super().__init__()
        if len(widths) < 2 or any(width <= 0 for width in widths):
            raise ValueError("widths must contain at least two positive levels")
        if bottleneck_attention_blocks < 0:
            raise ValueError("bottleneck_attention_blocks must be nonnegative")
        self.intro = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*[NAFBlock(width, None) for _ in range(blocks_per_level)])
            for width in widths
        )
        self.downs = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
            for i in range(len(widths) - 1)
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
            for i in range(len(widths) - 2, -1, -1)
        )
        self.decoders = nn.ModuleList(
            nn.Sequential(*[NAFBlock(widths[i], None) for _ in range(blocks_per_level)])
            for i in range(len(widths) - 2, -1, -1)
        )
        self.bottleneck_attention = nn.Sequential(*[
            BottleneckAttention(widths[-1], attention_heads)
            for _ in range(bottleneck_attention_blocks)
        ])
        self.outro = nn.Conv2d(widths[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = self.intro(x)
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            for block in encoder:
                current = block(current, None)
            if level < len(self.downs):
                skips.append(current)
                current = self.downs[level](current)
        current = self.bottleneck_attention(current)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            current = up(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = torch.nn.functional.interpolate(
                    current, size=skip.shape[-2:], mode="nearest"
                )
            current = current + skip
            for block in decoder:
                current = block(current, None)
        return self.outro(current)

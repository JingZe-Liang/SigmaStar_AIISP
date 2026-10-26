"""RViDeformer-style temporal conditioning for clean-video RAW diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return max(1, min(8, channels // 8))


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timestep.device) / max(half - 1, 1)
        )
        angle = timestep.float()[:, None] * frequency[None]
        return self.proj(torch.cat((angle.sin(), angle.cos()), dim=1))


class TemporalAttention(nn.Module):
    """Spatial window queries attend to all frames in an expanded neighborhood."""

    def __init__(self, channels: int, heads: int = 4, window: int = 4, halo: int = 2):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.window, self.halo = window, halo
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels))

    def forward(self, x: torch.Tensor, frame_embedding: torch.Tensor | None = None,
                context: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        w, halo = self.window, self.halo
        ph, pw = (-height) % w, (-width) % w
        source = x if context is None else context
        if source.shape != x.shape:
            raise ValueError("attention context must match query shape")
        if frame_embedding is not None:
            position = frame_embedding[None, :, :, None, None].to(x.dtype)
            x, source = x + position, source + position
        def windows(value, size, padding):
            flat = F.pad(value.reshape(batch * frames, channels, height, width), (0, pw, 0, ph))
            patches = F.unfold(flat, kernel_size=size, stride=w, padding=padding)
            count = patches.shape[-1]
            return patches.reshape(batch, frames, channels, size * size, count).permute(0, 4, 1, 3, 2).reshape(batch * count, frames * size * size, channels)
        query = windows(x, w, 0)
        size = w + 2 * halo
        keys = windows(source, size, halo)
        valid = F.pad(torch.ones(1, 1, height, width, device=x.device), (0, pw, 0, ph))
        valid = F.unfold(valid, kernel_size=size, stride=w, padding=halo)[0].T.bool()
        mask = ~valid[:, None, :].expand(-1, frames, -1).reshape(valid.shape[0], -1)
        mask = mask.repeat(batch, 1)
        results = []
        # Bound attention workspace without restricting the neighborhood.
        for offset in range(0, query.shape[0], 32):
            q = query[offset:offset + 32]
            kv = self.norm(keys[offset:offset + 32])
            attended, _ = self.attn(self.norm(q), kv, kv,
                                    key_padding_mask=mask[offset:offset + 32], need_weights=False)
            result = q + attended
            results.append(result + self.ff(self.ff_norm(result)))
        result = torch.cat(results)
        count = valid.shape[0]
        result = result.reshape(batch, count, frames, w * w, channels).permute(0, 2, 4, 3, 1)
        result = result.reshape(batch * frames, channels * w * w, count)
        result = F.fold(result, (height + ph, width + pw), kernel_size=w, stride=w)
        return result.reshape(batch, frames, channels, height + ph, width + pw)[..., :height, :width]


class SpatialResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class VideoResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self.time = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        y = y + self.time(time_embedding)[:, :, None, None, None]
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class VideoObservationEncoder(nn.Module):
    """Multi-scale spatial encoder with neighborhood cross-frame attention."""

    def __init__(self, in_channels: int, widths: tuple[int, ...], blocks: int, heads: int, camera_dim: int = 0):
        super().__init__()
        self.widths = widths
        self.camera_dim = camera_dim
        if camera_dim:
            self.camera = nn.ModuleList(nn.Linear(camera_dim, width) for width in widths)
            for layer in self.camera:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.intro = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(*[SpatialResBlock(width) for _ in range(blocks)]) for width in widths
        )
        self.downs = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], 2, stride=2) for i in range(len(widths) - 1)
        )
        self.temporal = nn.ModuleList(TemporalAttention(width, heads) for width in widths)
        self.aux_projections = nn.ModuleList(nn.Conv2d(width, widths[0], 1) for width in widths)
        self.aux = nn.Sequential(SpatialResBlock(widths[0]), nn.Conv2d(widths[0], in_channels, 3, padding=1))
        frame_dim = widths[0]
        self.frame_embedding = nn.Parameter(torch.zeros(32, frame_dim))
        nn.init.normal_(self.frame_embedding, std=0.02)
        self.frame_projections = nn.ModuleList(
            nn.Linear(frame_dim, width) for width in widths
        )

    def forward(self, noisy: torch.Tensor, camera: torch.Tensor | None = None) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if noisy.ndim != 5:
            raise ValueError("noisy must have shape [B,T,C,H,W]")
        batch, frames, channels, height, width = noisy.shape
        if self.camera_dim and (camera is None or camera.shape != (batch, self.camera_dim)):
            raise ValueError("measured camera features are required for every clip")
        if frames > self.frame_embedding.shape[0]:
            raise ValueError("clip length exceeds configured frame embedding table")
        current = self.intro(noisy.reshape(batch * frames, channels, height, width))
        conditions: list[torch.Tensor] = []
        for level, block in enumerate(self.blocks):
            if self.camera_dim:
                bias = self.camera[level](camera).repeat_interleave(frames, 0)
                current = current + bias[:, :, None, None]
            current = block(current)
            video = current.reshape(batch, frames, current.shape[1], current.shape[2], current.shape[3])
            positions = self.frame_projections[level](self.frame_embedding[:frames])
            video = self.temporal[level](video, positions)
            conditions.append(video)
            if level < len(self.downs):
                current = self.downs[level](video.reshape(batch * frames, video.shape[2], video.shape[3], video.shape[4]))
        fused = 0
        for feature, projection in zip(conditions, self.aux_projections):
            flat = feature.reshape(batch * frames, *feature.shape[2:])
            fused = fused + F.interpolate(projection(flat), size=(height, width), mode="bilinear", align_corners=False)
        aux = torch.sigmoid(self.aux(fused / len(conditions)))
        return tuple(conditions), aux.reshape(batch, frames, channels, height, width)


class VideoDiffusionUNet(nn.Module):
    def __init__(self, in_channels: int, widths: tuple[int, ...], blocks: int, time_dim: int, heads: int, camera_dim: int = 0, reconstruction_guidance: bool = False):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.camera_dim = camera_dim
        self.reconstruction_guidance = reconstruction_guidance
        if camera_dim:
            self.camera = nn.Linear(camera_dim, time_dim)
            nn.init.zeros_(self.camera.weight)
            nn.init.zeros_(self.camera.bias)
        if reconstruction_guidance:
            self.guidance = nn.ModuleList(nn.Conv3d(in_channels, width, 1) for width in widths)
            for layer in self.guidance:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.input = nn.Conv3d(in_channels, widths[0], (1, 3, 3), padding=(0, 1, 1))
        self.encoders = nn.ModuleList()
        self.condition = nn.ModuleList()
        self.temporal = nn.ModuleList()
        self.cross = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level, width in enumerate(widths):
            self.encoders.append(nn.ModuleList(VideoResBlock(width, time_dim) for _ in range(blocks)))
            self.condition.append(nn.Conv3d(width, width, 1))
            self.temporal.append(TemporalAttention(width, heads))
            self.cross.append(TemporalAttention(width, heads))
            if level < len(widths) - 1:
                self.downs.append(nn.Conv3d(width, widths[level + 1], (1, 2, 2), stride=(1, 2, 2)))
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(len(widths) - 2, -1, -1):
            self.ups.append(nn.ConvTranspose3d(widths[level + 1], widths[level], (1, 2, 2), stride=(1, 2, 2)))
            self.decoders.append(nn.ModuleList(VideoResBlock(widths[level], time_dim) for _ in range(blocks)))
        self.out = nn.Conv3d(widths[0], in_channels, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, state: torch.Tensor, timestep: torch.Tensor, conditions: tuple[torch.Tensor, ...], camera: torch.Tensor | None = None, guidance: torch.Tensor | None = None) -> torch.Tensor:
        time_embedding = self.time(timestep)
        if self.camera_dim:
            if camera is None or camera.shape != (state.shape[0], self.camera_dim):
                raise ValueError("measured camera features are required for sampling")
            time_embedding = time_embedding + self.camera(camera)
        if self.reconstruction_guidance and (guidance is None or guidance.shape != state.shape):
            raise ValueError("clean reconstruction guidance must match the video state")
        if state.ndim != 5:
            raise ValueError("state must have shape [B,T,C,H,W]")
        current = self.input(state.permute(0, 2, 1, 3, 4))
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.encoders):
            if self.reconstruction_guidance:
                anchor = F.interpolate(guidance.permute(0, 2, 1, 3, 4), size=current.shape[-3:], mode="trilinear", align_corners=False)
                current = current + self.guidance[level](anchor)
            for block in blocks:
                current = block(current, time_embedding)
            current = current + self.condition[level](conditions[level].permute(0, 2, 1, 3, 4))
            video = current.permute(0, 2, 1, 3, 4)
            video = self.temporal[level](video)
            video = self.cross[level](video, context=conditions[level])
            current = video.permute(0, 2, 1, 3, 4)
            if level < len(self.downs):
                skips.append(current)
                current = self.downs[level](current)
        for level, (up, blocks, skip) in enumerate(zip(self.ups, self.decoders, reversed(skips))):
            current = up(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = F.interpolate(current, size=(current.shape[2], skip.shape[-2], skip.shape[-1]), mode="trilinear", align_corners=False)
            current = current + skip
            if self.reconstruction_guidance:
                anchor = F.interpolate(guidance.permute(0, 2, 1, 3, 4), size=current.shape[-3:], mode="trilinear", align_corners=False)
                current = current + self.guidance[len(self.encoders) - 2 - level](anchor)
            for block in blocks:
                current = block(current, time_embedding)
        return self.out(current).permute(0, 2, 1, 3, 4)


class CleanVideoDiffusion(nn.Module):
    """Complete conditional clean-video diffusion model."""

    def __init__(self, in_channels: int = 4, widths: tuple[int, ...] = (64, 128, 256), blocks: int = 2, time_dim: int = 128, heads: int = 4, camera_dim: int = 0, reconstruction_guidance: bool = False):
        super().__init__()
        self.observation = VideoObservationEncoder(in_channels, widths, blocks, heads, camera_dim)
        self.denoiser = VideoDiffusionUNet(in_channels, widths, blocks, time_dim, heads, camera_dim, reconstruction_guidance)

    def forward(self, noisy: torch.Tensor, state: torch.Tensor, timestep: torch.Tensor, camera: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        conditions, auxiliary = self.observation(noisy, camera)
        prediction = self.denoiser(state, timestep, conditions, camera, auxiliary.detach())
        return prediction, auxiliary

"""Identity-safe multi-frame residual refinement above a frozen R0."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .video_diffusion import VideoObservationEncoder, TemporalAttention


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class TemporalResidualRefiner(nn.Module):
    """Use learned local warps of neighboring RAW frames to refine R0.

    The output head is zero initialized, so an untrained checkpoint is exactly
    the R0 baseline. Warps are bounded and confidence weighted to limit motion
    ghosts and occlusion leakage.
    """

    def __init__(self, channels: int = 32, blocks: int = 4, max_flow_px: float = 8.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_flow_px = float(max_flow_px)
        self.stem = nn.Sequential(nn.Conv2d(4, channels, 3, padding=1), nn.GELU())
        self.encoder = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.flow = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels + 8, channels, 3, padding=1),
            nn.GELU(),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )
        self.out = nn.Conv2d(channels, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1)[None]

    def _warp(self, feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        scale = feature.new_tensor((2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1))).view(1, 2, 1, 1)
        normalized = flow.tanh() * self.max_flow_px * scale
        grid = self._grid(height, width, feature.device, feature.dtype) + normalized.permute(0, 2, 3, 1)
        return F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, noisy: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 5 or noisy.shape[2] != 4:
            raise ValueError("noisy must have shape [B,T,4,H,W]")
        if baseline.shape != noisy[:, noisy.shape[1] // 2].shape:
            raise ValueError("baseline must be the center-frame shape [B,4,H,W]")
        batch, frames = noisy.shape[:2]
        encoded = self.encoder(self.stem(noisy.reshape(batch * frames, 4, *noisy.shape[-2:])))
        encoded = encoded.reshape(batch, frames, self.channels, *noisy.shape[-2:])
        center_index = frames // 2
        center = encoded[:, center_index]
        fused = center
        total = torch.ones_like(center[:, :1])
        for index in range(frames):
            if index == center_index:
                continue
            neighbor = encoded[:, index]
            pair = torch.cat((center, neighbor), dim=1)
            warped = self._warp(neighbor, self.flow(pair))
            confidence = torch.sigmoid(self.confidence(pair))
            fused = fused + confidence * warped
            total = total + confidence
        fused = fused / total
        decoder_input = torch.cat((fused, baseline, noisy[:, center_index]), dim=1)
        return baseline + self.out(self.decoder(decoder_input))


class TemporalResidualRefinerV2(TemporalResidualRefiner):
    """Multi-hypothesis local alignment for larger motion and occlusions."""

    def __init__(self, channels: int = 32, blocks: int = 4, max_flow_px: float = 12.0) -> None:
        super().__init__(channels=channels, blocks=blocks, max_flow_px=max_flow_px)
        self.alignment_logits = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 9, 3, padding=1),
        )

    def _fixed_warp(self, feature: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        _, _, height, width = feature.shape
        grid = self._grid(height, width, feature.device, feature.dtype).expand(feature.shape[0], -1, -1, -1).clone()
        grid[..., 0] += 2.0 * dx / max(width - 1, 1)
        grid[..., 1] += 2.0 * dy / max(height - 1, 1)
        return F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, noisy: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 5 or noisy.shape[2] != 4:
            raise ValueError("noisy must have shape [B,T,4,H,W]")
        if baseline.shape != noisy[:, noisy.shape[1] // 2].shape:
            raise ValueError("baseline must be the center-frame shape [B,4,H,W]")
        batch, frames = noisy.shape[:2]
        encoded = self.encoder(self.stem(noisy.reshape(batch * frames, 4, *noisy.shape[-2:])))
        encoded = encoded.reshape(batch, frames, self.channels, *noisy.shape[-2:])
        center_index = frames // 2
        center = encoded[:, center_index]
        fused = center
        total = torch.ones_like(center[:, :1])
        offsets = ((0, 0), (-self.max_flow_px, 0), (self.max_flow_px, 0),
                   (0, -self.max_flow_px), (0, self.max_flow_px),
                   (-self.max_flow_px, -self.max_flow_px), (-self.max_flow_px, self.max_flow_px),
                   (self.max_flow_px, -self.max_flow_px), (self.max_flow_px, self.max_flow_px))
        for index in range(frames):
            if index == center_index:
                continue
            neighbor = encoded[:, index]
            pair = torch.cat((center, neighbor), dim=1)
            weights = torch.softmax(self.alignment_logits(pair), dim=1)
            warped = torch.stack([self._fixed_warp(neighbor, dx, dy) for dx, dy in offsets], dim=1)
            aligned = (warped * weights[:, :, None]).sum(dim=1)
            confidence = torch.sigmoid(self.confidence(pair))
            fused = fused + confidence * aligned
            total = total + confidence
        fused = fused / total
        decoder_input = torch.cat((fused, baseline, noisy[:, center_index]), dim=1)
        return baseline + self.out(self.decoder(decoder_input))


class PyramidTemporalRefiner(nn.Module):
    """Large RAW video refiner with coarse-to-fine alignment and noise conditioning."""

    def __init__(self, widths=(64, 128, 256, 384), blocks=(2, 3, 4, 6), max_flow_px=32.0, residual_parameterization="noise", temporal_attention_levels=(), attention_heads=8, attention_window=4, attention_kind="neighbor", time_dim=0):
        super().__init__()
        if len(widths) != len(blocks) or len(widths) < 2:
            raise ValueError("widths and blocks must define the same >=2 levels")
        self.widths = tuple(widths)
        if len(set(temporal_attention_levels)) != len(temporal_attention_levels) or any(i < 0 or i >= len(widths) for i in temporal_attention_levels):
            raise ValueError("temporal attention levels must be unique valid pyramid levels")
        if attention_kind not in ("neighbor", "hat"):
            raise ValueError("attention_kind must be neighbor or hat")
        if attention_kind == "hat":
            from .hat_adapter import AuthorHATHybridAdapter
        self.temporal_adapters = nn.ModuleDict({str(i): (AuthorHATHybridAdapter(widths[i], attention_heads, attention_window) if attention_kind == "hat" else TemporalAttention(widths[i], heads=attention_heads, window=attention_window, halo=1)) for i in temporal_attention_levels})
        self.temporal_gates = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(())) for i in temporal_attention_levels})
        self.max_flow_px = float(max_flow_px)
        if residual_parameterization not in ("noise", "raw"):
            raise ValueError("residual_parameterization must be noise or raw")
        self.residual_parameterization = residual_parameterization
        self.time_dim = int(time_dim)
        self.intro = nn.Conv2d(4, widths[0], 3, padding=1)
        self.encoder = nn.ModuleList(nn.Sequential(*[ResidualBlock(c) for _ in range(n)]) for c, n in zip(widths, blocks))
        self.downs = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths[:-1], widths[1:]))
        self.condition = nn.ModuleList(nn.Linear(4 + self.time_dim, 2 * c) for c in widths)
        self.flow = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * c + 2, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 2, 3, padding=1)) for c in widths)
        self.confidence = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * c, c // 2, 3, padding=1), nn.GELU(), nn.Conv2d(c // 2, 1, 3, padding=1)) for c in widths)
        self.fuse = nn.ModuleList(nn.Conv2d(2 * c, c, 1) for c in widths)
        self.ups = nn.ModuleList(nn.Conv2d(widths[i + 1], widths[i], 1) for i in range(len(widths) - 2, -1, -1))
        self.decoder = nn.ModuleList(nn.Sequential(*[ResidualBlock(widths[i]) for _ in range(blocks[i])]) for i in range(len(widths) - 2, -1, -1))
        self.output_features = nn.Sequential(nn.Conv2d(widths[0] + 8, widths[0], 3, padding=1), nn.GELU(), ResidualBlock(widths[0]))
        self.out = nn.Conv2d(widths[0], 4, 3, padding=1)
        for layer in self.flow:
            nn.init.zeros_(layer[-1].weight); nn.init.zeros_(layer[-1].bias)
        for layer in self.condition:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    @staticmethod
    def estimate_noise(noisy):
        """Robust per-channel HH-wavelet estimate; needs no clean or ISO oracle."""
        h, w = noisy.shape[-2:]
        x = noisy[..., :h - h % 2, :w - w % 2]
        hh = (x[..., ::2, ::2] - x[..., 1::2, ::2] - x[..., ::2, 1::2] + x[..., 1::2, 1::2]) * 0.5
        values = hh.permute(0, 2, 1, 3, 4).flatten(2)
        return (values.abs().median(dim=-1).values / 0.67448975).clamp(1 / 4095, 0.2)

    @staticmethod
    def _warp(x, flow):
        h, w = x.shape[-2:]
        scale = x.new_tensor((2 / max(w - 1, 1), 2 / max(h - 1, 1))).view(1, 2, 1, 1)
        grid = TemporalResidualRefiner._grid(h, w, x.device, x.dtype)
        return F.grid_sample(x, grid + (flow * scale).permute(0, 2, 3, 1), align_corners=True, padding_mode="border")

    def forward(self, noisy, baseline, noise_level=None):
        if noisy.ndim != 5 or noisy.shape[2] != 4 or noisy.shape[1] % 2 != 1:
            raise ValueError("noisy must have shape [B,odd T,4,H,W]")
        b, t, _, h, w = noisy.shape
        if baseline.shape != (b, 4, h, w):
            raise ValueError("baseline must match the center frame")
        sigma = self.estimate_noise(noisy) if noise_level is None else noise_level
        features = self.encode_features(noisy, sigma)
        return self.refine_features(features, noisy, baseline, t // 2, sigma)

    def encode_features(self, noisy, sigma, time_embedding=None):
        """Shared seven-frame pyramid; diffusion supplies its exact noise scale."""
        b, t, _, h, w = noisy.shape
        conditioning = torch.log((sigma / 0.01).clamp_min(1e-4))
        if self.time_dim:
            if time_embedding is None or time_embedding.shape != (b, self.time_dim):
                raise ValueError("diffusion pyramid requires a matching timestep embedding")
            conditioning = torch.cat((conditioning, time_embedding), dim=-1)
        current = self.intro(noisy.reshape(b * t, 4, h, w))
        features = []
        for level, encoder in enumerate(self.encoder):
            gamma, bias = self.condition[level](conditioning).chunk(2, dim=-1)
            gamma = gamma[:, None].expand(-1, t, -1).reshape(b * t, -1, 1, 1)
            bias = bias[:, None].expand(-1, t, -1).reshape(b * t, -1, 1, 1)
            current = encoder(current * (1 + 0.1 * gamma) + 0.1 * bias)
            level_features = current.reshape(b, t, *current.shape[1:])
            key = str(level)
            if key in self.temporal_adapters:
                delta = self.temporal_adapters[key](level_features) - level_features
                level_features = level_features + self.temporal_gates[key].tanh() * delta
            features.append(level_features)
            if level < len(self.downs):
                current = self.downs[level](current)
        return features

    def refine_features(self, features, noisy, baseline, center_index, sigma):
        """Choose any target frame without re-encoding its six neighbors."""
        b, t = noisy.shape[:2]
        if not 0 <= center_index < t:
            raise ValueError("center index is outside the clip")
        flows = [None] * t
        skips = [None] * len(features)
        for level in reversed(range(len(features))):
            center = features[level][:, center_index]
            fused, total = center, torch.ones_like(center[:, :1])
            for index in range(t):
                if index == center_index:
                    continue
                neighbor = features[level][:, index]
                old = flows[index]
                if old is None:
                    flow = center.new_zeros((b, 2, *center.shape[-2:]))
                else:
                    flow = F.interpolate(old, size=center.shape[-2:], mode="bilinear", align_corners=False) * 2
                aligned = self._warp(neighbor, flow)
                increment = self.flow[level](torch.cat((center, aligned, flow), dim=1)).tanh() * 4
                bound = self.max_flow_px / 2 ** level
                flow = (flow + increment).clamp(-bound, bound)
                aligned = self._warp(neighbor, flow)
                confidence = self.confidence[level](torch.cat((center, aligned), dim=1)).sigmoid()
                fused = fused + confidence * aligned
                total = total + confidence
                flows[index] = flow
            skips[level] = self.fuse[level](torch.cat((center, fused / total), dim=1))
        current = skips[-1]
        for up, decoder, skip in zip(self.ups, self.decoder, reversed(skips[:-1])):
            current = decoder(up(F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)) + skip)
        residual = self.out(self.output_features(torch.cat((current, baseline, noisy[:, center_index]), dim=1)))
        return baseline + (residual * sigma[:, :, None, None] if self.residual_parameterization == "noise" else residual)


def refiner_from_checkpoint(checkpoint):
    """Reconstruct the architecture recorded in a temporal checkpoint."""
    variant = checkpoint.get("variant", "v1")
    config = checkpoint.get("model_config", {})
    if variant == "pyramid":
        return PyramidTemporalRefiner(**config)
    if variant == "attention":
        return TemporalAttentionRefiner(**config)
    cls = TemporalResidualRefinerV2 if variant == "v2" else TemporalResidualRefiner
    return cls(**(config or {"max_flow_px": 12.0 if variant == "v2" else 8.0}))


class TemporalAttentionRefiner(nn.Module):
    """RViDeformer-style multi-scale temporal attention with an R0 residual head."""

    def __init__(self, channels: int = 32, blocks: int = 1, max_flow_px: float = 8.0) -> None:
        del max_flow_px
        super().__init__()
        widths = (channels, channels * 2)
        self.observation = VideoObservationEncoder(4, widths, blocks, heads=4)
        self.head = nn.Sequential(
            nn.Conv2d(channels + 12, channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        self.out = nn.Conv2d(channels, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 5 or noisy.shape[2] != 4:
            raise ValueError("noisy must have shape [B,T,4,H,W]")
        conditions, auxiliary = self.observation(noisy)
        center = noisy.shape[1] // 2
        features = conditions[0][:, center]
        input_features = torch.cat((features, auxiliary[:, center], baseline, noisy[:, center]), dim=1)
        return baseline + self.out(self.head(input_features))

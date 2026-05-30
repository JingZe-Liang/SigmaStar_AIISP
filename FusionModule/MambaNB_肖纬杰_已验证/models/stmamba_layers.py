from __future__ import annotations

"""Reusable layers for the current implicit ST-Mamba-Lite backbone."""

import torch
import torch.nn as nn
from torch import Tensor

from .mamba_scan import MambaSelectiveScan1D, ScanBackend


def _validate_feature_map(name: str, feature_map: Tensor, channels: int | None = None) -> None:
    """Validate `[B, C, H, W]` feature-map layout."""
    if feature_map.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {feature_map.shape}")
    if channels is not None and feature_map.shape[1] != channels:
        raise ValueError(f"{name} channel mismatch: expected {channels}, got {feature_map.shape[1]}")


class LocalContextMixer(nn.Module):
    """Lightweight residual local context mixer shared by RAW frame features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            ),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )

    def forward(self, feature_map: Tensor) -> Tensor:
        _validate_feature_map("feature_map", feature_map, self.channels)
        return feature_map + self.net(feature_map)


class GatedFFN(nn.Module):
    """Lightweight gated channel feed-forward branch for image feature maps."""

    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expand <= 0:
            raise ValueError(f"expand must be positive, got {expand}")

        self.channels = channels
        self.expand = expand
        hidden_channels = channels * expand
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.in_proj = nn.Conv2d(
            in_channels=channels,
            out_channels=hidden_channels * 2,
            kernel_size=1,
            bias=True,
        )
        self.depthwise = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.out_proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feature_map: Tensor) -> Tensor:
        _validate_feature_map("feature_map", feature_map, self.channels)
        projected = self.depthwise(self.in_proj(self.norm(feature_map)))
        value, gate = projected.chunk(2, dim=1)
        return self.out_proj(value * self.activation(gate))


class SpatioTemporalBidirectionalSSM3D(nn.Module):
    """Implicit bidirectional scan over a `[B, C, T, H, W]` feature cube."""

    PATHS: tuple[tuple[str, str, str], ...] = (
        ("T", "H", "W"),
        ("T", "W", "H"),
        ("H", "W", "T"),
        ("W", "H", "T"),
    )
    DIM_INDEX = {"T": 2, "H": 3, "W": 4}

    def __init__(
        self,
        channels: int,
        in_channels: int | None = None,
        state_dim: int = 8,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_kernel: int = 3,
        scan_backend: ScanBackend = "auto",
        share_path_scan: bool = False,
        fusion: str = "softmax",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if fusion != "softmax":
            raise ValueError(f"Unsupported STB direction fusion: {fusion}")

        self.channels = channels
        self.in_channels = in_channels or channels
        self.share_path_scan = share_path_scan
        self.fusion = fusion
        self.input_proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=channels,
            kernel_size=1,
            bias=True,
        )

        num_scans = 1 if share_path_scan else len(self.PATHS) * 2
        self.scans = nn.ModuleList(
            [
                MambaSelectiveScan1D(
                    channels=channels,
                    state_dim=state_dim,
                    expand=expand,
                    dt_rank=dt_rank,
                    conv_kernel=conv_kernel,
                    scan_backend=scan_backend,
                )
                for _ in range(num_scans)
            ]
        )
        self.direction_logits = nn.Parameter(torch.zeros(len(self.PATHS) * 2))
        self.out_proj = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
        )

    def forward(self, video_feature: Tensor) -> Tensor:
        if video_feature.ndim != 5:
            raise ValueError(
                f"video_feature must have shape [B, C, T, H, W], got {video_feature.shape}"
            )
        if video_feature.shape[1] != self.in_channels:
            raise ValueError(
                f"video_feature channel mismatch: expected {self.in_channels}, "
                f"got {video_feature.shape[1]}"
            )

        projected = self.input_proj(video_feature)
        path_outputs: list[Tensor] = []
        scan_index = 0

        for order in self.PATHS:
            for reverse in (False, True):
                sequence, ordered_sizes = self._flatten_path(projected, order)
                if reverse:
                    sequence = torch.flip(sequence, dims=[1])

                scan_module = self.scans[0] if self.share_path_scan else self.scans[scan_index]
                scanned_sequence = scan_module(sequence)
                if reverse:
                    scanned_sequence = torch.flip(scanned_sequence, dims=[1])

                restored = self._restore_path(
                    scanned_sequence,
                    order,
                    ordered_sizes,
                    tuple(projected.shape),
                )
                path_outputs.append(restored)
                scan_index += 1

        direction_weights = torch.softmax(self.direction_logits, dim=0)
        directional_features = torch.stack(path_outputs, dim=0)
        mixed_feature = (
            direction_weights.view(len(path_outputs), 1, 1, 1, 1, 1) * directional_features
        ).sum(dim=0)
        return self.out_proj(mixed_feature + projected)

    @staticmethod
    def _flatten_path(
        video_feature: Tensor,
        order: tuple[str, str, str],
    ) -> tuple[Tensor, tuple[int, int, int]]:
        """Flatten `[B, C, T, H, W]` into `[B, L, C]` with a named axis order."""
        if video_feature.ndim != 5:
            raise ValueError(
                f"video_feature must have shape [B, C, T, H, W], got {video_feature.shape}"
            )
        if set(order) != {"T", "H", "W"} or len(order) != 3:
            raise ValueError(f"order must be a permutation of ('T', 'H', 'W'), got {order}")

        batch_size, channels, frames, height, width = video_feature.shape
        dim_sizes = {"T": frames, "H": height, "W": width}
        permute_dims = [0] + [SpatioTemporalBidirectionalSSM3D.DIM_INDEX[name] for name in order] + [1]
        sequence = video_feature.permute(*permute_dims).reshape(
            batch_size,
            frames * height * width,
            channels,
        )
        return sequence, tuple(dim_sizes[name] for name in order)

    @staticmethod
    def _restore_path(
        sequence: Tensor,
        order: tuple[str, str, str],
        ordered_sizes: tuple[int, int, int],
        target_shape: tuple[int, int, int, int, int],
    ) -> Tensor:
        """Restore a flattened path sequence back to `[B, C, T, H, W]`."""
        if sequence.ndim != 3:
            raise ValueError(f"sequence must have shape [B, L, C], got {sequence.shape}")
        if set(order) != {"T", "H", "W"} or len(order) != 3:
            raise ValueError(f"order must be a permutation of ('T', 'H', 'W'), got {order}")

        batch_size, channels, frames, height, width = target_shape
        expected_length = frames * height * width
        if sequence.shape != (batch_size, expected_length, channels):
            raise ValueError(
                "sequence must match target batch/length/channels, "
                f"got {sequence.shape} vs {(batch_size, expected_length, channels)}"
            )

        restored = sequence.reshape(batch_size, *ordered_sizes, channels)
        source_pos = {name: index + 1 for index, name in enumerate(order)}
        restore_dims = [0, 4, source_pos["T"], source_pos["H"], source_pos["W"]]
        return restored.permute(*restore_dims).contiguous()

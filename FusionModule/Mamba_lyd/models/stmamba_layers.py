from __future__ import annotations

"""Reusable layers for the current implicit ST-Mamba-Lite backbone."""

import torch
import torch.nn as nn
from torch import Tensor

from .mamba_scan import Mamba2SSD1D, MambaSelectiveScan1D, MambaVariant, ScanBackend
from .stmamba_config import ScanPathMode


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

    ALL_PATHS: tuple[tuple[str, str, str], ...] = (
        ("T", "H", "W"),
        ("T", "W", "H"),
        ("H", "W", "T"),
        ("W", "H", "T"),
    )
    TEMPORAL_FIRST_PATHS: tuple[tuple[str, str, str], ...] = (
        ("T", "H", "W"),
        ("T", "W", "H"),
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
        mamba_variant: MambaVariant = "mamba1",
        mamba2_state_dim: int = 64,
        mamba2_headdim: int | None = None,
        mamba2_groups: int = 1,
        mamba2_chunk_size: int = 256,
        share_path_scan: bool = False,
        fusion: str = "softmax",
        dynamic_direction_fusion: bool = False,
        scan_path_mode: ScanPathMode = "8path",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if fusion != "softmax":
            raise ValueError(f"Unsupported STB direction fusion: {fusion}")
        if scan_path_mode not in {"8path", "temporal4", "temporal4_grouped", "multiscale_grouped"}:
            raise ValueError(f"Unsupported scan_path_mode: {scan_path_mode}")

        self.channels = channels
        self.in_channels = in_channels or channels
        self.share_path_scan = share_path_scan
        self.fusion = fusion
        self.dynamic_direction_fusion = dynamic_direction_fusion
        self.mamba_variant = mamba_variant
        self.scan_path_mode = scan_path_mode
        self.paths = self.ALL_PATHS if scan_path_mode == "8path" else self.TEMPORAL_FIRST_PATHS
        self.grouped_scan = scan_path_mode in {"temporal4_grouped", "multiscale_grouped"}
        self.num_directions = len(self.paths) * 2
        if self.grouped_scan and channels % self.num_directions:
            raise ValueError(
                f"{scan_path_mode} requires channels ({channels}) divisible by "
                f"the number of directions ({self.num_directions})"
            )
        self.scan_channels = channels // self.num_directions if self.grouped_scan else channels
        self.input_proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=channels,
            kernel_size=1,
            bias=True,
        )

        num_scans = 1 if share_path_scan else self.num_directions
        if mamba_variant == "mamba1":
            self.scans = nn.ModuleList(
                [
                    MambaSelectiveScan1D(
                        channels=self.scan_channels,
                        state_dim=state_dim,
                        expand=expand,
                        dt_rank=dt_rank,
                        conv_kernel=conv_kernel,
                        scan_backend=scan_backend,
                    )
                    for _ in range(num_scans)
                ]
            )
        elif mamba_variant == "mamba2":
            self.scans = nn.ModuleList(
                [
                    Mamba2SSD1D(
                        channels=self.scan_channels,
                        state_dim=mamba2_state_dim,
                        expand=expand,
                        conv_kernel=conv_kernel,
                        head_dim=mamba2_headdim,
                        groups=mamba2_groups,
                        chunk_size=mamba2_chunk_size,
                    )
                    for _ in range(num_scans)
                ]
            )
        else:
            raise ValueError(f"Unsupported mamba_variant: {mamba_variant}")
        self.direction_logits = nn.Parameter(torch.zeros(self.num_directions))
        if dynamic_direction_fusion:
            hidden_channels = max(4, channels // 2)
            self.direction_conditioner: nn.Module | None = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, self.num_directions, kernel_size=1, bias=True),
            )
            # The dynamic branch starts as an exact no-op. It then learns a
            # frame/crop-level directional preference from motion features.
            nn.init.zeros_(self.direction_conditioner[-1].weight)
            nn.init.zeros_(self.direction_conditioner[-1].bias)
        else:
            self.direction_conditioner = None
        self.out_proj = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
        )

    def forward(self, video_feature: Tensor, direction_context: Tensor | None = None) -> Tensor:
        if video_feature.ndim != 5:
            raise ValueError(
                f"video_feature must have shape [B, C, T, H, W], got {video_feature.shape}"
            )
        if video_feature.shape[1] != self.in_channels:
            raise ValueError(
                f"video_feature channel mismatch: expected {self.in_channels}, "
                f"got {video_feature.shape[1]}"
            )
        if self.dynamic_direction_fusion:
            if direction_context is None:
                raise ValueError("direction_context is required when dynamic_direction_fusion is enabled")
            _validate_feature_map("direction_context", direction_context, self.channels)
            if direction_context.shape[0] != video_feature.shape[0] or direction_context.shape[-2:] != video_feature.shape[-2:]:
                raise ValueError("direction_context must align with the video feature spatial shape")

        projected = self.input_proj(video_feature)
        path_outputs: list[Tensor] = []
        scan_index = 0
        channel_groups = projected.chunk(self.num_directions, dim=1) if self.grouped_scan else None

        for order in self.paths:
            for reverse in (False, True):
                path_feature = channel_groups[scan_index] if channel_groups is not None else projected
                sequence, ordered_sizes = self._flatten_path(path_feature, order)
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
                    tuple(path_feature.shape),
                )
                path_outputs.append(restored)
                scan_index += 1

        if self.direction_conditioner is None:
            direction_weights = torch.softmax(self.direction_logits, dim=0)
        else:
            dynamic_logits = self.direction_conditioner(direction_context).flatten(1)
            direction_weights = torch.softmax(self.direction_logits.unsqueeze(0) + dynamic_logits, dim=1)

        if self.grouped_scan:
            # Each direction owns one channel group. Weighted concatenation
            # retains directional diversity; out_proj performs cross-group
            # channel mixing after the scan. Softmax weights sum to one,
            # while concatenation needs an N-way scale to preserve the
            # expected feature magnitude at initialization.
            grouped_weights = direction_weights * self.num_directions
            if direction_weights.ndim == 1:
                mixed_feature = torch.cat(
                    [weight * feature for weight, feature in zip(grouped_weights, path_outputs)],
                    dim=1,
                )
            else:
                mixed_feature = torch.cat(
                    [
                        grouped_weights[:, index].view(-1, 1, 1, 1, 1) * feature
                        for index, feature in enumerate(path_outputs)
                    ],
                    dim=1,
                )
        else:
            directional_features = torch.stack(path_outputs, dim=0)
            if direction_weights.ndim == 1:
                mixed_feature = (
                    direction_weights.view(len(path_outputs), 1, 1, 1, 1, 1)
                    * directional_features
                ).sum(dim=0)
            else:
                mixed_feature = (
                    direction_weights.transpose(0, 1)
                    .view(len(path_outputs), video_feature.shape[0], 1, 1, 1, 1)
                    * directional_features
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

from __future__ import annotations

"""ST-Mamba-Lite block implementation for lightweight RAW fusion."""

import torch
import torch.nn as nn
from torch import Tensor

from .stmamba_config import STMambaLiteConfig
from .stmamba_layers import (
    GatedFFN,
    LocalContextMixer,
    SpatioTemporalBidirectionalSSM3D,
)


def _validate_feature_triplet(
    prev_feature: Tensor,
    curr_feature: Tensor,
    motion_feature: Tensor,
    channels: int,
) -> None:
    """Validate the three block inputs follow `[B, C, H, W]` layout."""
    for name, feature_map in {
        "prev_feature": prev_feature,
        "curr_feature": curr_feature,
        "motion_feature": motion_feature,
    }.items():
        if feature_map.ndim != 4:
            raise ValueError(f"{name} must have shape [B, C, H, W], got {feature_map.shape}")
        if feature_map.shape[1] != channels:
            raise ValueError(
                f"{name} channel mismatch: expected {channels}, got {feature_map.shape[1]}"
            )
    if prev_feature.shape != curr_feature.shape or prev_feature.shape != motion_feature.shape:
        raise ValueError(
            "prev_feature, curr_feature and motion_feature must share the same shape, "
            f"got {prev_feature.shape}, {curr_feature.shape}, {motion_feature.shape}"
        )


class STMambaLiteBlock(nn.Module):
    """Current-anchored implicit spatio-temporal fusion block.

    The public forward interface:
    `forward(f_prev, f_curr, f_diff) -> (g_prev, g_curr)`.

    No explicit shift, offset candidate, optical flow, or feature warping is
    used. Cross-frame context is propagated by STB scanning over a
    `[B, C, T=2, H, W]` feature cube.
    """

    def __init__(
        self,
        channels: int = 24,
        config: STMambaLiteConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = STMambaLiteConfig(channels=channels)
        elif channels != 24 and channels != config.channels:
            raise ValueError(
                f"channels override ({channels}) must match config.channels ({config.channels})"
            )

        self.config = config
        self.channels = config.channels

        if config.share_local_mix:
            self.shared_local_mix = LocalContextMixer(config.channels)
            self.prev_local_mix = None
            self.curr_local_mix = None
        else:
            self.shared_local_mix = None
            self.prev_local_mix = LocalContextMixer(config.channels)
            self.curr_local_mix = LocalContextMixer(config.channels)

        self.stb_scan = SpatioTemporalBidirectionalSSM3D(
            channels=config.channels,
            in_channels=config.channels,
            state_dim=config.mamba_state_dim,
            expand=config.mamba_expand,
            dt_rank=config.mamba_dt_rank,
            conv_kernel=config.mamba_conv_kernel,
            scan_backend=config.mamba_scan_backend,
            share_path_scan=config.stb_share_path_scan,
            fusion=config.stb_direction_fusion,
        )
        self.gated_ffn = GatedFFN(
            channels=config.channels,
            expand=config.gated_ffn_expand,
        )

    def forward(
        self,
        prev_feature: Tensor,
        curr_feature: Tensor,
        motion_feature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run strict implicit STB fusion."""
        _validate_feature_triplet(prev_feature, curr_feature, motion_feature, self.channels)

        local_prev_feature = self._mix_prev(prev_feature)
        local_curr_feature = self._mix_curr(curr_feature)

        video_feature = torch.stack([local_prev_feature, local_curr_feature], dim=2)
        video_output = self.stb_scan(video_feature)

        prev_output = video_output[:, :, 0]
        curr_temporal = video_output[:, :, 1]
        curr_output = curr_temporal + self.gated_ffn(curr_temporal)

        return prev_output, curr_output

    def _mix_prev(self, prev_feature: Tensor) -> Tensor:
        """Apply local context mixing to the previous-frame feature."""
        if self.shared_local_mix is not None:
            return self.shared_local_mix(prev_feature)
        if self.prev_local_mix is None:
            raise RuntimeError("prev_local_mix is not initialized")
        return self.prev_local_mix(prev_feature)

    def _mix_curr(self, curr_feature: Tensor) -> Tensor:
        """Apply local context mixing to the current-frame feature."""
        if self.shared_local_mix is not None:
            return self.shared_local_mix(curr_feature)
        if self.curr_local_mix is None:
            raise RuntimeError("curr_local_mix is not initialized")
        return self.curr_local_mix(curr_feature)

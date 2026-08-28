from __future__ import annotations

"""ST-Mamba-Lite block implementation for lightweight RAW fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            mamba_variant=config.mamba_variant,
            mamba2_state_dim=config.mamba2_state_dim,
            mamba2_headdim=config.mamba2_headdim,
            mamba2_groups=config.mamba2_groups,
            mamba2_chunk_size=config.mamba2_chunk_size,
            share_path_scan=config.stb_share_path_scan,
            fusion=config.stb_direction_fusion,
            dynamic_direction_fusion=config.dynamic_direction_fusion,
            scan_path_mode=config.scan_path_mode,
        )
        self.gated_ffn = GatedFFN(
            channels=config.channels,
            expand=config.gated_ffn_expand,
        )
        if config.temporal_motion_modulation:
            self.temporal_gate: nn.Conv2d | None = nn.Conv2d(config.channels, config.channels, kernel_size=1)
            nn.init.zeros_(self.temporal_gate.weight)
            nn.init.constant_(self.temporal_gate.bias, config.temporal_gate_bias_init)
        else:
            self.temporal_gate = None

    def forward(
        self,
        prev_feature: Tensor,
        curr_feature: Tensor,
        motion_feature: Tensor,
        static_reliability: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run strict implicit STB fusion."""
        _validate_feature_triplet(prev_feature, curr_feature, motion_feature, self.channels)
        if self.temporal_gate is not None:
            if static_reliability is None:
                raise ValueError("static_reliability is required when temporal_motion_modulation is enabled")
            if static_reliability.ndim != 4 or static_reliability.shape[1] != 1 or static_reliability.shape[0] != curr_feature.shape[0] or static_reliability.shape[-2:] != curr_feature.shape[-2:]:
                raise ValueError("static_reliability must be [B,1,H,W] aligned with the feature maps")

        local_prev_feature = self._mix_prev(prev_feature)
        local_curr_feature = self._mix_curr(curr_feature)

        if self.temporal_gate is None:
            temporal_reliability = None
            temporal_prev_feature = local_prev_feature
        else:
            learned_reliability = torch.sigmoid(self.temporal_gate(motion_feature))
            temporal_reliability = static_reliability * learned_reliability
            # In moving regions the previous-frame slot becomes the current
            # feature, so selective scan cannot import stale temporal detail.
            temporal_prev_feature = temporal_reliability * local_prev_feature + (1.0 - temporal_reliability) * local_curr_feature

        video_feature = torch.stack([temporal_prev_feature, local_curr_feature], dim=2)
        if self.config.scan_path_mode == "multiscale_grouped":
            # Keep local features at full packed resolution for small RAW
            # edges and chroma texture; run the expensive global scans at
            # half resolution, where each token covers a 2x2 Bayer cell.
            height, width = video_feature.shape[-2:]
            if height >= 2 and width >= 2:
                global_feature = F.avg_pool3d(
                    video_feature,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
                global_motion = F.avg_pool2d(motion_feature, kernel_size=2, stride=2)
                global_output = self.stb_scan(
                    global_feature,
                    direction_context=global_motion if self.config.dynamic_direction_fusion else None,
                )
                video_output = F.interpolate(
                    global_output,
                    size=(video_feature.shape[2], height, width),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                video_output = self.stb_scan(
                    video_feature,
                    direction_context=motion_feature if self.config.dynamic_direction_fusion else None,
                )
        else:
            video_output = self.stb_scan(
                video_feature,
                direction_context=motion_feature if self.config.dynamic_direction_fusion else None,
            )

        prev_temporal = video_output[:, :, 0]
        curr_temporal = video_output[:, :, 1]
        if self.config.scan_path_mode == "multiscale_grouped":
            # A direct local residual prevents half-resolution global context
            # from washing out fine edges and Bayer-plane chroma texture.
            prev_temporal = prev_temporal + 0.5 * local_prev_feature
            curr_temporal = curr_temporal + 0.5 * local_curr_feature
        curr_temporal = curr_temporal + self.gated_ffn(curr_temporal)
        if temporal_reliability is None:
            prev_output, curr_output = prev_temporal, curr_temporal
        else:
            prev_output = temporal_reliability * prev_temporal + (1.0 - temporal_reliability) * local_prev_feature
            curr_output = temporal_reliability * curr_temporal + (1.0 - temporal_reliability) * local_curr_feature

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

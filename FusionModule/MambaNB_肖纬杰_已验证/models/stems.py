from __future__ import annotations

"""Stem modules for the MambaFusionWeightNet-Lite backbone.

这里先把文档中的 `shared stem` 和与之相邻的 `motion stem` 单独拆出来，
后续接 `ST-Mamba-Lite block`、`refine head`、`weight head` 时可以直接复用。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .layers import ActivationType, ConvDepthwisePointwise


@dataclass(frozen=True)
class StemConfig:
    """Configuration for the packed RAW stem encoder."""

    in_channels: int = 4
    stem_channels: int = 24
    motion_hidden_channels: int | None = None
    shared_activation: ActivationType = "gelu"
    motion_activation: ActivationType = "silu"

    @property
    def motion_hidden(self) -> int:
        """Use `C/2` by default for the motion branch, matching the design doc."""
        return self.motion_hidden_channels or max(1, self.stem_channels // 2)


@dataclass(frozen=True)
class StemFeatures:
    """Structured outputs from the packed RAW stem encoder.

    Keeping these tensors in a dataclass makes the later backbone code easier to read
    than passing multiple unnamed tensors around.
    """

    prev_feature: Tensor
    curr_feature: Tensor
    motion_feature: Tensor
    prev_with_motion: Tensor
    curr_with_motion: Tensor
    temporal_stack: Tensor


def _validate_packed_raw(name: str, x: Tensor, expected_channels: int) -> None:
    """Check that the packed RAW feature map follows `[B, C, H, W]` layout."""
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(x.shape)}")
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"{name} channel mismatch: expected {expected_channels}, got {x.shape[1]}"
        )


def _validate_same_shape(name_a: str, a: Tensor, name_b: str, b: Tensor) -> None:
    """Ensure tensors that will be added or stacked stay spatially aligned."""
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must share the same shape, got {a.shape} vs {b.shape}")


class SharedRawStem(nn.Module):
    """Encode one packed Bayer frame into the common feature space.

    调用方式上对 `prev4` 和 `curr4` 复用同一个模块实例，即完成“共享权重”。
    """

    def __init__(self, config: StemConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConvDepthwisePointwise(
            in_channels=config.in_channels,
            hidden_channels=config.stem_channels,
            out_channels=config.stem_channels,
            activation=config.shared_activation,
        )

    def forward(self, packed_raw: Tensor) -> Tensor:
        _validate_packed_raw("packed_raw", packed_raw, expected_channels=self.config.in_channels)
        return self.encoder(packed_raw)


class MotionPriorStem(nn.Module):
    """Encode packed motion prior `diff4` into the same feature space as frame stem."""

    def __init__(self, config: StemConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConvDepthwisePointwise(
            in_channels=config.in_channels,
            hidden_channels=config.motion_hidden,
            out_channels=config.stem_channels,
            activation=config.motion_activation,
        )

    def forward(self, motion_prior: Tensor) -> Tensor:
        _validate_packed_raw("motion_prior", motion_prior, expected_channels=self.config.in_channels)
        return self.encoder(motion_prior)


class PackedRawStemEncoder(nn.Module):
    """Bundle the shared stem, motion stem and temporal stacking logic.

    这一步输出正好对应设计文档里的：
    - `f_prev = Stem(prev4)`
    - `f_curr = Stem(curr4)`
    - `f_diff = MotionStem(diff4)`
    - `x = stack([f_prev + f_diff, f_curr + f_diff], dim=T)`
    """

    def __init__(self, config: StemConfig | None = None) -> None:
        super().__init__()
        self.config = config or StemConfig()
        self.shared_stem = SharedRawStem(self.config)
        self.motion_stem = MotionPriorStem(self.config)

    def forward(self, prev4: Tensor, curr4: Tensor, motion_prior: Tensor) -> StemFeatures:
        _validate_packed_raw("prev4", prev4, expected_channels=self.config.in_channels)
        _validate_packed_raw("curr4", curr4, expected_channels=self.config.in_channels)
        _validate_packed_raw("motion_prior", motion_prior, expected_channels=self.config.in_channels)
        _validate_same_shape("prev4", prev4, "curr4", curr4)
        _validate_same_shape("prev4", prev4, "motion_prior", motion_prior)

        # 同一个 shared_stem 实例分别编码上一帧和当前帧，确保两帧严格共享参数。
        prev_feature = self.shared_stem(prev4)
        curr_feature = self.shared_stem(curr4)
        motion_feature = self.motion_stem(motion_prior)

        # 先做简单稳定的加法注入，再沿 temporal 维堆成 [B, C, T=2, H, W]。
        prev_with_motion = prev_feature + motion_feature
        curr_with_motion = curr_feature + motion_feature
        temporal_stack = torch.stack([prev_with_motion, curr_with_motion], dim=2)

        return StemFeatures(
            prev_feature=prev_feature,
            curr_feature=curr_feature,
            motion_feature=motion_feature,
            prev_with_motion=prev_with_motion,
            curr_with_motion=curr_with_motion,
            temporal_stack=temporal_stack,
        )

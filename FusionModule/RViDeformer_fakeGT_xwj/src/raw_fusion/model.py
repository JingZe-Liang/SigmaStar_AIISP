from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from raw_fusion.config import ModelConfig


@dataclass(frozen=True, slots=True)
class FusionOutput:
    prediction: torch.Tensor
    base: torch.Tensor
    gate: torch.Tensor
    correction: torch.Tensor


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels), channels)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.depthwise_norm = _normalization(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.pointwise_norm = _normalization(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        x = self.pointwise_norm(self.pointwise(x))
        return self.activation(x + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            _normalization(out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DepthwiseResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.downsample(x))


class FusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            _normalization(channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DepthwiseResidualBlock(channels)

    def forward(self, temporal: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(torch.cat((temporal, candidate), dim=1)))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            _normalization(out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DepthwiseResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(self.fuse(torch.cat((upsampled, skip), dim=1)))


class _Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _normalization(out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DepthwiseResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(x))


class _TemporalBranch(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        first, second, third = channels
        self.noisy_stem = _Stem(4, first)
        self.delta_stem = _Stem(4, first)
        self.project = nn.Sequential(
            nn.Conv2d(3 * first, first, 1),
            _normalization(first),
            nn.SiLU(inplace=True),
            DepthwiseResidualBlock(first),
        )
        self.down1 = DownBlock(first, second)
        self.down2 = DownBlock(second, third)

    def forward(
        self, prev_noisy: torch.Tensor, curr_noisy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous = self.noisy_stem(prev_noisy)
        current = self.noisy_stem(curr_noisy)
        delta = self.delta_stem(torch.abs(curr_noisy - prev_noisy))
        first = self.project(torch.cat((previous, current, delta), dim=1))
        second = self.down1(first)
        third = self.down2(second)
        return first, second, third


class _CandidateBranch(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        first, second, third = channels
        self.stem = _Stem(12, first)
        self.down1 = DownBlock(first, second)
        self.down2 = DownBlock(second, third)

    def forward(
        self, denoised: torch.Tensor, fused: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_input = torch.cat(
            (denoised, fused, torch.abs(denoised - fused)), dim=1
        )
        first = self.stem(candidate_input)
        second = self.down1(first)
        third = self.down2(second)
        return first, second, third


class CausalRawFusionNet(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        channels = model_config.channels
        first, second, third = channels
        self.residual_scale = model_config.residual_scale
        self.temporal_branch: _TemporalBranch | None
        if model_config.use_temporal:
            self.temporal_branch = _TemporalBranch(channels)
        else:
            self.temporal_branch = None
        self.candidate_branch = _CandidateBranch(channels)
        self.fusion_blocks = nn.ModuleList(FusionBlock(channel) for channel in channels)
        self.decode1 = DecoderBlock(third, second, second)
        self.decode2 = DecoderBlock(second, first, first)
        self.gate_head = nn.Sequential(DepthwiseResidualBlock(first), nn.Conv2d(first, 1, 1))
        self.correction_head = nn.Sequential(
            DepthwiseResidualBlock(first), nn.Conv2d(first, 4, 1)
        )
        self._zero_output_head(self.gate_head[-1])
        self._zero_output_head(self.correction_head[-1])

    @staticmethod
    def _zero_output_head(last_conv: nn.Module) -> None:
        if not isinstance(last_conv, nn.Conv2d):
            raise TypeError("output head must end in Conv2d")
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        self._validate_inputs(prev_noisy, curr_noisy, denoised, fused)
        candidate_features = self.candidate_branch(denoised, fused)
        if self.temporal_branch is None:
            temporal_features = tuple(torch.zeros_like(feature) for feature in candidate_features)
        else:
            temporal_features = self.temporal_branch(prev_noisy, curr_noisy)

        fused_features = tuple(
            block(temporal, candidate)
            for block, temporal, candidate in zip(
                self.fusion_blocks, temporal_features, candidate_features, strict=True
            )
        )
        decoded = self.decode1(fused_features[2], fused_features[1])
        decoded = self.decode2(decoded, fused_features[0])

        gate = torch.sigmoid(self.gate_head(decoded)).to(dtype=denoised.dtype)
        base = gate * denoised + (1.0 - gate) * fused
        correction = (
            self.residual_scale * torch.tanh(self.correction_head(decoded))
        ).to(dtype=denoised.dtype)
        prediction = base + correction
        return FusionOutput(prediction=prediction, base=base, gate=gate, correction=correction)

    def _validate_inputs(self, *inputs: torch.Tensor) -> None:
        if any(not isinstance(value, torch.Tensor) for value in inputs):
            raise TypeError("all four inputs must be torch.Tensor instances")
        reference_shape = inputs[0].shape
        if any(value.shape != reference_shape for value in inputs[1:]):
            raise ValueError("all four inputs must have the same shape")
        if len(reference_shape) != 4 or reference_shape[1] != 4:
            raise ValueError("all four inputs must have shape [B, 4, H, W]")
        if any(not value.is_floating_point() for value in inputs):
            raise TypeError("all four inputs must be floating-point tensors")
        reference_dtype = inputs[0].dtype
        if any(value.dtype != reference_dtype for value in inputs[1:]):
            raise ValueError("all four inputs must have the same dtype")
        reference_device = inputs[0].device
        if any(value.device != reference_device for value in inputs[1:]):
            raise ValueError("all four inputs must have the same device")
        model_parameter = next(self.parameters())
        if model_parameter.device != reference_device:
            raise ValueError("all four inputs must be on the model parameter device")
        if model_parameter.dtype != reference_dtype:
            if not torch.is_autocast_enabled(reference_device.type):
                raise TypeError("all four inputs must have the model parameter dtype")
            allowed_dtypes = {
                torch.float32,
                torch.get_autocast_dtype(reference_device.type),
            }
            if (
                reference_dtype not in allowed_dtypes
                or model_parameter.dtype not in allowed_dtypes
            ):
                raise TypeError(
                    "autocast input and model parameter dtypes must be float32 or the autocast dtype"
                )

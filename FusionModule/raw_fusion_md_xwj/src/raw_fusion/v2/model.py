"""Shared, noise-conditioned logits core for V2 frequency fusion.

The learned module deliberately ends at four alpha-class logits.  Motion
protection, selector reduction, B2, and composition stay outside this module
so the same state dict is valid for both noise conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schemas.common import ContractError


CORE_INPUT_NAMES: Final[tuple[str, ...]] = (
    "prev_noisy",
    "curr_noisy",
    "denoised",
    "fused",
    "c_tilde",
)
CORE_HALO_PACKED: Final[int] = 32
PRODUCTION_HALO_PACKED: Final[int] = 32


@dataclass(frozen=True, slots=True)
class FrequencyFusionConfigV2:
    """The frozen architecture used by both 128x and 645x streams."""

    channels: tuple[int, int, int] = (24, 48, 72)
    channel_norm_epsilon: float = 1e-5
    condition_dim: int = 4
    condition_hidden: int = 32
    q_classes: tuple[float, float, float, float] = (0.0, 0.125, 0.25, 0.5)
    q_head_bias: tuple[float, float, float, float] = (4.575, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if tuple(self.channels) != (24, 48, 72):
            raise ContractError("FrequencyFusionConfigV2.channels must be [24,48,72]")
        if not math.isfinite(float(self.channel_norm_epsilon)) or float(self.channel_norm_epsilon) != 1e-5:
            raise ContractError("FrequencyFusionConfigV2.channel_norm_epsilon must be 1e-5")
        if self.condition_dim != 4 or self.condition_hidden != 32:
            raise ContractError("FrequencyFusionConfigV2 condition MLP must be 4->32")
        if tuple(self.q_classes) != (0.0, 0.125, 0.25, 0.5):
            raise ContractError("FrequencyFusionConfigV2.q_classes are fixed")
        if tuple(float(value) for value in self.q_head_bias) != (4.575, 0.0, 0.0, 0.0):
            raise ContractError("FrequencyFusionConfigV2.q_head_bias is fixed")

    @classmethod
    def production(cls) -> "FrequencyFusionConfigV2":
        return cls()


@dataclass(frozen=True, slots=True)
class CoreOutputV2:
    """Only the tile's valid, 32-pixel-cropped logits are observable."""

    q_logits_pixel_core: Tensor


class ChannelNorm(nn.Module):
    """Affine per-pixel channel normalization with no spatial statistics."""

    def __init__(self, channels: int, *, epsilon: float = 1e-5) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ContractError("ChannelNorm channels must be a positive integer")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ContractError("ChannelNorm epsilon must be finite and positive")
        self.channels = channels
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != self.channels:
            raise ContractError("ChannelNorm expects [B,C,H,W] with its configured channel count")
        if not value.is_floating_point():
            raise ContractError("ChannelNorm requires floating-point input")
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class _SpatialConvNormAct(nn.Module):
    """Explicit reflect padding keeps convolution support auditable."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int, epsilon: float, groups: int = 1) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, groups=groups)
        self.norm = ChannelNorm(out_channels, epsilon=epsilon)
        self.activation = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(self.pad(value))))


class _PointwiseNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, epsilon: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = ChannelNorm(out_channels, epsilon=epsilon)
        self.activation = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(value)))


class _DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, *, epsilon: float) -> None:
        super().__init__()
        self.depthwise = _SpatialConvNormAct(channels, channels, stride=1, epsilon=epsilon, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.pointwise_norm = ChannelNorm(channels, epsilon=epsilon)
        self.activation = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.pointwise_norm(self.pointwise(value))
        return self.activation(value + residual)


class _Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, epsilon: float) -> None:
        super().__init__()
        self.project = _SpatialConvNormAct(in_channels, out_channels, stride=1, epsilon=epsilon)
        self.refine = _DepthwiseResidualBlock(out_channels, epsilon=epsilon)

    def forward(self, value: Tensor) -> Tensor:
        return self.refine(self.project(value))


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, epsilon: float) -> None:
        super().__init__()
        self.downsample = _SpatialConvNormAct(in_channels, out_channels, stride=2, epsilon=epsilon)
        self.refine = _DepthwiseResidualBlock(out_channels, epsilon=epsilon)

    def forward(self, value: Tensor) -> Tensor:
        return self.refine(self.downsample(value))


class _FusionBlock(nn.Module):
    def __init__(self, channels: int, *, epsilon: float) -> None:
        super().__init__()
        self.project = _PointwiseNormAct(2 * channels, channels, epsilon=epsilon)
        self.refine = _DepthwiseResidualBlock(channels, epsilon=epsilon)

    def forward(self, temporal: Tensor, candidate: Tensor) -> Tensor:
        return self.refine(self.project(torch.cat((temporal, candidate), dim=1)))


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, epsilon: float) -> None:
        super().__init__()
        self.fuse = _SpatialConvNormAct(in_channels + skip_channels, out_channels, stride=1, epsilon=epsilon)
        self.refine = _DepthwiseResidualBlock(out_channels, epsilon=epsilon)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        upsampled = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.fuse(torch.cat((upsampled, skip), dim=1)))


class _TemporalBranch(nn.Module):
    def __init__(self, channels: tuple[int, int, int], *, epsilon: float) -> None:
        super().__init__()
        first, second, third = channels
        self.noisy_stem = _Stem(4, first, epsilon=epsilon)
        self.delta_stem = _Stem(4, first, epsilon=epsilon)
        self.project = _PointwiseNormAct(3 * first, first, epsilon=epsilon)
        self.refine = _DepthwiseResidualBlock(first, epsilon=epsilon)
        self.down1 = _DownBlock(first, second, epsilon=epsilon)
        self.down2 = _DownBlock(second, third, epsilon=epsilon)

    def forward(self, prev_noisy: Tensor, curr_noisy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        previous = self.noisy_stem(prev_noisy)
        current = self.noisy_stem(curr_noisy)
        delta = self.delta_stem(torch.abs(curr_noisy - prev_noisy))
        first = self.refine(self.project(torch.cat((previous, current, delta), dim=1)))
        second = self.down1(first)
        third = self.down2(second)
        return first, second, third


class _CandidateBranch(nn.Module):
    def __init__(self, channels: tuple[int, int, int], *, epsilon: float) -> None:
        super().__init__()
        first, second, third = channels
        self.stem = _Stem(12, first, epsilon=epsilon)
        self.down1 = _DownBlock(first, second, epsilon=epsilon)
        self.down2 = _DownBlock(second, third, epsilon=epsilon)

    def forward(self, denoised: Tensor, fused: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        candidate_input = torch.cat((denoised, fused, torch.abs(denoised - fused)), dim=1)
        first = self.stem(candidate_input)
        second = self.down1(first)
        third = self.down2(second)
        return first, second, third


class FrequencyFusionCore(nn.Module):
    """One shared 128x/645x core with a single continuous FiLM injection."""

    def __init__(self, config: FrequencyFusionConfigV2) -> None:
        super().__init__()
        if not isinstance(config, FrequencyFusionConfigV2):
            raise TypeError("FrequencyFusionCore requires FrequencyFusionConfigV2")
        self.config = config
        self.halo_packed = CORE_HALO_PACKED
        first, second, third = config.channels
        epsilon = config.channel_norm_epsilon
        self.temporal_branch = _TemporalBranch(config.channels, epsilon=epsilon)
        self.candidate_branch = _CandidateBranch(config.channels, epsilon=epsilon)
        self.fusion_blocks = nn.ModuleList(
            (_FusionBlock(first, epsilon=epsilon), _FusionBlock(second, epsilon=epsilon), _FusionBlock(third, epsilon=epsilon))
        )
        self.decode1 = _DecoderBlock(third, second, second, epsilon=epsilon)
        self.decode2 = _DecoderBlock(second, first, first, epsilon=epsilon)
        self.film_in = nn.Linear(config.condition_dim, config.condition_hidden)
        self.film_activation = nn.SiLU()
        # 24 gamma and 24 beta values for the 24-channel final decoder feature.
        self.film_out = nn.Linear(config.condition_hidden, 2 * first)
        self.q_head = nn.Conv2d(first, len(config.q_classes), kernel_size=1)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)
        nn.init.zeros_(self.q_head.weight)
        with torch.no_grad():
            self.q_head.bias.copy_(torch.tensor(config.q_head_bias, dtype=self.q_head.bias.dtype))
        # A snapshot makes later unregistered spatial edits fail closed.
        self._registered_spatial_module_names = frozenset(
            name
            for name, module in self.named_modules()
            if isinstance(module, (nn.Conv2d, nn.ReflectionPad2d))
        )

    def forward(
        self,
        prev_noisy: Tensor,
        curr_noisy: Tensor,
        denoised: Tensor,
        fused: Tensor,
        c_tilde: Tensor,
        *,
        condition_enabled: bool = True,
    ) -> CoreOutputV2:
        if not isinstance(condition_enabled, bool):
            raise TypeError("FrequencyFusionCore.condition_enabled must be bool")
        self._validate_inputs(prev_noisy, curr_noisy, denoised, fused, c_tilde)
        temporal_features = self.temporal_branch(prev_noisy, curr_noisy)
        candidate_features = self.candidate_branch(denoised, fused)
        fused_features = tuple(
            block(temporal, candidate)
            for block, temporal, candidate in zip(self.fusion_blocks, temporal_features, candidate_features, strict=True)
        )
        decoded = self.decode1(fused_features[2], fused_features[1])
        decoded = self.decode2(decoded, fused_features[0])
        if condition_enabled:
            film = self.film_out(self.film_activation(self.film_in(c_tilde)))
            gamma, beta = film.chunk(2, dim=1)
            conditioned = decoded * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        else:
            # The no-condition ablation must leave FiLM outside the graph,
            # rather than merely feed it a neutral vector.
            conditioned = decoded
        logits = self.q_head(conditioned)
        halo = self.halo_packed
        return CoreOutputV2(q_logits_pixel_core=logits[..., halo:-halo, halo:-halo])

    def forward_data_first(self, inputs: "object") -> CoreOutputV2:
        """Run the core through the exact data-first input boundary."""
        from .data_first_contracts import DataFirstInputBatch

        if not isinstance(inputs, DataFirstInputBatch):
            raise TypeError("forward_data_first requires DataFirstInputBatch")
        return self(
            **inputs.as_mapping(),
            condition_enabled=True,
        )

    def _validate_inputs(self, prev_noisy: Tensor, curr_noisy: Tensor, denoised: Tensor, fused: Tensor, c_tilde: Tensor) -> None:
        image_inputs = (prev_noisy, curr_noisy, denoised, fused)
        if any(not isinstance(value, Tensor) for value in image_inputs) or not isinstance(c_tilde, Tensor):
            raise TypeError("FrequencyFusionCore inputs must be torch tensors")
        reference = image_inputs[0]
        if reference.ndim != 4 or reference.shape[1] != 4:
            raise ContractError("FrequencyFusionCore image inputs must have shape [B,4,H,W]")
        if reference.shape[-2] <= 2 * self.halo_packed or reference.shape[-1] <= 2 * self.halo_packed:
            raise ContractError("FrequencyFusionCore only accepts inputs larger than its 32-pixel halo")
        if any(value.shape != reference.shape for value in image_inputs[1:]):
            raise ContractError("FrequencyFusionCore image inputs must have identical shapes")
        if any(not value.is_floating_point() for value in image_inputs):
            raise ContractError("FrequencyFusionCore image inputs must be floating point")
        if any(value.dtype != reference.dtype or value.device != reference.device for value in image_inputs[1:]):
            raise ContractError("FrequencyFusionCore image inputs must share dtype and device")
        if c_tilde.ndim != 2 or c_tilde.shape != (reference.shape[0], self.config.condition_dim):
            raise ContractError("FrequencyFusionCore.c_tilde must have shape [B,4]")
        if not c_tilde.is_floating_point() or c_tilde.dtype != reference.dtype or c_tilde.device != reference.device:
            raise ContractError("FrequencyFusionCore.c_tilde must share image dtype and device")
        parameter = next(self.parameters())
        if parameter.device != reference.device or parameter.dtype != reference.dtype:
            raise ContractError("FrequencyFusionCore inputs must match model parameter dtype and device")


@dataclass(frozen=True, slots=True)
class _ReceptiveField:
    radius: int
    jump: int


def _max_field(*fields: _ReceptiveField) -> _ReceptiveField:
    if not fields:
        raise AssertionError("at least one receptive field is required")
    jump = fields[0].jump
    if any(field.jump != jump for field in fields):
        raise ContractError("dependency radius trace joined incompatible spatial scales")
    return _ReceptiveField(radius=max(field.radius for field in fields), jump=jump)


def _trace_conv(field: _ReceptiveField, conv: nn.Conv2d) -> _ReceptiveField:
    kernel = conv.kernel_size
    dilation = conv.dilation
    stride = conv.stride
    if kernel[0] != kernel[1] or dilation[0] != dilation[1] or stride[0] != stride[1]:
        raise ContractError("dependency radius requires square registered convolutions")
    if kernel[0] not in (1, 3) or dilation[0] != 1 or stride[0] not in (1, 2):
        raise ContractError("dependency radius rejected an unsupported registered convolution")
    radius = field.radius + ((kernel[0] - 1) // 2) * field.jump
    return _ReceptiveField(radius=radius, jump=field.jump * stride[0])


def _trace_spatial(field: _ReceptiveField, block: _SpatialConvNormAct) -> _ReceptiveField:
    if not isinstance(block.pad, nn.ReflectionPad2d) or block.pad.padding != (1, 1, 1, 1):
        raise ContractError("dependency radius requires registered one-pixel reflect padding")
    return _trace_conv(field, block.conv)


def _trace_residual(field: _ReceptiveField, block: _DepthwiseResidualBlock) -> _ReceptiveField:
    transformed = _trace_spatial(field, block.depthwise)
    transformed = _trace_conv(transformed, block.pointwise)
    return _max_field(field, transformed)


def _trace_stem(field: _ReceptiveField, stem: _Stem) -> _ReceptiveField:
    return _trace_residual(_trace_spatial(field, stem.project), stem.refine)


def _trace_down(field: _ReceptiveField, block: _DownBlock) -> _ReceptiveField:
    return _trace_residual(_trace_spatial(field, block.downsample), block.refine)


def _trace_fusion(field: _ReceptiveField, block: _FusionBlock) -> _ReceptiveField:
    return _trace_residual(_trace_conv(field, block.project.conv), block.refine)


def _trace_upsample(field: _ReceptiveField) -> _ReceptiveField:
    if field.jump % 2:
        raise ContractError("dependency radius cannot trace a non-dyadic upsample")
    # Bilinear interpolation reads adjacent coarse-grid values.  Its support
    # bound is one full incoming-grid spacing, not half: align_corners=False
    # can choose the neighbor on either side of an output sample.
    return _ReceptiveField(radius=field.radius + field.jump, jump=field.jump // 2)


def _trace_decoder(field: _ReceptiveField, skip: _ReceptiveField, block: _DecoderBlock) -> _ReceptiveField:
    upsampled = _trace_upsample(field)
    joined = _max_field(upsampled, skip)
    return _trace_residual(_trace_spatial(joined, block.fuse), block.refine)


def _registered_spatial_names(core: FrequencyFusionCore) -> set[str]:
    return {
        name
        for name, module in core.named_modules()
        if isinstance(module, (nn.Conv2d, nn.ReflectionPad2d))
    }


def validate_core_architecture(core: FrequencyFusionCore) -> None:
    """Reject gain-specific, spatially nonlocal, or non-logit architecture edits."""
    if not isinstance(core, FrequencyFusionCore):
        raise ContractError("core must be a FrequencyFusionCore")
    if core.config != FrequencyFusionConfigV2.production():
        raise ContractError("FrequencyFusionCore configuration differs from the production contract")
    for name, module in core.named_modules():
        lower_name = name.lower()
        if any(token in lower_name for token in ("gain", "embedding", "expert")):
            raise ContractError("gain-specific modules and branches are forbidden")
        if name and ("gate" in lower_name or "correction" in lower_name or name.endswith("_head") and name != "q_head"):
            raise ContractError("FrequencyFusionCore exposes q logits only")
        if isinstance(module, (nn.GroupNorm, nn.modules.batchnorm._BatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)):
            raise ContractError("spatial or batch normalization is forbidden; use ChannelNorm")
        if not name:
            continue
        if any(True for _ in module.children()):
            continue
        if not isinstance(module, (nn.Conv2d, nn.ReflectionPad2d, nn.Linear, nn.SiLU, ChannelNorm)):
            raise ContractError(f"dependency radius cannot register spatial operator {name}")
    for name, _parameter in core.named_parameters():
        lower_name = name.lower()
        if any(token in lower_name for token in ("gain", "embedding", "expert")):
            raise ContractError("gain-specific parameters are forbidden")
    if not isinstance(core.film_in, nn.Linear) or core.film_in.in_features != 4 or core.film_in.out_features != 32:
        raise ContractError("FrequencyFusionCore requires one 4->32 condition MLP")
    if not isinstance(core.film_out, nn.Linear) or core.film_out.in_features != 32 or core.film_out.out_features != 48:
        raise ContractError("FrequencyFusionCore requires one 32->48 FiLM output")
    linear_names = {name for name, module in core.named_modules() if isinstance(module, nn.Linear)}
    if linear_names != {"film_in", "film_out"}:
        raise ContractError("FrequencyFusionCore permits only the shared FiLM MLP")
    if not isinstance(core.q_head, nn.Conv2d) or core.q_head.kernel_size != (1, 1) or core.q_head.in_channels != 24 or core.q_head.out_channels != 4:
        raise ContractError("FrequencyFusionCore requires exactly one four-class q-logit head")
    if _registered_spatial_names(core) != set(core._registered_spatial_module_names):
        raise ContractError("dependency radius registration is incomplete")


def measure_dependency_radius(core: FrequencyFusionCore, input_shape: tuple[int, int]) -> int:
    """Trace the fixed multiscale graph and return its packed-pixel radius.

    The trace is deliberately structural rather than weight-dependent, so a
    zero-initialized q head cannot hide a newly added spatial dependency.
    """
    if not isinstance(input_shape, tuple) or len(input_shape) != 2:
        raise ContractError("dependency radius input_shape must be (height,width)")
    height, width = input_shape
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 2 * CORE_HALO_PACKED for value in (height, width)):
        raise ContractError("dependency radius input shape must leave a valid 32-pixel core")
    validate_core_architecture(core)
    base = _ReceptiveField(radius=0, jump=1)
    temporal = core.temporal_branch
    previous = _trace_stem(base, temporal.noisy_stem)
    current = _trace_stem(base, temporal.noisy_stem)
    delta = _trace_stem(base, temporal.delta_stem)
    temporal_first = _trace_residual(_trace_conv(_max_field(previous, current, delta), temporal.project.conv), temporal.refine)
    temporal_second = _trace_down(temporal_first, temporal.down1)
    temporal_third = _trace_down(temporal_second, temporal.down2)
    candidate = core.candidate_branch
    candidate_first = _trace_stem(base, candidate.stem)
    candidate_second = _trace_down(candidate_first, candidate.down1)
    candidate_third = _trace_down(candidate_second, candidate.down2)
    fusion_first = _trace_fusion(_max_field(temporal_first, candidate_first), core.fusion_blocks[0])
    fusion_second = _trace_fusion(_max_field(temporal_second, candidate_second), core.fusion_blocks[1])
    fusion_third = _trace_fusion(_max_field(temporal_third, candidate_third), core.fusion_blocks[2])
    decoded_second = _trace_decoder(fusion_third, fusion_second, core.decode1)
    decoded_first = _trace_decoder(decoded_second, fusion_first, core.decode2)
    final = _trace_conv(decoded_first, core.q_head)
    if final.radius > CORE_HALO_PACKED:
        raise ContractError("dependency radius exceeds the 32-pixel production halo")
    return final.radius

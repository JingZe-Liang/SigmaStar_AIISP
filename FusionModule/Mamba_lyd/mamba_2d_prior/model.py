"""2DNR-prior ST-Mamba fusion architecture.

The network predicts only a bounded 3DNR correction. It cannot replace 2DNR
in motion or texture regions, and it never synthesizes a RAW image directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from torch import Tensor, nn
import torch

from Mamba.models.layers import ConvDepthwisePointwise
from Mamba.models.stmamba_config import STMambaLiteConfig
from Mamba.models.stmamba_stack import STMambaLiteStack


@dataclass(frozen=True)
class Mamba2DPriorConfig:
    cfa_pattern: str = "RGGB"
    channels: int = 24
    num_blocks: int = 2
    mamba_state_dim: int = 8
    mamba_expand: int = 2
    mamba_scan_backend: str = "auto"
    mamba_variant: str = "mamba1"
    # First-stage efficient scan: temporal-first H/W traversal, both ways.
    scan_path_mode: str = "multiscale_grouped"
    mamba2_state_dim: int = 64
    mamba2_headdim: int | None = None
    mamba2_groups: int = 1
    mamba2_chunk_size: int = 256
    max_3dnr_weight: float = 0.35
    beta_bias_init: float = -4.0
    block_motion_modulation: bool = True
    dynamic_direction_fusion: bool = True
    temporal_gate_bias_init: float = 4.0
    backbone: STMambaLiteConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cfa_pattern != "RGGB":
            raise ValueError("This SigmaStar adapter expects physical RGGB packing")
        if not 0.0 < self.max_3dnr_weight <= 1.0:
            raise ValueError("max_3dnr_weight must be in (0, 1]")
        if self.mamba_variant not in {"mamba1", "mamba2"}:
            raise ValueError("mamba_variant must be 'mamba1' or 'mamba2'")
        object.__setattr__(self, "backbone", STMambaLiteConfig(
            channels=self.channels,
            num_blocks=self.num_blocks,
            mamba_state_dim=self.mamba_state_dim,
            mamba_expand=self.mamba_expand,
            mamba_scan_backend=self.mamba_scan_backend,  # type: ignore[arg-type]
            mamba_variant=self.mamba_variant,  # type: ignore[arg-type]
            scan_path_mode=self.scan_path_mode,  # type: ignore[arg-type]
            mamba2_state_dim=self.mamba2_state_dim,
            mamba2_headdim=self.mamba2_headdim,
            mamba2_groups=self.mamba2_groups,
            mamba2_chunk_size=self.mamba2_chunk_size,
            temporal_motion_modulation=self.block_motion_modulation,
            dynamic_direction_fusion=self.dynamic_direction_fusion,
            temporal_gate_bias_init=self.temporal_gate_bias_init,
        ))

    def serializable(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "backbone"}


@dataclass
class Mamba2DPriorOutput:
    prediction: Tensor
    beta: Tensor
    weight_3d: Tensor
    motion: Tensor
    flatness: Tensor
    agreement: Tensor


class Mamba2DPriorFusionNet(nn.Module):
    """ST-Mamba weight predictor with hard 2DNR-prior constraints."""

    def __init__(self, config: Mamba2DPriorConfig | None = None) -> None:
        super().__init__()
        self.config = config or Mamba2DPriorConfig()
        channels = self.config.channels
        self.source_stem = ConvDepthwisePointwise(4, channels, channels, "gelu")
        # Plane-specific motion, flatness and 2DNR/3DNR agreement. Do not
        # average these gates before beta is predicted: each Bayer color has
        # different temporal noise and 3DNR behaviour.
        self.motion_stem = ConvDepthwisePointwise(12, max(1, channels // 2), channels, "silu")
        # current source, 2DNR, 3DNR, and 3DNR-2DNR residual: 16 channels.
        self.candidate_stem = ConvDepthwisePointwise(16, channels, channels, "silu")
        self.backbone = STMambaLiteStack(config=self.config.backbone)
        self.refine = nn.Sequential(
            nn.GroupNorm(1, channels * 3),
            nn.Conv2d(channels * 3, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(inplace=True),
        )
        self.beta_head = nn.Conv2d(channels, 4, 1)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, self.config.beta_bias_init)

    def forward(self, prev4: Tensor, curr4: Tensor, dnr2_4: Tensor, dnr3_4: Tensor, motion: Tensor, flatness: Tensor, agreement: Tensor) -> Mamba2DPriorOutput:
        self._validate(prev4, curr4, dnr2_4, dnr3_4, motion, flatness, agreement)
        motion_input = torch.cat((motion, flatness, agreement), dim=1)
        motion_feature = self.motion_stem(motion_input)
        source_prev = self.source_stem(prev4) + motion_feature
        source_curr = self.source_stem(curr4) + motion_feature
        # A strict plane-aware motion prior is supplied to every block. The
        # block can learn a channel-wise refinement but cannot treat a clearly
        # moving pixel as fully temporally reliable.
        static_reliability = 1.0 - motion.amax(dim=1, keepdim=True)
        temporal = self.backbone(source_prev, source_curr, motion_feature, static_reliability).curr_feature
        residual = dnr3_4 - dnr2_4
        candidate = self.candidate_stem(torch.cat((curr4, dnr2_4, dnr3_4, residual), dim=1))
        beta = torch.sigmoid(self.beta_head(self.refine(torch.cat((temporal, candidate, motion_feature), dim=1))))
        weight_3d = self.config.max_3dnr_weight * beta * (1.0 - motion) * flatness * agreement
        prediction = dnr2_4 + weight_3d * residual
        return Mamba2DPriorOutput(prediction=prediction, beta=beta, weight_3d=weight_3d, motion=motion, flatness=flatness, agreement=agreement)

    @staticmethod
    def _validate(prev4: Tensor, curr4: Tensor, dnr2_4: Tensor, dnr3_4: Tensor, motion: Tensor, flatness: Tensor, agreement: Tensor) -> None:
        candidates = {"prev4": prev4, "curr4": curr4, "dnr2_4": dnr2_4, "dnr3_4": dnr3_4, "agreement": agreement}
        for name, value in candidates.items():
            if value.ndim != 4 or value.shape[1] != 4:
                raise ValueError(f"{name} must be [B,4,H,W], got {tuple(value.shape)}")
        if any(value.shape != curr4.shape for value in candidates.values()):
            raise ValueError("All packed candidate inputs must have identical shapes")
        for name, value in {"motion": motion, "flatness": flatness}.items():
            if value.ndim != 4 or value.shape != curr4.shape:
                raise ValueError(f"{name} must be [B,4,H,W] aligned with curr4, got {tuple(value.shape)}")

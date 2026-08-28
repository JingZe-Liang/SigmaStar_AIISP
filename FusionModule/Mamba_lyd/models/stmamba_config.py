from __future__ import annotations

"""Configuration helpers for the ST-Mamba-Lite backbone blocks."""

from dataclasses import dataclass
from typing import Literal


ScanBackend = Literal["auto", "reference", "mamba_ssm"]
MambaVariant = Literal["mamba1", "mamba2"]
ScanPathMode = Literal["8path", "temporal4", "temporal4_grouped", "multiscale_grouped"]


@dataclass(frozen=True)
class STMambaLiteConfig:
    """Shared configuration for ST-Mamba-Lite blocks and block stacks."""

    channels: int = 24
    num_blocks: int = 2
    share_local_mix: bool = True
    stb_share_path_scan: bool = False
    stb_direction_fusion: Literal["softmax"] = "softmax"
    mamba_state_dim: int = 8
    mamba_expand: int = 2
    mamba_dt_rank: int | None = None
    mamba_conv_kernel: int = 3
    mamba_scan_backend: ScanBackend = "auto"
    mamba_variant: MambaVariant = "mamba1"
    # Keep the legacy backbone default intact. The 2DNR-prior adapter selects
    # a lower-cost temporal-first mode explicitly to avoid redundant traversals.
    scan_path_mode: ScanPathMode = "8path"
    mamba2_state_dim: int = 64
    mamba2_headdim: int | None = None
    mamba2_groups: int = 1
    mamba2_chunk_size: int = 256
    gated_ffn_expand: int = 2
    # Disabled by default to preserve the original H5 model behaviour. The
    # SigmaStar 2DNR-prior adapter enables both explicitly.
    temporal_motion_modulation: bool = False
    dynamic_direction_fusion: bool = False
    temporal_gate_bias_init: float = 4.0

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.stb_direction_fusion != "softmax":
            raise ValueError(f"Unsupported stb_direction_fusion: {self.stb_direction_fusion}")
        if self.mamba_state_dim <= 0:
            raise ValueError(f"mamba_state_dim must be positive, got {self.mamba_state_dim}")
        if self.mamba_expand <= 0:
            raise ValueError(f"mamba_expand must be positive, got {self.mamba_expand}")
        if self.mamba_dt_rank is not None and self.mamba_dt_rank <= 0:
            raise ValueError(f"mamba_dt_rank must be positive, got {self.mamba_dt_rank}")
        if self.mamba_conv_kernel <= 0:
            raise ValueError(f"mamba_conv_kernel must be positive, got {self.mamba_conv_kernel}")
        if self.mamba_scan_backend not in {"auto", "reference", "mamba_ssm"}:
            raise ValueError(f"Unsupported mamba_scan_backend: {self.mamba_scan_backend}")
        if self.mamba_variant not in {"mamba1", "mamba2"}:
            raise ValueError(f"Unsupported mamba_variant: {self.mamba_variant}")
        if self.scan_path_mode not in {"8path", "temporal4", "temporal4_grouped", "multiscale_grouped"}:
            raise ValueError(f"Unsupported scan_path_mode: {self.scan_path_mode}")
        if self.mamba2_state_dim <= 0 or self.mamba2_groups <= 0 or self.mamba2_chunk_size <= 0:
            raise ValueError("Mamba-2 state_dim, groups and chunk_size must be positive")
        if self.mamba2_headdim is not None and self.mamba2_headdim <= 0:
            raise ValueError("mamba2_headdim must be positive when provided")
        if self.gated_ffn_expand <= 0:
            raise ValueError(f"gated_ffn_expand must be positive, got {self.gated_ffn_expand}")

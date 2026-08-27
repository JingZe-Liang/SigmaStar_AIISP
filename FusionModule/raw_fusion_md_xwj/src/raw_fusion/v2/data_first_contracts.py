"""Input and provenance contracts for the data-first V2 protocol."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import torch
from torch import Tensor

from .schemas.common import ContractError


DATA_FIRST_PROTOCOL: Final[str] = "raw_fusion_v2_data_first"
DATA_FIRST_IMAGE_KEYS: Final[tuple[str, ...]] = ("prev_noisy", "curr_noisy", "denoised", "fused")
DATA_FIRST_INPUT_KEYS: Final[tuple[str, ...]] = (*DATA_FIRST_IMAGE_KEYS, "c_tilde")


def _validate_images(images: tuple[Tensor, ...]) -> None:
    if any(not isinstance(value, Tensor) for value in images):
        raise TypeError("data-first image inputs must be torch tensors")
    reference = images[0]
    if reference.ndim != 4 or reference.shape[1] != 4:
        raise ContractError("data-first image inputs must have shape [B,4,H,W]")
    if any(value.shape != reference.shape for value in images[1:]):
        raise ContractError("data-first image inputs must have identical shapes")
    if any(not value.is_floating_point() for value in images):
        raise ContractError("data-first image inputs must be floating point")
    if any(value.dtype != reference.dtype or value.device != reference.device for value in images[1:]):
        raise ContractError("data-first image inputs must share dtype and device")
    if not bool(torch.isfinite(torch.stack([value.detach().float().mean() for value in images])).all()):
        raise ContractError("data-first image inputs must contain finite values")


def derive_input_condition(
    *,
    prev_noisy: Tensor,
    curr_noisy: Tensor,
    denoised: Tensor,
    fused: Tensor,
) -> Tensor:
    """Derive a four-value condition from deployable image inputs only.

    The values are intentionally simple and stable: temporal noisy change,
    candidate change, denoised level, and fused level, each reduced per CFA
    channel.  No MD or supervision object is accepted by this function.
    """
    images = (prev_noisy, curr_noisy, denoised, fused)
    _validate_images(images)
    values = torch.stack(
        (
            (curr_noisy - prev_noisy).abs().mean(dim=(2, 3)).mean(dim=1),
            (fused - denoised).abs().mean(dim=(2, 3)).mean(dim=1),
            denoised.mean(dim=(2, 3)).mean(dim=1),
            fused.mean(dim=(2, 3)).mean(dim=1),
        ),
        dim=1,
    )
    # Normalize each batch row without introducing a learned or MD-dependent
    # statistic.  The scale is fixed so the result is reproducible across runs.
    centered = values - values.mean(dim=1, keepdim=True)
    scale = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    result = centered / scale
    if not bool(torch.isfinite(result).all()):
        raise ContractError("data-first input condition must be finite")
    return result.to(dtype=prev_noisy.dtype)


@dataclass(frozen=True, slots=True)
class DataFirstInputBatch:
    """Exact model input boundary; supervision fields cannot be attached."""

    prev_noisy: Tensor
    curr_noisy: Tensor
    denoised: Tensor
    fused: Tensor
    c_tilde: Tensor

    def __post_init__(self) -> None:
        _validate_images((self.prev_noisy, self.curr_noisy, self.denoised, self.fused))
        if not isinstance(self.c_tilde, Tensor):
            raise TypeError("data-first c_tilde must be a torch tensor")
        if self.c_tilde.shape != (self.prev_noisy.shape[0], 4):
            raise ContractError("data-first c_tilde must have shape [B,4]")
        if not self.c_tilde.is_floating_point() or self.c_tilde.dtype != self.prev_noisy.dtype:
            raise ContractError("data-first c_tilde must share image dtype")
        if self.c_tilde.device != self.prev_noisy.device or not bool(torch.isfinite(self.c_tilde).all()):
            raise ContractError("data-first c_tilde must share device and contain finite values")

    def as_mapping(self) -> MappingProxyType:
        return MappingProxyType({name: getattr(self, name) for name in DATA_FIRST_INPUT_KEYS})

    @classmethod
    def from_mapping(cls, mapping: object) -> "DataFirstInputBatch":
        if not isinstance(mapping, dict) and not hasattr(mapping, "keys"):
            raise TypeError("data-first input mapping must be a mapping")
        keys = tuple(mapping.keys())  # type: ignore[union-attr]
        if keys != DATA_FIRST_INPUT_KEYS or set(keys) != set(DATA_FIRST_INPUT_KEYS):
            raise TypeError("data-first input mapping must contain exactly model input keys")
        return cls(**{name: mapping[name] for name in DATA_FIRST_INPUT_KEYS})  # type: ignore[index]


__all__ = [
    "DATA_FIRST_INPUT_KEYS",
    "DATA_FIRST_PROTOCOL",
    "DataFirstInputBatch",
    "derive_input_condition",
]

"""Standalone Bayer RAW helpers used by the original and 2DNR-prior models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RawRange:
    black_level: float = 0.0
    white_level: float = 4095.0


DEFAULT_RAW_RANGE = RawRange()


@dataclass(frozen=True)
class RawPairFeatures:
    prev_packed: Tensor
    curr_packed: Tensor
    motion_prior: Tensor


def get_cfa_positions(cfa_pattern: str) -> tuple[int, int, int, int]:
    pattern = cfa_pattern.upper()
    if pattern not in {"RGGB", "BGGR", "GBRG", "GRBG"}:
        raise ValueError(f"Unsupported Bayer pattern: {cfa_pattern}")
    green_positions = [index for index, channel in enumerate(pattern) if channel == "G"]
    return pattern.index("R"), green_positions[0], green_positions[1], pattern.index("B")


def normalize_raw(raw: Tensor, raw_range: RawRange = DEFAULT_RAW_RANGE) -> Tensor:
    if raw_range.white_level <= raw_range.black_level:
        raise ValueError("white_level must exceed black_level")
    return ((raw.float() - raw_range.black_level) / (raw_range.white_level - raw_range.black_level)).clamp(0.0, 1.0)


def bayer_pack(raw: Tensor, cfa_pattern: str = "RGGB") -> Tensor:
    """Pack [B,H,W] or [B,1,H,W] RAW into physical [R,G1,G2,B] planes."""
    get_cfa_positions(cfa_pattern)
    if raw.ndim == 4:
        if raw.shape[1] != 1:
            raise ValueError(f"Expected a single RAW channel, got {tuple(raw.shape)}")
        raw = raw[:, 0]
    if raw.ndim != 3 or raw.shape[-2] % 2 or raw.shape[-1] % 2:
        raise ValueError(f"RAW must have even [B,H,W] shape, got {tuple(raw.shape)}")
    samples = torch.stack((raw[..., 0::2, 0::2], raw[..., 0::2, 1::2], raw[..., 1::2, 0::2], raw[..., 1::2, 1::2]), dim=1)
    pattern = cfa_pattern.upper()
    indices = [pattern.index("R"), [index for index, value in enumerate(pattern) if value == "G"][0], [index for index, value in enumerate(pattern) if value == "G"][1], pattern.index("B")]
    return samples[:, indices]


def bayer_unpack(packed: Tensor, cfa_pattern: str = "RGGB") -> Tensor:
    """Unpack physical [R,G1,G2,B] planes to a single Bayer RAW image."""
    get_cfa_positions(cfa_pattern)
    if packed.ndim != 4 or packed.shape[1] != 4:
        raise ValueError(f"Packed RAW must be [B,4,H,W], got {tuple(packed.shape)}")
    pattern = cfa_pattern.upper()
    plane_for_cell = [0, 1, 2, 3]
    r_index, g_indices, b_index = pattern.index("R"), [index for index, value in enumerate(pattern) if value == "G"], pattern.index("B")
    plane_for_cell[r_index] = 0
    plane_for_cell[g_indices[0]] = 1
    plane_for_cell[g_indices[1]] = 2
    plane_for_cell[b_index] = 3
    output = packed.new_empty((packed.shape[0], packed.shape[-2] * 2, packed.shape[-1] * 2))
    output[..., 0::2, 0::2] = packed[:, plane_for_cell[0]]
    output[..., 0::2, 1::2] = packed[:, plane_for_cell[1]]
    output[..., 1::2, 0::2] = packed[:, plane_for_cell[2]]
    output[..., 1::2, 1::2] = packed[:, plane_for_cell[3]]
    return output


def build_motion_prior(prev_packed: Tensor, curr_packed: Tensor) -> Tensor:
    if prev_packed.shape != curr_packed.shape:
        raise ValueError("prev_packed and curr_packed must have identical shapes")
    return (curr_packed - prev_packed).abs()


def prepare_noisy_pair_features(noisy_pair: Tensor, raw_range: RawRange = DEFAULT_RAW_RANGE, cfa_pattern: str = "RGGB") -> RawPairFeatures:
    if noisy_pair.ndim != 4 or noisy_pair.shape[1] != 2:
        raise ValueError(f"noisy_pair must be [B,2,H,W], got {tuple(noisy_pair.shape)}")
    prev = bayer_pack(normalize_raw(noisy_pair[:, 0], raw_range), cfa_pattern)
    curr = bayer_pack(normalize_raw(noisy_pair[:, 1], raw_range), cfa_pattern)
    return RawPairFeatures(prev_packed=prev, curr_packed=curr, motion_prior=build_motion_prior(prev, curr))

__all__ = [
    "DEFAULT_RAW_RANGE",
    "RawPairFeatures",
    "RawRange",
    "bayer_pack",
    "bayer_unpack",
    "build_motion_prior",
    "get_cfa_positions",
    "normalize_raw",
    "prepare_noisy_pair_features",
]

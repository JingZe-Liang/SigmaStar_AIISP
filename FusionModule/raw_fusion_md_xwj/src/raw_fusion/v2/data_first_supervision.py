"""MOG2-only supervision for the data-first V2 protocol."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .bands import b2
from .md import Mog2ConfigV2, classify_label_frame
from .schemas.common import ContractError


FrameProvider = Callable[[str, int], np.ndarray]
STATE_UNKNOWN, STATE_FLAT, STATE_TEXTURE, STATE_MOTION = 0, 1, 2, 3


def _classify_cells(pixel_state: np.ndarray, cell_size: int = 32) -> np.ndarray:
    height, width = pixel_state.shape
    result = np.full(((height + cell_size - 1) // cell_size, (width + cell_size - 1) // cell_size), STATE_UNKNOWN, dtype=np.uint8)
    for cell_y in range(result.shape[0]):
        for cell_x in range(result.shape[1]):
            cell = pixel_state[cell_y * cell_size : min((cell_y + 1) * cell_size, height), cell_x * cell_size : min((cell_x + 1) * cell_size, width)]
            if np.any(cell == STATE_MOTION):
                result[cell_y, cell_x] = STATE_MOTION
            elif np.mean(cell == STATE_TEXTURE) >= 0.9:
                result[cell_y, cell_x] = STATE_TEXTURE
            elif np.mean(cell == STATE_FLAT) >= 0.95:
                result[cell_y, cell_x] = STATE_FLAT
    return result


@dataclass(frozen=True, slots=True)
class MOG2SupervisionConfig:
    mog2: Mog2ConfigV2
    texture_threshold: float = 0.002
    alpha_target: float = 0.125
    alpha_class: int = 1
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.texture_threshold) or self.texture_threshold < 0.0:
            raise ContractError("data-first texture_threshold must be finite and non-negative")
        if self.alpha_target != 0.125 or self.alpha_class != 1:
            raise ContractError("data-first alpha heuristic is fixed at class 1 / target 0.125")
        if self.confidence != 1.0:
            raise ContractError("data-first confidence heuristic is fixed at 1.0")


@dataclass(frozen=True, slots=True)
class DataFirstSupervision:
    pixel_state: np.ndarray
    cell_state: np.ndarray
    texture_admission: np.ndarray
    policy_alpha_target: np.ndarray
    policy_alpha_class: np.ndarray
    policy_alpha_valid: np.ndarray
    policy_confidence: np.ndarray
    hf_target: np.ndarray
    hf_valid: np.ndarray
    valid_bits: np.ndarray
    md_mask: np.ndarray


def _packed(value: np.ndarray, context: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 4 or array.dtype != np.uint16:
        raise ContractError(f"{context} must be uint16 [4,H,W]")
    return np.ascontiguousarray(array)


class MOG2SupervisionGenerator:
    """Generate one target's labels from a fresh, deterministic MOG2 replay."""

    def __init__(self, frames: FrameProvider, *, config: MOG2SupervisionConfig) -> None:
        if not callable(frames):
            raise TypeError("data-first frames must be callable")
        self.frames = frames
        self.config = config

    def supervise(self, condition: str, frame: int) -> DataFirstSupervision:
        if condition not in {"128x", "645x"}:
            raise ContractError("data-first condition must be 128x or 645x")
        query = classify_label_frame(
            self.config.mog2,
            lambda index: _packed(self.frames(condition, index), f"frame {index}"),
            query_frame=int(frame),
        )
        return self.supervise_from_mask(condition, frame, query.mask)

    def supervise_from_mask(self, condition: str, frame: int, mask: np.ndarray) -> DataFirstSupervision:
        if condition not in {"128x", "645x"}:
            raise ContractError("data-first condition must be 128x or 645x")
        target = _packed(self.frames(condition, int(frame)), "target frame")
        md_mask = np.asarray(mask > 0, dtype=np.uint8)
        if md_mask.shape != target.shape[1:]:
            raise ContractError("cached MOG2 mask shape does not match the target frame")
        normalized = target.astype(np.float32) / np.float32(4095.0)
        band = b2(torch.from_numpy(normalized[None]))[0].numpy()
        energy = np.mean(np.abs(band), axis=0)
        valid = np.all(np.isfinite(normalized), axis=0)
        motion = md_mask.astype(bool)
        texture = (~motion) & valid & (energy >= self.config.texture_threshold)
        flat = (~motion) & valid & ~texture
        pixel_state = np.full(md_mask.shape, STATE_UNKNOWN, dtype=np.uint8)
        pixel_state[flat] = STATE_FLAT
        pixel_state[texture] = STATE_TEXTURE
        pixel_state[motion & valid] = STATE_MOTION
        cell_state = _classify_cells(pixel_state)
        texture_admission = np.ascontiguousarray((cell_state == STATE_TEXTURE).astype(np.uint8)[None])
        policy_alpha_valid = texture_admission.copy()
        policy_alpha_target = np.where(policy_alpha_valid, self.config.alpha_target, 0.0).astype(np.float32)
        policy_alpha_target = policy_alpha_target[None] if policy_alpha_target.ndim == 2 else policy_alpha_target
        policy_alpha_class = np.where(policy_alpha_valid, self.config.alpha_class, 0).astype(np.uint8)
        policy_confidence = np.where(policy_alpha_valid, self.config.confidence, 0.0).astype(np.float32)
        hf_valid = np.broadcast_to(valid[None], (4, *valid.shape)).copy()
        hf_target = np.where(hf_valid, band, 0.0).astype(np.float32)
        valid_bits = np.where(valid & ~motion, np.uint8(1), np.uint8(0))
        return DataFirstSupervision(
            pixel_state=np.ascontiguousarray(pixel_state),
            cell_state=np.ascontiguousarray(cell_state[None]),
            texture_admission=texture_admission,
            policy_alpha_target=np.ascontiguousarray(policy_alpha_target),
            policy_alpha_class=np.ascontiguousarray(policy_alpha_class),
            policy_alpha_valid=np.ascontiguousarray(policy_alpha_valid),
            policy_confidence=np.ascontiguousarray(policy_confidence),
            hf_target=np.ascontiguousarray(hf_target),
            hf_valid=np.ascontiguousarray(hf_valid),
            valid_bits=np.ascontiguousarray(valid_bits[None]),
            md_mask=np.ascontiguousarray(md_mask),
        )


__all__ = ["DataFirstSupervision", "MOG2SupervisionConfig", "MOG2SupervisionGenerator"]

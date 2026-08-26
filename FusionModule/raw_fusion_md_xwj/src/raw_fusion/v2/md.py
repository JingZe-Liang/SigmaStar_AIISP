"""MOG2 utilities used exclusively for data-first training supervision."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .schemas.common import ContractError


@dataclass(frozen=True, slots=True)
class Mog2ConfigV2:
    history: int
    warmup_frames: int = 50
    var_threshold: float = 64.0
    detect_shadows: bool = False

    def __post_init__(self) -> None:
        if self.history <= 0 or self.warmup_frames != 50 or not math.isfinite(self.var_threshold):
            raise ContractError("MOG2 requires positive history, 50-frame warmup, and finite threshold")


@dataclass(frozen=True, slots=True)
class LabelQueryResult:
    mask: np.ndarray


def green_plane_u16(packed: np.ndarray) -> np.ndarray:
    value = np.asarray(packed)
    if value.dtype != np.uint16 or value.ndim != 3 or value.shape[0] != 4:
        raise ValueError("packed must be uint16 [4,H,W]")
    return np.ascontiguousarray(((value[1].astype(np.uint32) + value[3].astype(np.uint32)) // 2).astype(np.float32))


def create_mog2(config: Mog2ConfigV2):
    return cv2.createBackgroundSubtractorMOG2(history=config.history, varThreshold=config.var_threshold, detectShadows=config.detect_shadows)


def classify_label_frame(config: Mog2ConfigV2, frames: Mapping[int, np.ndarray] | Sequence[np.ndarray] | Callable[[int], np.ndarray], *, query_frame: int) -> LabelQueryResult:
    def read(index: int) -> np.ndarray:
        value = frames(index) if callable(frames) else frames[index]
        return green_plane_u16(np.asarray(value))
    if not 56 <= int(query_frame) <= 199:
        raise ContractError("MOG2 query frame must be in 56..199")
    subtractor = create_mog2(config)
    for index in range(50):
        subtractor.apply(read(index), learningRate=-1.0)
    raw_mask = subtractor.apply(read(int(query_frame)), learningRate=0.0)
    return LabelQueryResult(np.ascontiguousarray(cv2.medianBlur(np.ascontiguousarray(raw_mask, dtype=np.uint8), 3)))


__all__ = ["LabelQueryResult", "Mog2ConfigV2", "classify_label_frame", "create_mog2", "green_plane_u16"]

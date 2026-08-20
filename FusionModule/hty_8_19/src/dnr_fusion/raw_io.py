from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class RawSpec:
    width: int
    height: int
    shift: int = 0

    @property
    def pixels_per_frame(self) -> int:
        return self.width * self.height

    @property
    def bytes_per_frame(self) -> int:
        return self.pixels_per_frame * np.dtype("<u2").itemsize


class RawStream:
    def __init__(self, path: str | Path, spec: RawSpec):
        self.path = Path(path)
        self.spec = spec
        size = self.path.stat().st_size
        if size % spec.bytes_per_frame:
            raise ValueError(
                f"{self.path} has {size} bytes, not a multiple of "
                f"{spec.bytes_per_frame} bytes/frame"
            )
        self.frame_count = size // spec.bytes_per_frame
        self._array = np.memmap(
            self.path,
            dtype="<u2",
            mode="r",
            shape=(self.frame_count, spec.height, spec.width),
        )

    def frame(self, index: int, *, copy: bool = False) -> np.ndarray:
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame {index} outside [0, {self.frame_count})")
        frame = np.asarray(self._array[index])
        if self.spec.shift:
            frame = np.right_shift(frame, self.spec.shift).astype(np.uint16, copy=False)
        elif copy:
            frame = frame.copy()
        return frame

    def iter_frames(self) -> Iterator[np.ndarray]:
        for index in range(self.frame_count):
            yield self.frame(index)


def pack_rggb(mosaic: np.ndarray) -> np.ndarray:
    if mosaic.ndim != 2:
        raise ValueError(f"Expected HxW mosaic, got shape {mosaic.shape}")
    if mosaic.shape[0] % 2 or mosaic.shape[1] % 2:
        raise ValueError(f"Bayer dimensions must be even, got {mosaic.shape}")
    return np.stack(
        (
            mosaic[0::2, 0::2],  # R
            mosaic[0::2, 1::2],  # Gr
            mosaic[1::2, 1::2],  # B
            mosaic[1::2, 0::2],  # Gb
        ),
        axis=0,
    )


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected 4xHxW packed Bayer, got shape {packed.shape}")
    _, height, width = packed.shape
    mosaic = np.empty((height * 2, width * 2), dtype=packed.dtype)
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 1::2] = packed[2]
    mosaic[1::2, 0::2] = packed[3]
    return mosaic


def normalize_raw(
    packed: np.ndarray,
    black: float,
    white: float,
    *,
    clip: tuple[float, float] | None = (-0.10, 1.10),
) -> np.ndarray:
    output = (packed.astype(np.float32) - black) / (white - black)
    if clip is not None:
        np.clip(output, clip[0], clip[1], out=output)
    return output


def frame_count(path: str | Path, width: int, height: int) -> int:
    spec = RawSpec(width=width, height=height)
    size = Path(path).stat().st_size
    if size % spec.bytes_per_frame:
        raise ValueError(f"Invalid RAW stream length: {path}")
    return size // spec.bytes_per_frame


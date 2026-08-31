"""RAW stream reading and packed-RGGB conversion for FGRF-Net v2.0."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_FRAME_NUMBER = re.compile(r"(?:out[_-])?(\d+)", re.IGNORECASE)


def _natural_key(path: Path) -> tuple[int, str]:
    match = _FRAME_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else 10**12, path.name.lower())


class RawSequence:
    """Read a uint16 RAW stream or a directory of uint16 RAW frames."""

    def __init__(self, path: str | Path, height: int, width: int, dtype: str = "uint16") -> None:
        self.path = Path(path)
        self.height = int(height)
        self.width = int(width)
        self.dtype = np.dtype(dtype)
        frame_bytes = self.height * self.width * self.dtype.itemsize
        if self.path.is_file():
            size = self.path.stat().st_size
            if size % frame_bytes:
                raise ValueError(f"RAW stream does not contain whole frames: {self.path}")
            self._frame_count = size // frame_bytes
            self._memmap = np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                shape=(self._frame_count, self.height, self.width),
            )
            self._files: list[Path] | None = None
        elif self.path.is_dir():
            self._files = sorted(self.path.glob("*.raw"), key=_natural_key)
            if not self._files:
                raise FileNotFoundError(f"No RAW files in {self.path}")
            if any(file.stat().st_size != frame_bytes for file in self._files):
                raise ValueError(f"Per-frame RAW size mismatch in {self.path}")
            self._frame_count = len(self._files)
            self._memmap = None
        else:
            raise FileNotFoundError(self.path)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read_uint16(self, index: int, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
        if not 0 <= index < self._frame_count:
            raise IndexError(f"Frame {index} outside [0, {self._frame_count})")
        if self._memmap is not None:
            frame = np.asarray(self._memmap[index])
        else:
            assert self._files is not None
            frame = np.fromfile(self._files[index], dtype=self.dtype).reshape(self.height, self.width)
        if crop is None:
            return frame
        top, bottom, left, right = crop
        return frame[top:bottom, left:right]


def pack_rggb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 2 or frame.shape[0] % 2 or frame.shape[1] % 2:
        raise ValueError(f"Expected an even 2D Bayer frame, got {frame.shape}")
    return np.stack(
        (frame[0::2, 0::2], frame[0::2, 1::2], frame[1::2, 0::2], frame[1::2, 1::2]),
        axis=0,
    ).astype(np.float32, copy=False)


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected [4, h, w], got {packed.shape}")
    _, height, width = packed.shape
    frame = np.empty((height * 2, width * 2), dtype=packed.dtype)
    frame[0::2, 0::2] = packed[0]
    frame[0::2, 1::2] = packed[1]
    frame[1::2, 0::2] = packed[2]
    frame[1::2, 1::2] = packed[3]
    return frame


def read_packed_normalized(
    sequence: RawSequence,
    index: int,
    black_level: float,
    white_level: float,
    right_shift: int = 0,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    frame = sequence.read_uint16(index, crop=crop).astype(np.float32)
    if right_shift:
        frame /= float(1 << right_shift)
    frame = (frame - float(black_level)) / max(float(white_level) - float(black_level), 1.0)
    return pack_rggb(np.clip(frame, 0.0, 1.0))

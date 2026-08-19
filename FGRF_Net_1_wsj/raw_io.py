"""RAW sequence readers and RGGB packing utilities for FGRF-Net."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np


_FRAME_NUMBER = re.compile(r"(?:out[_-])?(\d+)", re.IGNORECASE)


def _natural_key(path: Path) -> tuple[int, str]:
    match = _FRAME_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else 10**12, path.name.lower())


class RawSequence:
    """Memory-mapped 16-bit RAW stream or a directory of per-frame RAW files."""

    def __init__(
        self,
        path: str | Path,
        height: int,
        width: int,
        dtype: str = "uint16",
    ) -> None:
        self.path = Path(path)
        self.height = int(height)
        self.width = int(width)
        self.dtype = np.dtype(dtype)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")

        if self.path.is_file():
            item_bytes = self.dtype.itemsize
            frame_bytes = self.height * self.width * item_bytes
            size = self.path.stat().st_size
            if size % frame_bytes:
                raise ValueError(
                    f"RAW size is not divisible by one frame: {self.path} "
                    f"({size} bytes, {frame_bytes} bytes/frame)"
                )
            self._frames = size // frame_bytes
            self._memmap = np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                shape=(self._frames, self.height, self.width),
            )
            self._files: list[Path] | None = None
        elif self.path.is_dir():
            self._files = sorted(self.path.glob("*.raw"), key=_natural_key)
            if not self._files:
                raise FileNotFoundError(f"No .raw frames found in {self.path}")
            expected = self.height * self.width * self.dtype.itemsize
            bad = [p for p in self._files if p.stat().st_size != expected]
            if bad:
                raise ValueError(
                    f"Per-frame RAW size mismatch in {self.path}: {bad[0].name}"
                )
            self._frames = len(self._files)
            self._memmap = None
        else:
            raise FileNotFoundError(self.path)

    @property
    def frame_count(self) -> int:
        return self._frames

    def read_uint16(
        self,
        index: int,
        crop: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        if not 0 <= index < self._frames:
            raise IndexError(f"frame index {index} outside [0, {self._frames})")
        if self._memmap is not None:
            if crop is None:
                return np.asarray(self._memmap[index])
            return np.asarray(self._memmap[index, crop[0]:crop[1], crop[2]:crop[3]])
        assert self._files is not None
        frame = np.fromfile(self._files[index], dtype=self.dtype).reshape(self.height, self.width)
        return frame if crop is None else frame[crop[0]:crop[1], crop[2]:crop[3]]

    def read_normalized(
        self,
        index: int,
        black_level: float,
        white_level: float = 4095.0,
        right_shift: int = 0,
        crop: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Return black-level corrected linear data in approximately [0, 1]."""
        frame = self.read_uint16(index, crop=crop).astype(np.float32)
        if right_shift:
            frame = frame / float(1 << right_shift)
        frame = (frame - float(black_level)) / max(float(white_level) - black_level, 1.0)
        return np.clip(frame, 0.0, 1.0)


def pack_rggb(frame: np.ndarray) -> np.ndarray:
    """Pack a single even-sized RGGB mosaic into [R, G1, G2, B]."""
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D Bayer frame, got {frame.shape}")
    height, width = frame.shape
    if height % 2 or width % 2:
        raise ValueError(f"RGGB packing requires even dimensions, got {frame.shape}")
    return np.stack(
        (
            frame[0::2, 0::2],
            frame[0::2, 1::2],
            frame[1::2, 0::2],
            frame[1::2, 1::2],
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    """Unpack [R, G1, G2, B] into one RGGB mosaic frame."""
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected [4, h, w] packed RAW, got {packed.shape}")
    _, height, width = packed.shape
    output = np.empty((height * 2, width * 2), dtype=packed.dtype)
    output[0::2, 0::2] = packed[0]
    output[0::2, 1::2] = packed[1]
    output[1::2, 0::2] = packed[2]
    output[1::2, 1::2] = packed[3]
    return output


def read_packed_normalized(
    sequence: RawSequence,
    index: int,
    black_level: float,
    white_level: float,
    right_shift: int,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    return pack_rggb(
        sequence.read_normalized(
            index,
            black_level=black_level,
            white_level=white_level,
            right_shift=right_shift,
            crop=crop,
        )
    )


def validate_same_sequence_length(sequences: Iterable[RawSequence]) -> int:
    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one RAW sequence is required")
    counts = {sequence.frame_count for sequence in sequences}
    if len(counts) != 1:
        details = ", ".join(
            f"{sequence.path.name}={sequence.frame_count}" for sequence in sequences
        )
        raise ValueError(f"RAW frame counts do not match: {details}")
    return sequences[0].frame_count

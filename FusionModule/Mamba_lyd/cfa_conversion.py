"""Bayer CFA conversion helpers for RAW frames and raw video streams.

The conversion is a lossless permutation of samples within every 2x2 Bayer
cell.  It changes the CFA phase without demosaicing or changing pixel values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def rggb_to_gbrg(frame: NDArray[np.generic]) -> NDArray[np.generic]:
    """Return one even-sized RGGB mosaic re-phased as GBRG.

    RGGB cells are ``[[R, G], [G, B]]`` and GBRG cells are
    ``[[G, B], [R, G]]``.  Swapping the two rows inside each cell performs
    exactly that sample permutation.
    """
    if frame.ndim != 2:
        raise ValueError(f"frame must be two-dimensional, got shape {frame.shape}")
    height, width = frame.shape
    if height % 2 or width % 2:
        raise ValueError(f"frame dimensions must be even, got {width}x{height}")

    converted = frame.copy()
    converted[0::2, :] = frame[1::2, :]
    converted[1::2, :] = frame[0::2, :]
    return converted


def convert_rggb_video_to_gbrg(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
    dtype: np.dtype = np.dtype("<u2"),
) -> int:
    """Convert a headerless RGGB RAW video to GBRG without loading it all.

    The source must contain consecutive, tightly packed single-channel frames.
    Returns the converted frame count.
    """
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even numbers")

    dtype = np.dtype(dtype)
    frame_bytes = width * height * dtype.itemsize
    source_bytes = source.stat().st_size
    if source_bytes % frame_bytes:
        raise ValueError(
            f"source size ({source_bytes} bytes) is not divisible by one "
            f"{width}x{height} frame ({frame_bytes} bytes)"
        )

    frame_count = source_bytes // frame_bytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_frames = np.memmap(source, mode="r", dtype=dtype, shape=(frame_count, height, width))
    destination_frames = np.memmap(destination, mode="w+", dtype=dtype, shape=(frame_count, height, width))
    try:
        for index in range(frame_count):
            destination_frames[index, 0::2, :] = source_frames[index, 1::2, :]
            destination_frames[index, 1::2, :] = source_frames[index, 0::2, :]
        destination_frames.flush()
    finally:
        del source_frames
        del destination_frames
    return int(frame_count)

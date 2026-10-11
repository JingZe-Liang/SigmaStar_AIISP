"""Chunked RAW stream readers and RGGB packing.

Both readers yield uint16 sensor-domain frames and packed [n, 4, H/2, W/2]
arrays in the canonical channel order (R, Gr, Gb, B). Bit alignment and CFA
phase always come from a SourceAdapter, never from the file.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import FilenameMetadata, SourceAdapter, parse_filename_metadata

# Canonical packed channel order.
CHANNEL_NAMES: tuple[str, str, str, str] = ("R", "Gr", "Gb", "B")

# Per-CFA sampling offsets (row offset, col offset) for each canonical channel
# inside the 2x2 Bayer cell.
_CFA_CELL_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    # row0: R Gr | row1: Gb B
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    # row0: Gr R | row1: B Gb
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
}


def pack_frames(frames: np.ndarray, adapter: SourceAdapter) -> np.ndarray:
    """Pack sensor-domain frames [n, H, W] uint16 into [n, 4, H/2, W/2] uint16.

    The bit shift and the CFA phase remap are both applied here, so everything
    downstream sees canonical 12-bit DN (R, Gr, Gb, B) regardless of source.
    """

    if frames.ndim != 3:
        raise ValueError(f"expected [n, H, W] frames, got shape {frames.shape}")
    shifted = frames >> adapter.bit_shift
    offsets = _CFA_CELL_OFFSETS[adapter.cfa]
    return np.stack([shifted[:, rs::2, cs::2] for rs, cs in offsets], axis=1)


def pack_frame(frame: np.ndarray, adapter: SourceAdapter) -> np.ndarray:
    """Pack a single sensor-domain frame [H, W] into [4, H/2, W/2]."""

    return pack_frames(frame[None], adapter)[0]


class RawStreamReader:
    """Memmap reader for a single multi-frame .raw file."""

    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height
        frame_bytes = width * height * 2
        file_bytes = path.stat().st_size
        if file_bytes % frame_bytes != 0:
            raise ValueError(
                f"{path}: size {file_bytes} is not a multiple of frame size {frame_bytes}; "
                "wrong width/height or corrupted stream"
            )
        self.frames = file_bytes // frame_bytes
        self._data = np.memmap(path, dtype="<u2", mode="r", shape=(self.frames, height, width))

    def read_frame(self, index: int) -> np.ndarray:
        return np.array(self._data[index])

    def iter_chunks(self, chunk_frames: int) -> Iterator[np.ndarray]:
        for start in range(0, self.frames, chunk_frames):
            yield np.array(self._data[start : start + chunk_frames])


class RawFrameDirectoryReader:
    """Reader for per-frame .raw files (group 03 layout: out_0000.raw, ...)."""

    def __init__(self, directory: Path, width: int, height: int, pattern: str = "out_*.raw") -> None:
        self.directory = directory
        self.width = width
        self.height = height
        frame_bytes = width * height * 2
        self._files = sorted(directory.glob(pattern))
        if not self._files:
            raise ValueError(f"{directory}: no files match {pattern!r}")
        for path in self._files:
            if path.stat().st_size != frame_bytes:
                raise ValueError(
                    f"{path}: size {path.stat().st_size} != frame size {frame_bytes}"
                )
        self.frames = len(self._files)

    def read_frame(self, index: int) -> np.ndarray:
        return np.fromfile(self._files[index], dtype="<u2").reshape(self.height, self.width)

    def iter_chunks(self, chunk_frames: int) -> Iterator[np.ndarray]:
        for start in range(0, self.frames, chunk_frames):
            batch = [
                np.fromfile(p, dtype="<u2").reshape(self.height, self.width)
                for p in self._files[start : start + chunk_frames]
            ]
            yield np.stack(batch)


def open_stream(path: Path, adapter: SourceAdapter) -> tuple[RawStreamReader | RawFrameDirectoryReader, FilenameMetadata]:
    """Open a stream with the reader kind demanded by its adapter."""

    if adapter.reader_kind == "single_file":
        meta = parse_filename_metadata(path)
        return RawStreamReader(path, meta.width, meta.height), meta
    if adapter.reader_kind == "frame_directory":
        # Per-frame layouts encode the size in the parent stream directory name.
        meta = parse_filename_metadata(path.parent.name)
        return RawFrameDirectoryReader(path, meta.width, meta.height), meta
    raise ValueError(f"unknown reader_kind {adapter.reader_kind!r}")

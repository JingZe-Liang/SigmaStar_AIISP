"""Strict Bayer RAW packing, normalization, and file readers."""

from pathlib import Path
import operator

import numpy as np


def _validate_levels(black_level: int, white_level: int) -> tuple[int, int]:
    black = operator.index(black_level)
    white = operator.index(white_level)
    if black < 0 or white <= black or white > np.iinfo(np.uint16).max:
        raise ValueError("black_level and white_level must satisfy 0 <= black < white <= 65535")
    return black, white


def pack_rggb(mosaic: np.ndarray) -> np.ndarray:
    """Pack a two-dimensional RGGB mosaic as ``[R, Gr, B, Gb]`` planes."""
    array = np.asarray(mosaic)
    if array.ndim != 2:
        raise ValueError("mosaic must be a two-dimensional array")
    height, width = array.shape
    if height % 2 or width % 2:
        raise ValueError("mosaic dimensions must be even")
    return np.ascontiguousarray(
        np.stack(
            (
                array[0::2, 0::2],
                array[0::2, 1::2],
                array[1::2, 1::2],
                array[1::2, 0::2],
            ),
            axis=0,
        )
    )


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    """Unpack ``[R, Gr, B, Gb]`` planes into a two-dimensional RGGB mosaic."""
    array = np.asarray(packed)
    if array.ndim != 3 or array.shape[0] != 4:
        raise ValueError("packed must have shape [4, height, width]")
    _, height, width = array.shape
    mosaic = np.empty((height * 2, width * 2), dtype=array.dtype)
    mosaic[0::2, 0::2] = array[0]
    mosaic[0::2, 1::2] = array[1]
    mosaic[1::2, 1::2] = array[2]
    mosaic[1::2, 0::2] = array[3]
    return np.ascontiguousarray(mosaic)


def normalize_raw(raw12: np.ndarray, black_level: int, white_level: int) -> np.ndarray:
    """Map RAW code values to float32 in the closed interval [0, 1]."""
    black, white = _validate_levels(black_level, white_level)
    raw = np.asarray(raw12, dtype=np.float32)
    normalized = (raw - np.float32(black)) / np.float32(white - black)
    return np.ascontiguousarray(np.clip(normalized, 0.0, 1.0, dtype=np.float32))


def quantize_normalized(
    normalized: np.ndarray, black_level: int, white_level: int
) -> np.ndarray:
    """Quantize normalized values to right-aligned little-endian uint16 RAW codes."""
    black, white = _validate_levels(black_level, white_level)
    values = np.asarray(normalized, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized contains 非有限值")
    clipped = np.clip(values, 0.0, 1.0)
    quantized = np.rint(clipped * np.float32(white - black) + np.float32(black))
    return np.ascontiguousarray(quantized, dtype=np.dtype("<u2"))


def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
    columns = operator.index(width)
    rows = operator.index(height)
    if columns <= 0 or rows <= 0:
        raise ValueError("width and height must be positive")
    return columns, rows


def _validate_shift(shift: int) -> int:
    value = operator.index(shift)
    if value < 0 or value > 15:
        raise ValueError("shift must be between 0 and 15")
    return value


def _expected_bytes(width: int, height: int) -> int:
    return width * height * 2


def _validate_frame_file(path: Path, expected_bytes: int) -> None:
    try:
        actual_bytes = path.stat().st_size
    except OSError as error:
        raise ValueError(f"无法读取 RAW 文件: {path}") from error
    if actual_bytes != expected_bytes:
        raise ValueError(f"RAW 文件 {path} 字节数为 {actual_bytes}，期望字节数 {expected_bytes}")


class RawStreamReader:
    """Read fixed-size little-endian uint16 frames from a RAW stream."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        frame_count: int,
        shift: int,
    ) -> None:
        self.path = Path(path)
        self.width, self.height = _validate_dimensions(width, height)
        self.frame_count = operator.index(frame_count)
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.shift = _validate_shift(shift)
        expected_bytes = _expected_bytes(self.width, self.height) * self.frame_count
        _validate_frame_file(self.path, expected_bytes)
        self._frames = np.memmap(
            self.path,
            dtype=np.dtype("<u2"),
            mode="r",
            shape=(self.frame_count, self.height, self.width),
        )

    def read_frame(self, index: int) -> np.ndarray:
        frame_index = operator.index(index)
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"frame index out of range: {frame_index}")
        frame = np.array(self._frames[frame_index], dtype=np.dtype("<u2"), copy=True)
        if self.shift:
            frame = np.right_shift(frame, self.shift)
        return np.ascontiguousarray(frame)

    def read_crop(
        self, index: int, top: int, left: int, height: int, width: int
    ) -> np.ndarray:
        frame_index = operator.index(index)
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"frame index out of range: {frame_index}")
        crop_width, crop_height = _validate_dimensions(width, height)
        top_index = operator.index(top)
        left_index = operator.index(left)
        if (
            top_index < 0
            or left_index < 0
            or top_index + crop_height > self.height
            or left_index + crop_width > self.width
        ):
            raise ValueError("crop is outside frame bounds")
        crop = np.array(
            self._frames[
                frame_index,
                top_index : top_index + crop_height,
                left_index : left_index + crop_width,
            ],
            dtype=np.dtype("<u2"),
            copy=True,
        )
        if self.shift:
            crop = np.right_shift(crop, self.shift)
        return np.ascontiguousarray(crop)


class RawFrameDirectoryReader:
    """Read individually stored fixed-size little-endian RAW frames."""

    def __init__(
        self, directory: Path, pattern: str, width: int, height: int, shift: int
    ) -> None:
        self.directory = Path(directory)
        self.pattern = pattern
        self.width, self.height = _validate_dimensions(width, height)
        self.shift = _validate_shift(shift)
        self._frame_bytes = _expected_bytes(self.width, self.height)

    def _read(self, index: int) -> np.ndarray:
        frame_index = operator.index(index)
        if frame_index < 0:
            raise IndexError(f"frame index cannot be negative: {frame_index}")
        path = self.directory / self.pattern.format(index=frame_index)
        _validate_frame_file(path, self._frame_bytes)
        frame = np.fromfile(path, dtype=np.dtype("<u2"), count=self.width * self.height)
        frame = frame.reshape(self.height, self.width)
        if self.shift:
            frame = np.right_shift(frame, self.shift)
        return np.ascontiguousarray(frame)

    def read_frame(self, index: int) -> np.ndarray:
        return self._read(index)

    def read_crop(
        self, index: int, top: int, left: int, height: int, width: int
    ) -> np.ndarray:
        crop_width, crop_height = _validate_dimensions(width, height)
        top_index = operator.index(top)
        left_index = operator.index(left)
        if (
            top_index < 0
            or left_index < 0
            or top_index + crop_height > self.height
            or left_index + crop_width > self.width
        ):
            raise ValueError("crop is outside frame bounds")
        frame = self._read(index)
        return np.ascontiguousarray(
            frame[top_index : top_index + crop_height, left_index : left_index + crop_width]
        )

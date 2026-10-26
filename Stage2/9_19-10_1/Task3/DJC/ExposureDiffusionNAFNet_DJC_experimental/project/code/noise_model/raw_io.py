"""Explicit RAW container decoding and Bayer packing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawSpec:
    width: int
    height: int
    bit_shift: int = 4
    valid_bits: int = 12
    cfa: str = "GRBG"
    dtype: str = "<u2"
    container_bits: int = 16
    strict_alignment: bool = True

    def __post_init__(self) -> None:
        storage_bits = np.dtype(self.dtype).itemsize * 8
        if self.container_bits != storage_bits:
            raise ValueError(
                "container_bits must match dtype storage width; "
                f"got {self.container_bits} for {self.dtype}"
            )
        if self.container_bits != 16 or self.dtype != "<u2":
            raise ValueError("MIS20S1 RAW requires little-endian uint16 storage")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RAW dimensions must be positive")
        if self.bit_shift < 0 or self.valid_bits <= 0:
            raise ValueError("bit_shift and valid_bits must be positive")
        if self.bit_shift + self.valid_bits > self.container_bits:
            raise ValueError("valid bits exceed the container width")

    @property
    def white_level(self) -> int:
        return (1 << self.valid_bits) - 1

    @property
    def valid_mask(self) -> int:
        return self.white_level

    @property
    def alignment_mask(self) -> int:
        return (1 << self.bit_shift) - 1


def decode_raw16(raw16: np.ndarray, spec: RawSpec) -> np.ndarray:
    """Decode the left-aligned valid DN from the uint16 RAW container.

    The returned array is always float32 DN in [0, spec.white_level].
    Keeping this conversion explicit prevents container values from entering
    calibration or training code by accident.
    """
    values = np.asarray(raw16)
    if values.dtype.kind != "u" or values.dtype.itemsize != 2:
        raise ValueError("raw16 must be an unsigned 16-bit array")
    values = values.astype(np.uint16, copy=False)
    if spec.strict_alignment and np.any(values & spec.alignment_mask):
        raise ValueError(
            "RAW container has nonzero low padding bits; expected "
            f"12-bit DN left-aligned by {spec.bit_shift} bits"
        )
    decoded = values >> spec.bit_shift
    if np.any(decoded > spec.valid_mask):
        raise ValueError("RAW values exceed the declared valid-bit range")
    return decoded.astype(np.float32, copy=False)


def encode_raw16(dn: np.ndarray, spec: RawSpec) -> np.ndarray:
    """Encode integral 12-bit DN into the project uint16 RAW container."""
    values = np.asarray(dn)
    if not np.isfinite(values).all():
        raise ValueError("DN values must be finite")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("RAW serialization requires integral DN values")
    if np.any(rounded < 0) or np.any(rounded > spec.white_level):
        raise ValueError("DN values exceed the declared valid-bit range")
    return (rounded.astype(np.uint16) << spec.bit_shift)


def count_frames(path: str | Path, spec: RawSpec) -> int:
    byte_count = Path(path).stat().st_size
    item_size = np.dtype(spec.dtype).itemsize
    if byte_count % item_size:
        raise ValueError(f"{path} has a partial {spec.dtype} value")
    item_count = byte_count // item_size
    pixels_per_frame = spec.width * spec.height
    if item_count % pixels_per_frame:
        raise ValueError(f"{path} is not divisible by one RAW frame")
    return item_count // pixels_per_frame


def read_frames(
    path: str | Path,
    spec: RawSpec,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    total = count_frames(path, spec)
    if start_frame < 0 or start_frame >= total:
        raise ValueError(f"start_frame={start_frame} outside [0, {total})")
    stop = total if max_frames is None else min(total, start_frame + max_frames)
    raw16 = np.memmap(path, mode="r", dtype=spec.dtype)
    frames = raw16.reshape(total, spec.height, spec.width)[start_frame:stop]
    return decode_raw16(np.asarray(frames, dtype=np.uint16), spec)


def _channel_positions(cfa: str) -> dict[str, tuple[tuple[int, int], ...]]:
    cfa = cfa.upper()
    if len(cfa) != 4 or any(ch not in "RGB" for ch in cfa):
        raise ValueError(f"unsupported CFA: {cfa}")
    positions: dict[str, list[tuple[int, int]]] = {"R": [], "G": [], "B": []}
    for index, channel in enumerate(cfa):
        positions[channel].append((index // 2, index % 2))
    if len(positions["G"]) != 2:
        raise ValueError(f"CFA must contain two green sites: {cfa}")
    return {key: tuple(value) for key, value in positions.items()}


def pack_bayer(frames: np.ndarray, cfa: str = "GRBG") -> np.ndarray:
    """Pack [N,H,W] Bayer frames into [N,4,H/2,W/2] in R,G1,G2,B order."""
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError("frames must have shape [N,H,W] or [H,W]")
    _, height, width = frames.shape
    if height % 2 or width % 2:
        raise ValueError("RAW dimensions must be even for Bayer packing")
    positions = _channel_positions(cfa)
    output = np.empty((len(frames), 4, height // 2, width // 2), dtype=np.float32)
    output[:, 0] = frames[:, positions["R"][0][0] :: 2, positions["R"][0][1] :: 2]
    output[:, 1] = frames[:, positions["G"][0][0] :: 2, positions["G"][0][1] :: 2]
    output[:, 2] = frames[:, positions["G"][1][0] :: 2, positions["G"][1][1] :: 2]
    output[:, 3] = frames[:, positions["B"][0][0] :: 2, positions["B"][0][1] :: 2]
    return output


def unpack_bayer(packed: np.ndarray, cfa: str = "GRBG") -> np.ndarray:
    """Inverse of pack_bayer for testing and diagnostic output."""
    if packed.ndim == 3:
        packed = packed[None]
    if packed.ndim != 4 or packed.shape[1] != 4:
        raise ValueError("packed must have shape [N,4,H,W]")
    _, _, half_h, half_w = packed.shape
    positions = _channel_positions(cfa)
    output = np.empty((len(packed), half_h * 2, half_w * 2), dtype=packed.dtype)
    output[:, positions["R"][0][0] :: 2, positions["R"][0][1] :: 2] = packed[:, 0]
    output[:, positions["G"][0][0] :: 2, positions["G"][0][1] :: 2] = packed[:, 1]
    output[:, positions["G"][1][0] :: 2, positions["G"][1][1] :: 2] = packed[:, 2]
    output[:, positions["B"][0][0] :: 2, positions["B"][0][1] :: 2] = packed[:, 3]
    return output


def load_packed(path: str | Path, spec: RawSpec, max_frames: int = 1) -> np.ndarray:
    return pack_bayer(read_frames(path, spec, max_frames=max_frames), spec.cfa)

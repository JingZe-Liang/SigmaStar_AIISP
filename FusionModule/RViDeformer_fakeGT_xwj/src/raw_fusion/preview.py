"""Fixed-parameter simple ISP previews and ffmpeg comparison-video helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import operator
from pathlib import Path
import subprocess

import cv2
import numpy as np


_RAW12_MAX = 4095


def simple_isp(
    raw12: np.ndarray,
    black: int,
    white: int,
    white_balance: Sequence[float],
    exposure: float,
) -> np.ndarray:
    """Render a 2-D RGGB RAW12 mosaic to fixed uint8 RGB using a simple ISP.

    Black subtraction and linear mapping happen in Bayer space. White balance and
    exposure are caller-provided fixed sequence settings, never image statistics.
    """
    raw, black_level, white_level, balance, fixed_exposure = _validate_isp_inputs(
        raw12, black, white, white_balance, exposure
    )
    linear16 = np.rint(
        np.clip(
            (raw.astype(np.float32) - np.float32(black_level))
            / np.float32(white_level - black_level),
            0.0,
            1.0,
        )
        * np.float32(np.iinfo(np.uint16).max)
    ).astype(np.uint16)
    rgb16 = cv2.cvtColor(linear16, cv2.COLOR_BayerRGGB2RGB)
    rgb_linear = rgb16.astype(np.float32) / np.float32(np.iinfo(np.uint16).max)
    rgb_linear *= np.asarray(balance, dtype=np.float32).reshape(1, 1, 3)
    rgb_linear *= np.float32(fixed_exposure)
    encoded = _srgb_transfer(np.clip(rgb_linear, 0.0, 1.0))
    return np.rint(encoded * 255.0).astype(np.uint8)


def build_comparison_frame(method_frames: Mapping[str, np.ndarray]) -> np.ndarray:
    """Horizontally concatenate equal HWC uint8 RGB frames in insertion order."""
    if not isinstance(method_frames, Mapping) or not method_frames:
        raise ValueError("method_frames must be a non-empty mapping")
    frames = list(method_frames.values())
    reference_shape: tuple[int, int, int] | None = None
    for frame in frames:
        if not isinstance(frame, np.ndarray):
            raise TypeError("comparison frames must be numpy arrays")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("comparison frames must have HWC RGB shape")
        if frame.dtype != np.uint8:
            raise TypeError("comparison frames must be uint8")
        if reference_shape is None:
            reference_shape = frame.shape
        elif frame.shape != reference_shape:
            raise ValueError("comparison frames must have the same shape")
    return np.ascontiguousarray(np.concatenate(frames, axis=1))


def encode_video_ffmpeg(frame_pattern: Path, output_path: Path, fps: int) -> None:
    """Encode PNG frames with the fixed H.264 command required by the preview flow."""
    try:
        rate = operator.index(fps)
    except TypeError as error:
        raise TypeError("fps must be a positive integer") from error
    if isinstance(fps, bool) or rate <= 0:
        raise ValueError("fps must be a positive integer")
    pattern = Path(frame_pattern)
    destination = Path(output_path)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(rate),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    if not destination.is_file():
        raise RuntimeError(f"ffmpeg succeeded but did not create output: {destination}")


def _validate_isp_inputs(
    raw12: np.ndarray,
    black: int,
    white: int,
    white_balance: Sequence[float],
    exposure: float,
) -> tuple[np.ndarray, int, int, tuple[float, float, float], float]:
    if not isinstance(raw12, np.ndarray):
        raise TypeError("raw12 must be a numpy array")
    if raw12.ndim != 2 or raw12.shape[0] == 0 or raw12.shape[1] == 0:
        raise ValueError("raw12 must be a non-empty two-dimensional mosaic")
    if raw12.shape[0] % 2 or raw12.shape[1] % 2:
        raise ValueError("raw12 dimensions must be even")
    if raw12.dtype != np.uint16:
        raise TypeError("raw12 must have dtype uint16")
    if bool(np.any(raw12 > _RAW12_MAX)):
        raise ValueError("raw12 contains a code above RAW12 maximum 4095")
    black_level, white_level = _validate_levels(black, white)
    if isinstance(white_balance, (str, bytes)):
        raise TypeError("white_balance must have exactly three finite positive values")
    try:
        balance = tuple(float(value) for value in white_balance)
    except (TypeError, ValueError) as error:
        raise TypeError("white_balance must have exactly three finite positive values") from error
    if len(balance) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in balance):
        raise ValueError("white_balance must have exactly three finite positive values")
    try:
        fixed_exposure = float(exposure)
    except (TypeError, ValueError) as error:
        raise TypeError("exposure must be finite and positive") from error
    if not math.isfinite(fixed_exposure) or fixed_exposure <= 0.0:
        raise ValueError("exposure must be finite and positive")
    return raw12, black_level, white_level, (balance[0], balance[1], balance[2]), fixed_exposure


def _validate_levels(black: int, white: int) -> tuple[int, int]:
    if isinstance(black, bool) or isinstance(white, bool):
        raise TypeError("black and white must be integers")
    try:
        black_level = operator.index(black)
        white_level = operator.index(white)
    except TypeError as error:
        raise TypeError("black and white must be integers") from error
    if black_level < 0 or white_level <= black_level or white_level > _RAW12_MAX:
        raise ValueError("levels must satisfy 0 <= black < white <= 4095 for RAW12")
    return black_level, white_level


def _srgb_transfer(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def main(argv: list[str] | None = None) -> None:
    """Write a fixed-exposure RGB preview from one packed RGGB RAW frame."""
    import argparse

    parser = argparse.ArgumentParser(description="将一帧 12-bit RGGB RAW 转为 ISP 预览")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--black", required=True, type=int)
    parser.add_argument("--white", type=int, default=4095)
    parser.add_argument("--white-balance", required=True, help="R,G,B 三个乘数")
    parser.add_argument("--exposure", type=float, default=1.0)
    args = parser.parse_args(argv)
    values = np.fromfile(args.input, dtype=np.dtype("<u2"))
    expected = args.width * args.height
    if values.size != expected:
        raise ValueError(f"RAW 元素数错误：得到 {values.size}，期望 {expected}")
    try:
        wb = tuple(float(part.strip()) for part in args.white_balance.split(","))
    except ValueError as error:
        raise ValueError("--white-balance 必须是 R,G,B") from error
    if len(wb) != 3:
        raise ValueError("--white-balance 必须包含三个数")
    frame = values.reshape(args.height, args.width)
    rgb = simple_isp(frame, args.black, args.white, wb, args.exposure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入预览：{args.output}")


if __name__ == "__main__":
    main()

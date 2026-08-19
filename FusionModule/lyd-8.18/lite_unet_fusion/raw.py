from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BLACK_SOURCE = 252.0
BLACK_DENOISED = 300.0
WHITE_LEVEL = 4095.0


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def pack_bayer(frame: np.ndarray) -> np.ndarray:
    return np.stack((frame[0::2, 0::2], frame[0::2, 1::2], frame[1::2, 0::2], frame[1::2, 1::2]))


def pack_scalar(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)[None]


def normalize(code: np.ndarray) -> np.ndarray:
    return np.clip((code - BLACK_DENOISED) / (WHITE_LEVEL - BLACK_DENOISED), 0.0, 1.0)


@dataclass(frozen=True)
class StreamPaths:
    source: Path
    two_d: Path
    three_d: Path


class FusionStream:
    """Memory-mapped, frame-aligned 16-bit Bayer streams."""

    def __init__(self, paths: StreamPaths, width: int = 1920, height: int = 1080) -> None:
        self.paths = paths
        self.width = width
        self.height = height
        frame_bytes = width * height * np.dtype("<u2").itemsize
        sizes = {name: path.stat().st_size for name, path in vars(paths).items()}
        counts = {name: size // frame_bytes for name, size in sizes.items()}
        if any(size % frame_bytes for size in sizes.values()) or len(set(counts.values())) != 1:
            raise ValueError(f"RAW streams require equal complete frame counts: {sizes}")
        self.frame_count = next(iter(counts.values()))
        shape = (self.frame_count, height, width)
        self.source = np.memmap(paths.source, dtype="<u2", mode="r", shape=shape)
        self.two_d = np.memmap(paths.two_d, dtype="<u2", mode="r", shape=shape)
        self.three_d = np.memmap(paths.three_d, dtype="<u2", mode="r", shape=shape)

    def _source_code(self, index: int) -> np.ndarray:
        # Source stores 12-bit samples left-aligned in uint16; outputs store code directly.
        return self.source[index].astype(np.float32) / 16.0 - BLACK_SOURCE + BLACK_DENOISED

    def features(self, index: int) -> dict[str, np.ndarray]:
        source = self._source_code(index)
        two_d = self.two_d[index].astype(np.float32)
        three_d = self.three_d[index].astype(np.float32)
        guidance = cv2.GaussianBlur(source, (0, 0), 1.5)
        motion_reference = cv2.GaussianBlur(source, (0, 0), 4.0)
        if index == 0:
            motion = np.zeros_like(source)
        else:
            previous = self._source_code(index - 1)
            previous = cv2.GaussianBlur(previous, (0, 0), 4.0)
            motion = np.abs(motion_reference - previous)
        gradient = np.abs(cv2.Sobel(guidance, cv2.CV_32F, 1, 0, ksize=3))
        gradient += np.abs(cv2.Sobel(guidance, cv2.CV_32F, 0, 1, ksize=3))
        motion = pack_scalar(sigmoid((motion - 5.0) / 2.0))
        flatness = pack_scalar(sigmoid((10.0 - gradient) / 3.0))
        source_packed = pack_bayer(source)
        two_packed = pack_bayer(two_d)
        three_packed = pack_bayer(three_d)
        agreement = sigmoid((12.0 - np.abs(two_packed - three_packed)) / 4.0)
        teacher = 0.35 * (1.0 - motion) * flatness * agreement
        return {
            "two_d": normalize(two_packed).astype(np.float32),
            "three_d": normalize(three_packed).astype(np.float32),
            "source": normalize(source_packed).astype(np.float32),
            "motion": motion.astype(np.float32),
            "flatness": flatness.astype(np.float32),
            "teacher": teacher.astype(np.float32),
        }

    def network_input(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        values = self.features(index)
        return np.concatenate((values["two_d"], values["three_d"], values["source"], values["motion"], values["flatness"])), values

    def output_code(self, index: int, beta: np.ndarray) -> np.ndarray:
        values = self.features(index)
        two = values["two_d"]
        three = values["three_d"]
        effective = 0.35 * beta * (1.0 - values["motion"]) * values["flatness"]
        packed = np.clip(two + effective * (three - two), 0.0, 1.0)
        packed = packed * (WHITE_LEVEL - BLACK_DENOISED) + BLACK_DENOISED
        output = np.empty((self.height, self.width), dtype=np.uint16)
        output[0::2, 0::2] = np.rint(packed[0]).astype(np.uint16)
        output[0::2, 1::2] = np.rint(packed[1]).astype(np.uint16)
        output[1::2, 0::2] = np.rint(packed[2]).astype(np.uint16)
        output[1::2, 1::2] = np.rint(packed[3]).astype(np.uint16)
        return output

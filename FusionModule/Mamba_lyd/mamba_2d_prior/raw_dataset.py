"""Direct reader for SigmaStar aligned source/2DNR/3DNR RAW streams."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RawStreamConfig:
    source: Path
    two_d: Path
    three_d: Path
    width: int = 1920
    height: int = 1080
    cfa_pattern: str = "RGGB"
    # The darkroom source is a 12-bit RAW value left-aligned in a uint16
    # container. denoised/fused are already direct 12-bit code values.
    source_container_scale: float = 16.0
    source_black_level: float = 252.0
    denoised_black_level: float = 300.0
    white_level: float = 4095.0
    max_3dnr_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.cfa_pattern.upper() != "RGGB":
            raise ValueError("This adapter uses the physical RGGB [R,G1,G2,B] packing")
        if self.source_container_scale <= 0.0:
            raise ValueError("source_container_scale must be positive")
        if not self.white_level > self.denoised_black_level:
            raise ValueError("white_level must exceed denoised_black_level")


@dataclass(frozen=True)
class SequenceSplit:
    train_frames: list[int]
    validation_frames: list[int]
    test_frames: list[int]
    guard_frames: list[int]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def _pack(frame: np.ndarray) -> np.ndarray:
    return np.stack((frame[0::2, 0::2], frame[0::2, 1::2], frame[1::2, 0::2], frame[1::2, 1::2]))


def _blur_planes(values: np.ndarray, sigma: float) -> np.ndarray:
    """Blur each physical Bayer plane without mixing R/G/B samples."""
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError(f"Expected [4,H,W] Bayer planes, got {values.shape}")
    return np.stack([cv2.GaussianBlur(plane, (0, 0), sigma) for plane in values])


def _gradient_planes(values: np.ndarray) -> np.ndarray:
    return np.stack([
        np.abs(cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3))
        + np.abs(cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3))
        for plane in values
    ])


class RawFusionStream:
    """Memory maps aligned streams and creates physically meaningful features."""

    def __init__(self, config: RawStreamConfig) -> None:
        self.config = config
        self.source = _RawFrameReader(config.source, config.width, config.height)
        self.two_d = _RawFrameReader(config.two_d, config.width, config.height)
        self.three_d = _RawFrameReader(config.three_d, config.width, config.height)
        counts = {"source": len(self.source), "two_d": len(self.two_d), "three_d": len(self.three_d)}
        if len(set(counts.values())) != 1:
            raise ValueError(f"RAW inputs must contain the same frame count: {counts}")
        self.frame_count = next(iter(counts.values()))
        self.packed_height = config.height // 2
        self.packed_width = config.width // 2
        self.prior_cache: PriorMemmapCache | None = None

    def enable_prior_cache(self, cache_dir: Path, rebuild: bool = False) -> "PriorMemmapCache":
        """Build or open a validated full-frame prior memmap for this stream."""
        self.prior_cache = PriorMemmapCache.open_or_build(self, cache_dir, rebuild=rebuild)
        return self.prior_cache

    def _frame_planes(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source = self._source_code(index)
        previous = self._source_code(max(0, index - 1))
        return _pack(previous), _pack(source), _pack(self.two_d[index].astype(np.float32)), _pack(self.three_d[index].astype(np.float32))

    def _compute_priors_uncached(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the same four priors used by the original patch path."""
        previous4, current4, two4, three4 = self._frame_planes(index)
        motion4 = _sigmoid((np.abs(_blur_planes(current4, 4.0) - _blur_planes(previous4, 4.0)) - 5.0) / 2.0)
        flatness4 = _sigmoid((10.0 - _gradient_planes(_blur_planes(current4, 1.5))) / 3.0)
        agreement4 = _sigmoid((12.0 - np.abs(two4 - three4)) / 4.0)
        teacher_w3 = self.config.max_3dnr_weight * (1.0 - motion4) * flatness4 * agreement4
        return motion4.astype(np.float32), flatness4.astype(np.float32), agreement4.astype(np.float32), teacher_w3.astype(np.float32)

    def _source_code(self, index: int) -> np.ndarray:
        return self._source_code_from_raw(self.source[index])

    def _source_code_from_raw(self, value: np.ndarray) -> np.ndarray:
        value = value.astype(np.float32) / self.config.source_container_scale
        return value - self.config.source_black_level + self.config.denoised_black_level

    @staticmethod
    def _pack_region(reader: _RawFrameReader, index: int, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        """Read an RGGB-aligned region directly, without materializing a full frame."""
        raw = reader[index]
        return _pack(raw[2 * top : 2 * bottom, 2 * left : 2 * right])

    def _normalize(self, value: np.ndarray) -> np.ndarray:
        return np.clip((value - self.config.denoised_black_level) / (self.config.white_level - self.config.denoised_black_level), 0.0, 1.0)

    def sample(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame index {index} outside [0, {self.frame_count})")
        previous4, current4, two4, three4 = self._frame_planes(index)
        if self.prior_cache is None:
            motion4, flatness4, agreement4, teacher_w3 = self._compute_priors_uncached(index)
        else:
            motion4, flatness4, agreement4, teacher_w3 = self.prior_cache.frame(index)
        return {
            "prev4": self._normalize(previous4).astype(np.float32),
            "curr4": self._normalize(current4).astype(np.float32),
            "dnr2_4": self._normalize(two4).astype(np.float32),
            "dnr3_4": self._normalize(three4).astype(np.float32),
            "motion": motion4.astype(np.float32),
            "flatness": flatness4.astype(np.float32),
            "agreement": agreement4.astype(np.float32),
            "teacher_w3": teacher_w3.astype(np.float32),
        }

    def sample_patch(self, index: int, top: int, left: int, patch_size: int) -> dict[str, np.ndarray]:
        """Create one patch before feature extraction to keep the input pipeline CPU-efficient.

        The margin covers four sigma of the motion blur.  Features in the returned
        central patch therefore match full-frame processing to negligible Gaussian-tail error.
        """
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame index {index} outside [0, {self.frame_count})")
        if patch_size <= 0 or not 0 <= top <= self.packed_height - patch_size or not 0 <= left <= self.packed_width - patch_size:
            raise ValueError("Patch is outside the packed Bayer frame")

        margin = 16
        region_top = max(0, top - margin)
        region_bottom = min(self.packed_height, top + patch_size + margin)
        region_left = max(0, left - margin)
        region_right = min(self.packed_width, left + patch_size + margin)

        current4 = self._source_code_from_raw(self._pack_region(self.source, index, region_top, region_bottom, region_left, region_right))
        previous4 = self._source_code_from_raw(self._pack_region(self.source, max(0, index - 1), region_top, region_bottom, region_left, region_right))
        two4 = self._pack_region(self.two_d, index, region_top, region_bottom, region_left, region_right).astype(np.float32)
        three4 = self._pack_region(self.three_d, index, region_top, region_bottom, region_left, region_right).astype(np.float32)

        crop_top, crop_left = top - region_top, left - region_left
        crop = np.s_[..., crop_top : crop_top + patch_size, crop_left : crop_left + patch_size]
        if self.prior_cache is None:
            motion4 = _sigmoid((np.abs(_blur_planes(current4, 4.0) - _blur_planes(previous4, 4.0)) - 5.0) / 2.0)
            flatness4 = _sigmoid((10.0 - _gradient_planes(_blur_planes(current4, 1.5))) / 3.0)
            agreement4 = _sigmoid((12.0 - np.abs(two4 - three4)) / 4.0)
            teacher_w3 = self.config.max_3dnr_weight * (1.0 - motion4) * flatness4 * agreement4
            motion_patch, flatness_patch, agreement_patch, teacher_patch = motion4[crop], flatness4[crop], agreement4[crop], teacher_w3[crop]
        else:
            motion_patch, flatness_patch, agreement_patch, teacher_patch = self.prior_cache.patch(index, top, left, patch_size)
        return {
            "prev4": self._normalize(previous4[crop]).astype(np.float32),
            "curr4": self._normalize(current4[crop]).astype(np.float32),
            "dnr2_4": self._normalize(two4[crop]).astype(np.float32),
            "dnr3_4": self._normalize(three4[crop]).astype(np.float32),
            "motion": motion_patch.astype(np.float32),
            "flatness": flatness_patch.astype(np.float32),
            "agreement": agreement_patch.astype(np.float32),
            "teacher_w3": teacher_patch.astype(np.float32),
        }

    def unpack_to_codes(self, packed: np.ndarray) -> np.ndarray:
        codes = np.clip(
            np.rint(packed * (self.config.white_level - self.config.denoised_black_level) + self.config.denoised_black_level),
            0,
            self.config.white_level,
        ).astype(np.uint16)
        output = np.empty((self.config.height, self.config.width), dtype=np.uint16)
        output[0::2, 0::2], output[0::2, 1::2] = codes[0], codes[1]
        output[1::2, 0::2], output[1::2, 1::2] = codes[2], codes[3]
        return output


class _RawFrameReader:
    """Read either one concatenated stream or an out_XXXX.raw frame directory."""

    def __init__(self, input_path: Path, width: int, height: int) -> None:
        self.input_path = input_path.resolve()
        self.width, self.height = width, height
        self.frame_bytes = width * height * np.dtype("<u2").itemsize
        if input_path.is_file():
            count, remainder = divmod(input_path.stat().st_size, self.frame_bytes)
            if remainder:
                raise ValueError(f"{input_path} has {remainder} non-frame-aligned trailing bytes")
            if count <= 0:
                raise ValueError(f"{input_path} contains no frames")
            self.stream = np.memmap(input_path, dtype="<u2", mode="r", shape=(count, height, width))
            self.frames: list[Path] | None = None
        elif input_path.is_dir():
            frames = sorted(input_path.glob("out_*.raw"))
            if not frames:
                raise ValueError(f"{input_path} must contain out_XXXX.raw files")
            invalid = [path for path in frames if path.stat().st_size != self.frame_bytes]
            if invalid:
                raise ValueError(f"Unexpected frame size in {invalid[0]}")
            self.stream = None
            self.frames = frames
        else:
            raise FileNotFoundError(f"RAW input not found: {input_path}")

    def __len__(self) -> int:
        return len(self.frames) if self.frames is not None else self.stream.shape[0]  # type: ignore[union-attr]

    def __getitem__(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.frames is None:
            return self.stream[index]  # type: ignore[index]
        return np.fromfile(self.frames[index], dtype="<u2").reshape(self.height, self.width)


class PriorMemmapCache:
    """Validated, read-only memmap containing full-frame Bayer priors."""

    VERSION = 1
    PRIOR_NAMES = ("motion", "flatness", "agreement", "teacher_w3")

    def __init__(self, cache_dir: Path, shape: tuple[int, int, int, int, int]) -> None:
        self.cache_dir = cache_dir
        self.data_path = cache_dir / "priors.float32.dat"
        self.shape = shape
        self._data: np.memmap | None = None

    def _open(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.data_path, dtype="<f4", mode="r", shape=self.shape)
        return self._data

    def __getstate__(self) -> dict[str, object]:
        # DataLoader workers reopen the mapping lazily after process spawning.
        return {"cache_dir": self.cache_dir, "data_path": self.data_path, "shape": self.shape}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.cache_dir = state["cache_dir"]  # type: ignore[assignment]
        self.data_path = state["data_path"]  # type: ignore[assignment]
        self.shape = state["shape"]  # type: ignore[assignment]
        self._data = None

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        values = self._open()[index]
        return tuple(values[position] for position in range(4))  # type: ignore[return-value]

    def patch(self, index: int, top: int, left: int, patch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if patch_size <= 0 or top < 0 or left < 0 or top + patch_size > self.shape[-2] or left + patch_size > self.shape[-1]:
            raise ValueError("Requested prior patch is outside the packed frame")
        values = self._open()[index, :, :, top : top + patch_size, left : left + patch_size]
        return tuple(values[position] for position in range(4))  # type: ignore[return-value]

    @classmethod
    def open_or_build(cls, stream: RawFusionStream, cache_dir: Path, rebuild: bool = False) -> "PriorMemmapCache":
        cache_dir = cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_dir / "metadata.json"
        shape = (stream.frame_count, 4, 4, stream.packed_height, stream.packed_width)
        expected = cls._metadata(stream, shape)
        if not rebuild and metadata_path.is_file() and (cache_dir / "priors.float32.dat").is_file():
            try:
                actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                actual = None
            if actual == expected:
                return cls(cache_dir, shape)

        cache_size_gib = int(np.prod(shape)) * np.dtype("<f4").itemsize / (1024 ** 3)
        print(f"Building prior memmap: {cache_dir} ({cache_size_gib:.2f} GiB)", flush=True)
        data_path = cache_dir / "priors.float32.dat"
        data = np.memmap(data_path, dtype="<f4", mode="w+", shape=shape)
        for index in range(stream.frame_count):
            data[index] = np.stack(stream._compute_priors_uncached(index), axis=0)
            if index == 0 or (index + 1) % max(1, stream.frame_count // 10) == 0 or index + 1 == stream.frame_count:
                print(f"Prior cache {index + 1}/{stream.frame_count}", flush=True)
        data.flush()
        del data
        metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        return cls(cache_dir, shape)

    @classmethod
    def _metadata(cls, stream: RawFusionStream, shape: tuple[int, int, int, int, int]) -> dict[str, object]:
        files = {}
        for name, reader in (("source", stream.source), ("two_d", stream.two_d), ("three_d", stream.three_d)):
            stat = reader.input_path.stat()
            files[name] = {"path": str(reader.input_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        config = stream.config
        return {
            "version": cls.VERSION,
            "dtype": "float32",
            "shape": list(shape),
            "files": files,
            "width": config.width,
            "height": config.height,
            "source_black_level": config.source_black_level,
            "denoised_black_level": config.denoised_black_level,
            "source_container_scale": config.source_container_scale,
            "max_3dnr_weight": config.max_3dnr_weight,
        }

def split_sequence(
    frame_count: int,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    guard_frames: int = 10,
) -> SequenceSplit:
    """Create ordered train/guard/validation/test regions without frame shuffling."""
    if frame_count < 12:
        raise ValueError("At least twelve frames are required for a train/guard/validation/test split")
    if not 0.0 < validation_fraction < 0.5 or not 0.0 < test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must be in (0, 0.5)")
    validation_count = max(1, round(frame_count * validation_fraction))
    test_count = max(1, round(frame_count * test_fraction))
    train_stop = frame_count - validation_count - test_count - guard_frames
    if train_stop < 1:
        raise ValueError("Not enough frames after applying validation/test fractions and guard frames")
    validation_start = train_stop + guard_frames
    test_start = validation_start + validation_count
    return SequenceSplit(
        train_frames=list(range(train_stop)),
        validation_frames=list(range(validation_start, test_start)),
        test_frames=list(range(test_start, frame_count)),
        guard_frames=list(range(train_stop, validation_start)),
    )


class RandomPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, stream: RawFusionStream, frames: list[int], patch_size: int, samples: int, seed: int) -> None:
        if patch_size % 8:
            raise ValueError("patch_size must be divisible by 8")
        self.stream, self.frames, self.patch_size, self.samples, self.seed = stream, frames, patch_size, samples, seed
        if not frames:
            raise ValueError("Dataset has no frames")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        frame = self.frames[int(rng.integers(0, len(self.frames)))]
        top = int(rng.integers(0, self.stream.packed_height - self.patch_size + 1))
        left = int(rng.integers(0, self.stream.packed_width - self.patch_size + 1))
        values = self.stream.sample_patch(frame, top, left, self.patch_size)
        return {key: torch.from_numpy(value) for key, value in values.items()}

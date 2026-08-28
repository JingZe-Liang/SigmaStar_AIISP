from __future__ import annotations

import math
import multiprocessing as mp
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


WIDTH, HEIGHT, FRAME_COUNT, CODE_MAX = 1920, 1080, 200, 4095
SOURCE_BLACK_LEVEL, NR_BLACK_LEVEL = 252, 300
FRAME_BYTES = WIDTH * HEIGHT * np.dtype("<u2").itemsize
FRAME_PATTERN = re.compile(r"out_(\d{4})\.raw$", re.IGNORECASE)


@dataclass(frozen=True)
class SequenceStatistics:
    source_to_dnr_offset: tuple[float, float, float, float]
    noise_sigma: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "source_to_dnr_offset": list(self.source_to_dnr_offset),
            "noise_sigma": list(self.noise_sigma),
        }


@dataclass(frozen=True)
class SequenceData:
    name: str
    source_path: Path
    dnr2_paths: tuple[Path, ...]
    dnr3_paths: tuple[Path, ...]
    motion_paths: tuple[Path, ...] | None
    source: np.memmap
    cfa_pattern: str
    source_black_level: int
    dnr_black_level: int
    white_level: int


def raw_to_linear(raw: np.ndarray, black_level: int, white_level: int = CODE_MAX) -> np.ndarray:
    denominator = max(white_level - black_level, 1)
    return np.clip((raw.astype(np.float32) - black_level) / denominator, 0.0, 1.0)


def linear_to_nr_raw(
    linear: np.ndarray, black_level: int = NR_BLACK_LEVEL, white_level: int = CODE_MAX
) -> np.ndarray:
    return np.rint(
        black_level + np.clip(linear, 0.0, 1.0) * (white_level - black_level)
    ).astype("<u2")


def _labels(cfa_pattern: str) -> tuple[str, ...]:
    pattern = cfa_pattern.upper()
    if len(pattern) != 4 or pattern.count("R") != 1 or pattern.count("B") != 1 or pattern.count("G") != 2:
        raise ValueError(f"不支持的 CFA pattern: {cfa_pattern}")
    green_index = 0
    labels = []
    for color in pattern:
        if color == "G":
            green_index += 1
            labels.append(f"G{green_index}")
        else:
            labels.append(color)
    return tuple(labels)


def pack_bayer(mosaic: np.ndarray, cfa_pattern: str) -> np.ndarray:
    """Keep the packed helper for the separate alpha ablation tools."""
    if mosaic.ndim != 2 or mosaic.shape[0] % 2 or mosaic.shape[1] % 2:
        raise ValueError("Bayer mosaic 必须是偶数尺寸二维数组")
    positions = (
        mosaic[0::2, 0::2],
        mosaic[0::2, 1::2],
        mosaic[1::2, 0::2],
        mosaic[1::2, 1::2],
    )
    physical = dict(zip(_labels(cfa_pattern), positions, strict=True))
    return np.stack([physical[label] for label in ("R", "G1", "G2", "B")], axis=0)


def unpack_bayer(packed: np.ndarray, cfa_pattern: str) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"packed Bayer 必须为 [4,H,W]，实际为 {packed.shape}")
    physical = dict(zip(("R", "G1", "G2", "B"), packed, strict=True))
    positions = [physical[label] for label in _labels(cfa_pattern)]
    mosaic = np.empty((packed.shape[1] * 2, packed.shape[2] * 2), dtype=packed.dtype)
    for index, plane in enumerate(positions):
        row, column = divmod(index, 2)
        mosaic[row::2, column::2] = plane
    return mosaic


def _single_raw(directory: Path) -> Path:
    paths = sorted(directory.glob("*.raw"))
    if len(paths) != 1:
        raise FileNotFoundError(f"{directory} 应有且仅有一个 source RAW，实际为 {len(paths)} 个")
    return paths[0]


def _frame_paths(directory: Path) -> tuple[Path, ...]:
    indexed = []
    for path in directory.glob("out_*.raw"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(FRAME_COUNT)):
        raise ValueError(f"{directory} 必须连续包含 out_0000.raw 到 out_0199.raw")
    return tuple(path for _, path in indexed)


def _motion_paths(cache_root: Path, name: str) -> tuple[Path, ...]:
    paths = tuple(cache_root / name / "masks" / f"{index:04d}.png" for index in range(FRAME_COUNT))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{name} 的 MD 缓存不完整，缺少: {missing[0]}")
    return paths


def discover_sequences(
    data_root: Path,
    names: tuple[str, ...],
    motion_cache_root: Path | None = None,
    cfa_pattern: str = "RGGB",
    source_black_level: int = SOURCE_BLACK_LEVEL,
    dnr_black_level: int = NR_BLACK_LEVEL,
    white_level: int = CODE_MAX,
) -> tuple[SequenceData, ...]:
    _labels(cfa_pattern)
    source_root = data_root / "Sigmastar_7_30" / "shdarkroom"
    candidate_root = data_root / "mis20s1_2D&3D"
    sequences = []
    for name in names:
        source_path = _single_raw(source_root / name)
        candidate_dirs = [
            path
            for path in candidate_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{name}")
        ]
        if len(candidate_dirs) != 1:
            raise FileNotFoundError(f"{name} 未唯一匹配 2D/3DNR 目录")
        if source_path.stat().st_size != FRAME_COUNT * FRAME_BYTES:
            raise ValueError(f"source RAW 大小错误: {source_path}")
        dnr2_paths = _frame_paths(candidate_dirs[0] / "denoised")
        dnr3_paths = _frame_paths(candidate_dirs[0] / "fused")
        if any(path.stat().st_size != FRAME_BYTES for path in (*dnr2_paths, *dnr3_paths)):
            raise ValueError(f"{name} 候选 RAW 大小错误")
        source = np.memmap(source_path, dtype="<u2", mode="r", shape=(FRAME_COUNT, HEIGHT, WIDTH))
        if np.any(np.asarray(source[::32, ::128, ::128]) & 15):
            raise ValueError(f"source RAW 不是预期的 12-bit 左对齐格式: {source_path}")
        motion_paths = None if motion_cache_root is None else _motion_paths(motion_cache_root, name)
        sequences.append(
            SequenceData(
                name,
                source_path,
                dnr2_paths,
                dnr3_paths,
                motion_paths,
                source,
                cfa_pattern.upper(),
                source_black_level,
                dnr_black_level,
                white_level,
            )
        )
    return tuple(sequences)


@lru_cache(maxsize=64)
def _candidate_memmap(path_text: str) -> np.memmap:
    """Keep a bounded per-worker cache to avoid reopening RAW files every sample."""
    return np.memmap(path_text, dtype="<u2", mode="r", shape=(HEIGHT, WIDTH))


def read_candidate(path: Path) -> np.ndarray:
    return np.asarray(_candidate_memmap(str(path))).copy()


def read_candidate_crop(path: Path, top: int, left: int, size: int) -> np.ndarray:
    mapped = _candidate_memmap(str(path))
    return np.asarray(mapped[top : top + size, left : left + size]).copy()


def read_source(sequence: SequenceData, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, FRAME_COUNT - 1))
    return (np.asarray(sequence.source[index]) >> 4).astype(np.uint16, copy=False)


def read_source_crop(sequence: SequenceData, index: int, top: int, left: int, size: int) -> np.ndarray:
    index = int(np.clip(index, 0, FRAME_COUNT - 1))
    return np.asarray(sequence.source[index, top : top + size, left : left + size]).astype(np.uint16) >> 4


def read_motion(sequence: SequenceData, index: int) -> np.ndarray:
    if sequence.motion_paths is None:
        raise ValueError("当前序列没有配置训练用 motion cache")
    mask = cv2.imread(str(sequence.motion_paths[index]), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (HEIGHT // 2, WIDTH // 2):
        raise ValueError(f"MD mask 格式错误: {sequence.motion_paths[index]}")
    return mask.astype(np.float32) / 255.0


def read_motion_mosaic(sequence: SequenceData, index: int) -> np.ndarray:
    packed_mask = read_motion(sequence, index)
    return cv2.resize(packed_mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def _cfa_index_map(cfa_pattern: str) -> np.ndarray:
    labels = _labels(cfa_pattern)
    physical = {label: index for index, label in enumerate(("R", "G1", "G2", "B"))}
    return np.asarray(
        [[physical[label] for label in labels[0:2]], [physical[label] for label in labels[2:4]]],
        dtype=np.int64,
    )


def _expand_cfa_values(values: Sequence[float], cfa_pattern: str) -> np.ndarray:
    phase = _cfa_index_map(cfa_pattern)
    value_map = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    for row in range(2):
        for column in range(2):
            value_map[row::2, column::2] = values[int(phase[row, column])]
    return value_map


def estimate_sequence_statistics(
    sequence: SequenceData,
    *,
    start_frame: int = 3,
    stop_frame: int = FRAME_COUNT - 3,
    sample_frames: int = 8,
    spatial_step: int = 8,
) -> SequenceStatistics:
    indices = np.linspace(
        start_frame,
        max(start_frame, stop_frame - 1),
        min(sample_frames, max(stop_frame - start_frame, 1)),
        dtype=int,
    )
    offsets: list[list[np.ndarray]] = [[], [], [], []]
    temporal: list[list[np.ndarray]] = [[], [], [], []]
    phase = _cfa_index_map(sequence.cfa_pattern)
    for frame_index in sorted(set(indices.tolist())):
        source = read_source(sequence, frame_index).astype(np.float32)
        previous = read_source(sequence, frame_index - 1).astype(np.float32)
        dnr = read_candidate(sequence.dnr2_paths[frame_index]).astype(np.float32)
        source_signal = source - sequence.source_black_level
        dnr_signal = dnr - sequence.dnr_black_level
        for channel in range(4):
            row, column = np.argwhere(phase == channel)[0]
            values_source = source_signal[row::2, column::2][::spatial_step, ::spatial_step]
            values_dnr = dnr_signal[row::2, column::2][::spatial_step, ::spatial_step]
            values_temporal = (source_signal - (previous - sequence.source_black_level))[row::2, column::2]
            offsets[channel].append((values_dnr - values_source).reshape(-1))
            temporal[channel].append(values_temporal[::spatial_step, ::spatial_step].reshape(-1))
    offset_values = [np.concatenate(values) for values in offsets]
    temporal_values = [np.concatenate(values) for values in temporal]
    offset = np.asarray([np.median(values) for values in offset_values], dtype=np.float32)
    sigma = []
    for values in temporal_values:
        centered = values - np.median(values)
        sigma.append(max(float(np.median(np.abs(centered)) / (0.67448975 * math.sqrt(2.0))), 1.0))
    return SequenceStatistics(tuple(float(value) for value in offset), tuple(float(value) for value in sigma))


def _linear_candidate(sequence: SequenceData, path: Path) -> np.ndarray:
    return raw_to_linear(read_candidate(path), sequence.dnr_black_level, sequence.white_level)


def _linear_source(sequence: SequenceData, index: int) -> np.ndarray:
    return raw_to_linear(read_source(sequence, index), sequence.source_black_level, sequence.white_level)


def _linear_candidate_crop(sequence: SequenceData, path: Path, top: int, left: int, size: int) -> np.ndarray:
    return raw_to_linear(
        read_candidate_crop(path, top, left, size),
        sequence.dnr_black_level,
        sequence.white_level,
    )


def _linear_source_crop(sequence: SequenceData, index: int, top: int, left: int, size: int) -> np.ndarray:
    return raw_to_linear(
        read_source_crop(sequence, index, top, left, size),
        sequence.source_black_level,
        sequence.white_level,
    )


class WeakFusionDataset(Dataset):
    """Full-resolution Bayer patches for NAF-BPN weakly supervised fine-tuning."""

    def __init__(
        self,
        sequences: Sequence[SequenceData],
        frame_indices: Sequence[int],
        samples: int,
        patch_size: int,
        seed: int,
        *,
        proxy_radius: int = 3,
        training: bool = True,
        motion_dropout: float = 0.2,
        motion_jitter: bool = True,
    ) -> None:
        if not sequences or not frame_indices or samples <= 0:
            raise ValueError("弱监督数据集需要非空序列、帧列表和 samples")
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("patch_size 必须为正偶数")
        if min(frame_indices) < proxy_radius or max(frame_indices) >= FRAME_COUNT - proxy_radius:
            raise ValueError("训练帧超出七帧 proxy 可用范围")
        if any(sequence.motion_paths is None for sequence in sequences):
            raise ValueError("弱监督训练必须配置离线 motion target")
        self.sequences = tuple(sequences)
        self.frame_indices = tuple(int(value) for value in frame_indices)
        self.samples = int(samples)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.proxy_radius = int(proxy_radius)
        self.training = training
        self.motion_dropout = float(motion_dropout)
        self.motion_jitter = bool(motion_jitter)
        self.epoch = 0
        self.statistics = {
            sequence.name: estimate_sequence_statistics(sequence) for sequence in self.sequences
        }
        self.offset_maps = {
            sequence.name: _expand_cfa_values(
                self.statistics[sequence.name].source_to_dnr_offset,
                sequence.cfa_pattern,
            )
            for sequence in self.sequences
        }
        self.sigma_maps = {
            sequence.name: _expand_cfa_values(
                self.statistics[sequence.name].noise_sigma,
                sequence.cfa_pattern,
            )
            for sequence in self.sequences
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    def _rng(self, index: int) -> np.random.Generator:
        epoch = self.epoch if self.training else 0
        return np.random.default_rng(self.seed + epoch * self.samples + index)

    def _crop_origin(self, rng: np.random.Generator, packed_motion: np.ndarray) -> tuple[int, int]:
        max_top = HEIGHT - self.patch_size
        max_left = WIDTH - self.patch_size
        if not self.training:
            return max_top // 2, max_left // 2
        if rng.random() < 0.45 and packed_motion.max() > 0:
            candidates = np.column_stack(np.nonzero(packed_motion > 0.5))
            center_y, center_x = candidates[int(rng.integers(0, len(candidates)))]
            packed_size = self.patch_size // 2
            top = 2 * int(
                np.clip(center_y - rng.integers(0, packed_size), 0, max_top // 2)
            )
            left = 2 * int(
                np.clip(center_x - rng.integers(0, packed_size), 0, max_left // 2)
            )
        else:
            top = int(rng.integers(0, max_top + 1))
            left = int(rng.integers(0, max_left + 1))
        return top - top % 2, left - left % 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        sequence = self.sequences[int(rng.integers(0, len(self.sequences)))]
        frame_index = self.frame_indices[int(rng.integers(0, len(self.frame_indices)))]
        packed_motion = read_motion(sequence, frame_index)
        top, left = self._crop_origin(rng, packed_motion)
        bottom, right = top + self.patch_size, left + self.patch_size
        crop = np.s_[top:bottom, left:right]
        stats = self.statistics[sequence.name]
        offset_map = self.offset_maps[sequence.name][crop]
        sigma_map = self.sigma_maps[sequence.name][crop]
        source_window = np.stack(
            [
                _linear_source_crop(
                    sequence,
                    frame_index + delta,
                    top,
                    left,
                    self.patch_size,
                )
                for delta in range(-self.proxy_radius, self.proxy_radius + 1)
            ],
            axis=0,
        )
        source_window = np.clip(
            source_window + offset_map[None] / max(sequence.white_level - sequence.source_black_level, 1),
            0.0,
            1.0,
        )
        sorted_window = np.sort(source_window, axis=0)
        proxy = sorted_window[1:-1].mean(axis=0)
        temporal_range = source_window.max(axis=0) - source_window.min(axis=0)
        center = self.proxy_radius
        temporal_difference = np.maximum(
            np.abs(source_window[center] - source_window[center - 1]),
            np.abs(source_window[center] - source_window[center + 1]),
        )
        current = source_window[center]
        valid_signal = (
            (current > 8.0 / sequence.white_level)
            & (current < 4090.0 / sequence.white_level)
        ).astype(np.float32)
        packed_crop = packed_motion[top // 2 : bottom // 2, left // 2 : right // 2]
        motion = np.repeat(np.repeat(packed_crop, 2, axis=0), 2, axis=1)[None].astype(np.float32)
        if self.training and self.motion_jitter:
            operation = int(rng.integers(0, 4))
            if operation == 1:
                motion = cv2.dilate(motion[0], np.ones((3, 3), np.uint8))[None]
            elif operation == 2:
                motion = cv2.erode(motion[0], np.ones((3, 3), np.uint8))[None]
            elif operation == 3:
                motion = cv2.GaussianBlur(motion[0], (3, 3), 0)[None]
        if self.training and rng.random() < self.motion_dropout:
            motion[...] = 0.0

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

        return {
            "image_2dnr": tensor(_linear_candidate_crop(sequence, sequence.dnr2_paths[frame_index], top, left, self.patch_size)[None]),
            "image_3dnr": tensor(_linear_candidate_crop(sequence, sequence.dnr3_paths[frame_index], top, left, self.patch_size)[None]),
            "noisy_current": tensor(current[None]),
            "noisy_previous": tensor(source_window[center - 1][None]),
            "proxy": tensor(proxy[None]),
            "temporal_difference": tensor(temporal_difference[None]),
            "temporal_range": tensor(temporal_range[None]),
            "valid_signal": tensor(valid_signal[None]),
            "motion_target": tensor(np.clip(motion, 0.0, 1.0)),
            "noise_sigma": tensor((sigma_map / max(sequence.white_level - sequence.source_black_level, 1))[None]),
            "sequence_id": sequence.name,
            "frame_index": frame_index,
        }


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def stage1_h5_split(root_dir: str | Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Use the naturally last H5 shard from every scene as the validation set."""
    root = Path(root_dir)
    files = tuple(sorted(root.rglob("*.h5"), key=lambda item: _natural_key(str(item.relative_to(root)))))
    if not files:
        raise FileNotFoundError(f"未在 {root} 及其子目录中找到 .h5 文件")
    scenes: dict[str, list[Path]] = {}
    for path in files:
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(f"Stage 1 H5 必须位于 scene 子目录: {path}")
        scenes.setdefault(relative.parts[0], []).append(path)

    train_files: list[Path] = []
    validation_files: list[Path] = []
    for scene, scene_files in sorted(scenes.items(), key=lambda item: _natural_key(item[0])):
        ordered = sorted(scene_files, key=lambda item: _natural_key(item.name))
        if len(ordered) < 2:
            raise ValueError(f"{scene} 至少需要两个 H5 文件，才能保留最后一个作为验证集")
        train_files.extend(ordered[:-1])
        validation_files.append(ordered[-1])
    return tuple(train_files), tuple(validation_files)


class CleanH5Dataset(Dataset):
    """Clean-GT Stage 1 dataset with a scene-local final-shard validation split."""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int | None = 256,
        *,
        split: str,
        seed: int,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"未知 Stage 1 split: {split}")
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.split = split
        self.training = split == "train"
        self.seed = int(seed)
        train_files, validation_files = stage1_h5_split(self.root_dir)
        self.h5_files = train_files if self.training else validation_files
        self._epoch = mp.Value("q", 0)
        self.samples: list[tuple[Path, int]] = []
        for path in self.h5_files:
            with h5py.File(path, "r") as handle:
                required = {"2dnr", "3dnr", "clean", "noisy"}
                missing = required - set(handle.keys())
                if missing:
                    raise KeyError(f"{path} 缺少 H5 键: {sorted(missing)}")
                length = int(handle["clean"].shape[0])
                if handle["noisy"].shape[0] != length or handle["noisy"].shape[1] < 2:
                    raise ValueError(f"{path} 的 noisy 数据维度不符合 [N,2,H,W]")
                self.samples.extend((path, frame_index) for frame_index in range(length))
        if not self.samples:
            raise ValueError(f"Stage 1 {split} split 没有可用样本")

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> np.random.Generator:
        with self._epoch.get_lock():
            epoch = self._epoch.value
        return np.random.default_rng(self.seed + epoch * max(len(self.samples), 1) + int(index))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, frame_index = self.samples[index]
        with h5py.File(path, "r") as handle:
            height, width = handle["clean"].shape[-2:]
            if self.patch_size is None:
                crop = np.s_[:, :]
            else:
                if height < self.patch_size or width < self.patch_size:
                    raise ValueError("Stage 1 patch_size 大于 H5 图像尺寸")
            if self.patch_size is not None and self.training:
                rng = self._rng(index)
                top = int(rng.integers(0, height - self.patch_size + 1))
                left = int(rng.integers(0, width - self.patch_size + 1))
                crop = np.s_[top : top + self.patch_size, left : left + self.patch_size]
            elif self.patch_size is not None:
                top, left = (height - self.patch_size) // 2, (width - self.patch_size) // 2
                crop = np.s_[top : top + self.patch_size, left : left + self.patch_size]
            dnr2 = handle["2dnr"][frame_index, crop[0], crop[1]].astype(np.float32) / CODE_MAX
            dnr3 = handle["3dnr"][frame_index, crop[0], crop[1]].astype(np.float32) / CODE_MAX
            clean = handle["clean"][frame_index, crop[0], crop[1]].astype(np.float32) / CODE_MAX
            current = handle["noisy"][frame_index, 1, crop[0], crop[1]].astype(np.float32) / CODE_MAX
            previous = handle["noisy"][max(frame_index - 1, 0), 1, crop[0], crop[1]].astype(np.float32) / CODE_MAX

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array[None], dtype=np.float32))

        return {
            "image_2dnr": tensor(dnr2),
            "image_3dnr": tensor(dnr3),
            "noisy_current": tensor(current),
            "noisy_previous": tensor(previous),
            "target": tensor(clean),
        }

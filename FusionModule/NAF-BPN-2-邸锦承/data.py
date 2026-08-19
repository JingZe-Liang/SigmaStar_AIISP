from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


WIDTH, HEIGHT, FRAME_COUNT, CODE_MAX = 1920, 1080, 200, 4095
SOURCE_BLACK_LEVEL = 252
NR_BLACK_LEVEL = 300
FRAME_BYTES = WIDTH * HEIGHT * np.dtype("<u2").itemsize
FRAME_PATTERN = re.compile(r"out_(\d{4})\.raw$", re.IGNORECASE)


@dataclass(frozen=True)
class SequenceData:
    name: str
    source_path: Path
    dnr2_paths: tuple[Path, ...]
    dnr3_paths: tuple[Path, ...]
    motion_paths: tuple[Path, ...] | None
    source: np.memmap


def raw_to_linear(raw: np.ndarray, black_level: int) -> np.ndarray:
    """12-bit 编码 RAW 转为统一的黑电平校正线性域 [0, 1]。"""
    return np.clip((raw.astype(np.float32) - black_level) / (CODE_MAX - black_level), 0.0, 1.0)


def linear_to_nr_raw(linear: np.ndarray) -> np.ndarray:
    """统一线性域转为供 2DNR/3DNR ISP 使用的 NR 黑电平编码域。"""
    encoded = NR_BLACK_LEVEL + np.clip(linear, 0.0, 1.0) * (CODE_MAX - NR_BLACK_LEVEL)
    return np.rint(encoded).astype("<u2")


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


def discover_sequences(data_root: Path, names: tuple[str, ...], motion_cache_root: Path | None = None) -> tuple[SequenceData, ...]:
    source_root = data_root / "Sigmastar_7_30" / "shdarkroom"
    candidate_root = data_root / "mis20s1_2D&3D"
    sequences = []
    for name in names:
        source_path = _single_raw(source_root / name)
        candidate_dirs = [path for path in candidate_root.iterdir() if path.is_dir() and path.name.endswith(f"_{name}")]
        if len(candidate_dirs) != 1:
            raise FileNotFoundError(f"{name} 未唯一匹配 2D/3DNR 目录")
        if source_path.stat().st_size != FRAME_COUNT * FRAME_BYTES:
            raise ValueError(f"source RAW 大小错误: {source_path}")
        dnr2_paths = _frame_paths(candidate_dirs[0] / "denoised")
        dnr3_paths = _frame_paths(candidate_dirs[0] / "fused")
        for path in (*dnr2_paths, *dnr3_paths):
            if path.stat().st_size != FRAME_BYTES:
                raise ValueError(f"候选 RAW 大小错误: {path}")
        source = np.memmap(source_path, dtype="<u2", mode="r", shape=(FRAME_COUNT, HEIGHT, WIDTH))
        if np.any(np.asarray(source[::32, ::128, ::128]) & 15):
            raise ValueError(f"source RAW 不是预期的 12-bit 左对齐格式: {source_path}")
        motion_paths = None if motion_cache_root is None else _motion_paths(motion_cache_root, name)
        sequences.append(SequenceData(name, source_path, dnr2_paths, dnr3_paths, motion_paths, source))
    return tuple(sequences)


def read_candidate(path: Path) -> np.ndarray:
    return np.asarray(np.memmap(path, dtype="<u2", mode="r", shape=(HEIGHT, WIDTH))).copy()


def read_source(sequence: SequenceData, index: int) -> np.ndarray:
    return (np.asarray(sequence.source[index]) >> 4).astype(np.uint16, copy=False)


def read_motion(sequence: SequenceData, index: int) -> np.ndarray:
    if sequence.motion_paths is None:
        raise RuntimeError("当前 SequenceData 未配置 MD 缓存")
    mask = cv2.imread(str(sequence.motion_paths[index]), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (HEIGHT // 2, WIDTH // 2):
        raise ValueError(f"MD mask 格式错误: {sequence.motion_paths[index]}")
    return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1).astype(np.float32) / 255.0


class FullMosaicMotionDataset(Dataset):
    """随机完整 Bayer crop，所有图像均已转到统一线性域。"""

    def __init__(self, sequences: tuple[SequenceData, ...], frame_indices: tuple[int, ...], samples: int, patch_size: int, seed: int):
        if patch_size % 8 != 0:
            raise ValueError("patch_size 必须是 8 的倍数")
        if not frame_indices or min(frame_indices) < 0 or max(frame_indices) >= FRAME_COUNT:
            raise ValueError("训练/验证帧必须位于 0~199")
        if any(sequence.motion_paths is None for sequence in sequences):
            raise ValueError("训练数据必须已绑定完整 MD 缓存")
        self.sequences = sequences
        self.frame_indices = frame_indices
        self.samples = samples
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * self.samples + index)
        sequence = self.sequences[int(rng.integers(len(self.sequences)))]
        frame_index = int(self.frame_indices[int(rng.integers(len(self.frame_indices)))])
        top = int(rng.integers(0, HEIGHT - self.patch_size + 1))
        left = int(rng.integers(0, WIDTH - self.patch_size + 1))
        images = (
            raw_to_linear(read_candidate(sequence.dnr2_paths[frame_index]), NR_BLACK_LEVEL),
            raw_to_linear(read_candidate(sequence.dnr3_paths[frame_index]), NR_BLACK_LEVEL),
            raw_to_linear(read_source(sequence, frame_index), SOURCE_BLACK_LEVEL),
            read_motion(sequence, frame_index),
        )
        tensors = []
        for image in images:
            crop = np.ascontiguousarray(image[top:top + self.patch_size, left:left + self.patch_size]).astype(np.float32)
            tensors.append(torch.from_numpy(crop).unsqueeze(0))
        return tuple(tensors)

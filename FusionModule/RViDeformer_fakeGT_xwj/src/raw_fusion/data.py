"""Strict MIS20S1 validation and deterministic paired RAW patch datasets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import DatasetConfig, RawLayout, SequenceConfig, load_dataset_config
from .raw import RawFrameDirectoryReader, RawStreamReader, normalize_raw, pack_rggb


CHECKED_INDICES: Final[tuple[int, ...]] = (0, 1, 99, 100, 198, 199)


@dataclass(frozen=True, slots=True)
class SequenceValidationReport:
    frame_counts: dict[str, int]
    minimums: dict[str, int]
    maximums: dict[str, int]
    checked_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    sequences: dict[str, SequenceValidationReport]


def build_frame_indices(frame_range: tuple[int, int]) -> tuple[int, ...]:
    """Expand an inclusive frame range after validating its direction."""
    first, last = frame_range
    if first > last:
        raise ValueError("帧范围必须是递增的闭区间")
    return tuple(range(first, last + 1))


def validate_dataset(config: DatasetConfig) -> ValidationReport:
    """Validate stream geometry, stored noisy bits, and sampled RAW value ranges."""
    _validate_layout(config.layout)
    indices = _validation_indices(config.layout.frame_count)
    reports: dict[str, SequenceValidationReport] = {}
    for name, sequence in config.sequences.items():
        _validate_sequence_name(name, sequence)
        readers = _open_readers(config.layout, sequence)
        target_frame_count = _validate_all_target_files(config.layout, sequence)
        _validate_noisy_low_bits(config.layout, sequence, indices)
        signal_readers = {
            "noisy": readers.noisy,
            "denoised": readers.denoised,
            "fused": readers.fused,
            "target": readers.target,
        }
        black_levels = {
            "noisy": config.layout.noisy_black_level,
            "denoised": config.layout.candidate_black_level,
            "fused": config.layout.candidate_black_level,
            "target": config.layout.target_black_level,
        }
        minimums: dict[str, int] = {}
        maximums: dict[str, int] = {}
        for signal, reader in signal_readers.items():
            values = [reader.read_frame(index) for index in indices]
            _validate_raw_range(signal, values, black_levels[signal], config.layout.white_level)
            minimums[signal] = min(int(frame.min()) for frame in values)
            maximums[signal] = max(int(frame.max()) for frame in values)
        reports[name] = SequenceValidationReport(
            frame_counts={
                "noisy": config.layout.frame_count,
                "denoised": config.layout.frame_count,
                "fused": config.layout.frame_count,
                "target": target_frame_count,
            },
            minimums=minimums,
            maximums=maximums,
            checked_indices=indices,
        )
    if not reports:
        raise ValueError("数据集至少需要一个序列")
    return ValidationReport(sequences=reports)


class FusionPatchDataset(Dataset[dict[str, Tensor]]):
    """Sample reproducible causally paired patches from one configured sequence."""

    def __init__(
        self,
        config: DatasetConfig,
        *,
        sequence_name: str,
        frame_range: tuple[int, int],
        patch_size_packed: int,
        samples_per_epoch: int,
        seed: int,
        force_transform: tuple[bool, bool, bool] | None = None,
    ) -> None:
        _validate_layout(config.layout)
        if sequence_name not in config.sequences:
            raise ValueError(f"未知序列: {sequence_name}")
        self.config = config
        self.sequence_name = sequence_name
        self.sequence_index = tuple(config.sequences).index(sequence_name)
        self.frame_indices = build_frame_indices(frame_range)
        if (
            not self.frame_indices
            or self.frame_indices[0] < 1
            or self.frame_indices[-1] >= config.layout.frame_count
        ):
            raise ValueError("帧范围必须从至少为 1 的索引开始，且不能超出数据集边界")
        self.patch_size_packed = int(patch_size_packed)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        if self.patch_size_packed <= 0:
            raise ValueError("packed patch 尺寸必须为正数")
        if self.samples_per_epoch <= 0:
            raise ValueError("每个 epoch 的样本数必须为正数")
        packed_height = config.layout.height // 2
        packed_width = config.layout.width // 2
        if self.patch_size_packed > packed_height or self.patch_size_packed > packed_width:
            raise ValueError("packed patch 尺寸超出 RAW 边界")
        if force_transform is not None and len(force_transform) != 3:
            raise ValueError("force_transform 必须有三个布尔值")
        self.force_transform = force_transform
        self.epoch = 0
        sequence = config.sequences[sequence_name]
        _validate_sequence_name(sequence_name, sequence)
        self.readers = _open_readers(config.layout, sequence)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample_index = int(index)
        if sample_index < 0 or sample_index >= self.samples_per_epoch:
            raise IndexError(f"sample index out of range: {sample_index}")
        rng = np.random.default_rng(self.seed + self.epoch * self.samples_per_epoch + sample_index)
        frame_index = int(rng.choice(self.frame_indices))
        packed_height = self.config.layout.height // 2
        packed_width = self.config.layout.width // 2
        packed_top = int(rng.integers(0, packed_height - self.patch_size_packed + 1))
        packed_left = int(rng.integers(0, packed_width - self.patch_size_packed + 1))
        sensor_top = packed_top * 2
        sensor_left = packed_left * 2
        sensor_size = self.patch_size_packed * 2
        previous_index = frame_index - 1
        crops = {
            "prev_noisy": self.readers.noisy.read_crop(
                previous_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "curr_noisy": self.readers.noisy.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "denoised": self.readers.denoised.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "fused": self.readers.fused.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "target": self.readers.target.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
        }
        black_levels = {
            "prev_noisy": self.config.layout.noisy_black_level,
            "curr_noisy": self.config.layout.noisy_black_level,
            "denoised": self.config.layout.candidate_black_level,
            "fused": self.config.layout.candidate_black_level,
            "target": self.config.layout.target_black_level,
        }
        packed = {
            signal: pack_rggb(normalize_raw(crop, black_levels[signal], self.config.layout.white_level))
            for signal, crop in crops.items()
        }
        horizontal, vertical, transpose = self._transform_flags(rng)
        packed = {
            signal: _spatial_transform(image, horizontal, vertical, transpose)
            for signal, image in packed.items()
        }
        result = {signal: torch.from_numpy(image) for signal, image in packed.items()}
        result["sequence_index"] = torch.tensor(self.sequence_index, dtype=torch.int64)
        result["frame_index"] = torch.tensor(frame_index, dtype=torch.int64)
        return result

    def _transform_flags(self, rng: np.random.Generator) -> tuple[bool, bool, bool]:
        if self.force_transform is not None:
            return self.force_transform
        return tuple(bool(rng.integers(0, 2)) for _ in range(3))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _Readers:
    noisy: RawStreamReader
    denoised: RawStreamReader
    fused: RawStreamReader
    target: RawFrameDirectoryReader


def _open_readers(layout: RawLayout, sequence: SequenceConfig) -> _Readers:
    return _Readers(
        noisy=RawStreamReader(sequence.noisy_stream, layout.width, layout.height, layout.frame_count, layout.noisy_shift),
        denoised=RawStreamReader(sequence.denoised_stream, layout.width, layout.height, layout.frame_count, 0),
        fused=RawStreamReader(sequence.fused_stream, layout.width, layout.height, layout.frame_count, 0),
        target=RawFrameDirectoryReader(
            sequence.pseudo_gt_dir, layout.pseudo_gt_pattern, layout.width, layout.height, 0
        ),
    )


def _validation_indices(frame_count: int) -> tuple[int, ...]:
    if frame_count <= max(CHECKED_INDICES):
        raise ValueError(f"数据集帧数必须至少为 {max(CHECKED_INDICES) + 1}，以校验固定索引")
    return CHECKED_INDICES


def _validate_layout(layout: RawLayout) -> None:
    if layout.cfa_pattern != "RGGB":
        raise ValueError("仅支持 RGGB CFA")
    if layout.dtype != "<u2":
        raise ValueError("RAW 数据类型必须为 <u2")
    if layout.width <= 0 or layout.height <= 0 or layout.width % 2 or layout.height % 2:
        raise ValueError("RAW 宽高必须为正偶数")
    if layout.frame_count <= 0:
        raise ValueError("RAW 帧数必须为正数")
    if not 0 <= layout.noisy_shift <= 15:
        raise ValueError("noisy shift 必须在 0 到 15 之间")
    for name, black in (
        ("noisy", layout.noisy_black_level),
        ("candidate", layout.candidate_black_level),
        ("target", layout.target_black_level),
    ):
        if black < 0 or black >= layout.white_level:
            raise ValueError(f"{name} 黑电平必须小于白电平")


def _validate_sequence_name(name: str, sequence: SequenceConfig) -> None:
    if sequence.name != name:
        raise ValueError(f"序列键 {name} 与序列名称 {sequence.name} 不一致")


def _validate_all_target_files(layout: RawLayout, sequence: SequenceConfig) -> int:
    expected_bytes = layout.width * layout.height * np.dtype(layout.dtype).itemsize
    verified_target_files = 0
    for index in range(layout.frame_count):
        path = sequence.pseudo_gt_dir / layout.pseudo_gt_pattern.format(index=index)
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise ValueError(f"伪 GT 第 {index} 帧无法读取: {path}") from error
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"伪 GT 第 {index} 帧字节数为 {actual_bytes}，期望字节数 {expected_bytes}: {path}"
            )
        verified_target_files += 1

    return verified_target_files

def _validate_noisy_low_bits(
    layout: RawLayout, sequence: SequenceConfig, indices: tuple[int, ...]
) -> None:
    stored = np.memmap(
        sequence.noisy_stream,
        dtype=np.dtype(layout.dtype),
        mode="r",
        shape=(layout.frame_count, layout.height, layout.width),
    )
    mask = (1 << layout.noisy_shift) - 1
    try:
        for index in indices:
            if np.any(np.bitwise_and(stored[index], mask)):
                raise ValueError(f"noisy 第 {index} 帧低 {layout.noisy_shift} bit 不全为零")
    finally:
        del stored


def _validate_raw_range(signal: str, frames: list[np.ndarray], black_level: int, white_level: int) -> None:
    minimum = min(int(frame.min()) for frame in frames)
    maximum = max(int(frame.max()) for frame in frames)
    if minimum < 0 or maximum > white_level:
        raise ValueError(f"{signal} RAW 值范围为 {minimum}-{maximum}，期望 0-{white_level}")
    if black_level < 0 or black_level >= white_level:
        raise ValueError(f"{signal} 黑电平无效")


def _spatial_transform(image: np.ndarray, horizontal: bool, vertical: bool, transpose: bool) -> np.ndarray:
    transformed = image
    if horizontal:
        transformed = transformed[:, :, ::-1]
    if vertical:
        transformed = transformed[:, ::-1, :]
    if transpose:
        transformed = transformed.transpose(0, 2, 1)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def main_validate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="严格校验 MIS20S1 RAW 数据集")
    parser.add_argument("--dataset", required=True, type=Path, help="数据集 JSON 配置路径")
    arguments = parser.parse_args(argv)
    report = validate_dataset(load_dataset_config(arguments.dataset))
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main_validate()

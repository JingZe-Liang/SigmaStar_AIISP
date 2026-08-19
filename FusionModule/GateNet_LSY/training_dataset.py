from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_io import DatasetCatalog, RawStreamReader, pack_bayer


@dataclass(frozen=True)
class SequenceStatistics:
    source_to_dnr_offset: tuple[float, float, float, float]
    noise_sigma: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "source_to_dnr_offset": list(self.source_to_dnr_offset),
            "noise_sigma": list(self.noise_sigma),
        }


def _packed_frame(reader: RawStreamReader, index: int, crop=None) -> np.ndarray:
    mosaic = reader.read_frame(index, crop=crop)
    origin = (0, 0) if crop is None else crop[:2]
    return pack_bayer(mosaic, reader.spec.cfa_pattern, origin=origin)


def estimate_sequence_statistics(
    sequence,
    *,
    black_source: float,
    black_dnr: float,
    start_frame: int,
    stop_frame: int,
    sample_frames: int = 8,
    spatial_step: int = 8,
) -> SequenceStatistics:
    source_reader = RawStreamReader(sequence.source)
    denoised_reader = RawStreamReader(sequence.denoised)
    indices = np.linspace(
        start_frame, stop_frame - 1, min(sample_frames, stop_frame - start_frame), dtype=int
    )
    offsets: list[np.ndarray] = []
    temporal_differences: list[np.ndarray] = []
    for frame_index in sorted(set(indices.tolist())):
        source = _packed_frame(source_reader, frame_index)[
            :, ::spatial_step, ::spatial_step
        ].astype(np.float32)
        source_prev = _packed_frame(source_reader, frame_index - 1)[
            :, ::spatial_step, ::spatial_step
        ].astype(np.float32)
        denoised = _packed_frame(denoised_reader, frame_index)[
            :, ::spatial_step, ::spatial_step
        ].astype(np.float32)
        source_signal = source - black_source
        denoised_signal = denoised - black_dnr
        offsets.append((denoised_signal - source_signal).reshape(4, -1))
        temporal_differences.append((source_signal - (source_prev - black_source)).reshape(4, -1))

    offset_values = np.concatenate(offsets, axis=1)
    temporal_values = np.concatenate(temporal_differences, axis=1)
    offset = np.median(offset_values, axis=1)
    centered = temporal_values - np.median(temporal_values, axis=1, keepdims=True)
    sigma = np.median(np.abs(centered), axis=1) / (0.67448975 * math.sqrt(2.0))
    sigma = np.maximum(sigma, 1.0)
    return SequenceStatistics(
        source_to_dnr_offset=tuple(float(value) for value in offset),
        noise_sigma=tuple(float(value) for value in sigma),
    )


class FusionTrainingDataset(Dataset):
    """Virtual patch dataset for weakly supervised D2/D3 fusion training."""

    def __init__(
        self,
        catalog: DatasetCatalog,
        md_root: str | Path,
        *,
        sequence_ids: Sequence[str],
        samples_per_epoch: int,
        crop_size: int = 256,
        temporal_radius: int = 3,
        warmup_frames: int = 20,
        black_source: float = 252.0,
        black_dnr: float = 300.0,
        training: bool = True,
        seed: int = 1234,
    ):
        if crop_size <= 0 or crop_size % 2:
            raise ValueError("crop_size must be a positive even integer")
        if temporal_radius < 1:
            raise ValueError("temporal_radius must be at least 1")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        requested_ids = set(sequence_ids)
        self.sequences = tuple(
            sequence
            for sequence in catalog.fusion_sequences
            if sequence.sequence_id in requested_ids
        )
        found_ids = {sequence.sequence_id for sequence in self.sequences}
        if found_ids != requested_ids:
            raise ValueError(f"Unknown sequence IDs: {sorted(requested_ids - found_ids)}")

        self.md_root = Path(md_root).resolve()
        self.samples_per_epoch = samples_per_epoch
        self.crop_size = crop_size
        self.packed_crop_size = crop_size // 2
        self.temporal_radius = temporal_radius
        self.warmup_frames = warmup_frames
        self.black_source = black_source
        self.black_dnr = black_dnr
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.records: list[tuple[int, int]] = []
        self.readers: list[dict[str, RawStreamReader]] = []
        self.md_streams: list[np.memmap] = []
        self.statistics: dict[str, SequenceStatistics] = {}
        self.motion_coordinates: dict[tuple[int, int], np.ndarray] = {}

        for sequence_index, sequence in enumerate(self.sequences):
            start_frame = warmup_frames + temporal_radius
            stop_frame = sequence.frame_count - temporal_radius
            if start_frame >= stop_frame:
                raise ValueError(f"No valid frames in {sequence.sequence_id}")
            self.records.extend(
                (sequence_index, frame_index)
                for frame_index in range(start_frame, stop_frame)
            )
            readers = {
                "source": RawStreamReader(sequence.source),
                "denoised": RawStreamReader(sequence.denoised),
                "fused": RawStreamReader(sequence.fused),
            }
            self.readers.append(readers)
            md_path = self.md_root / sequence.sequence_id / "md_mog2.raw"
            expected_bytes = sequence.frame_count * 540 * 960
            if not md_path.is_file() or md_path.stat().st_size != expected_bytes:
                raise FileNotFoundError(
                    f"Missing or invalid MD stream for {sequence.sequence_id}: {md_path}"
                )
            md_stream = np.memmap(
                md_path,
                dtype=np.uint8,
                mode="r",
                shape=(sequence.frame_count, 540, 960),
            )
            self.md_streams.append(md_stream)
            self.statistics[sequence.sequence_id] = estimate_sequence_statistics(
                sequence,
                black_source=black_source,
                black_dnr=black_dnr,
                start_frame=start_frame,
                stop_frame=stop_frame,
            )
            for frame_index in range(start_frame, stop_frame):
                coordinates = np.column_stack(np.nonzero(md_stream[frame_index] > 0)).astype(
                    np.int32, copy=False
                )
                if len(coordinates):
                    self.motion_coordinates[(sequence_index, frame_index)] = coordinates

        self.motion_records = [
            record for record in self.records if record in self.motion_coordinates
        ]
        if not self.motion_records:
            raise ValueError("No motion pixels found in the selected sequences")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, item: int) -> np.random.Generator:
        epoch = self.epoch if self.training else 0
        return np.random.default_rng(self.seed + epoch * self.samples_per_epoch + item)

    def _random_crop_origin(
        self,
        rng: np.random.Generator,
        sequence_index: int,
        frame_index: int,
        sampling_mode: int,
    ) -> tuple[int, int]:
        max_top = 540 - self.packed_crop_size
        max_left = 960 - self.packed_crop_size
        if sampling_mode == 1 and (sequence_index, frame_index) in self.motion_coordinates:
            coordinates = self.motion_coordinates[(sequence_index, frame_index)]
            center_y, center_x = coordinates[int(rng.integers(0, len(coordinates)))]
            top = int(
                np.clip(
                    center_y - rng.integers(0, self.packed_crop_size), 0, max_top
                )
            )
            left = int(
                np.clip(
                    center_x - rng.integers(0, self.packed_crop_size), 0, max_left
                )
            )
            return top, left

        for _ in range(12 if sampling_mode == 0 else 1):
            top = int(rng.integers(0, max_top + 1))
            left = int(rng.integers(0, max_left + 1))
            if sampling_mode != 0:
                return top, left
            patch = self.md_streams[sequence_index][
                frame_index,
                top : top + self.packed_crop_size,
                left : left + self.packed_crop_size,
            ]
            if np.count_nonzero(patch) / patch.size < 0.002:
                return top, left
        return top, left

    def __getitem__(self, item: int) -> dict[str, Any]:
        rng = self._rng(item)
        probability = float(rng.random())
        sampling_mode = 1 if probability < 0.45 else (0 if probability < 0.80 else 2)
        records = self.motion_records if sampling_mode == 1 else self.records
        sequence_index, frame_index = records[int(rng.integers(0, len(records)))]
        sequence = self.sequences[sequence_index]
        packed_top, packed_left = self._random_crop_origin(
            rng, sequence_index, frame_index, sampling_mode
        )
        crop = (
            packed_top * 2,
            packed_left * 2,
            self.crop_size,
            self.crop_size,
        )
        readers = self.readers[sequence_index]
        raw_window = np.stack(
            [
                _packed_frame(readers["source"], offset, crop=crop)
                for offset in range(
                    frame_index - self.temporal_radius,
                    frame_index + self.temporal_radius + 1,
                )
            ],
            axis=0,
        ).astype(np.float32)
        denoised = _packed_frame(readers["denoised"], frame_index, crop=crop).astype(
            np.float32
        )
        fused = _packed_frame(readers["fused"], frame_index, crop=crop).astype(
            np.float32
        )
        stats = self.statistics[sequence.sequence_id]
        offset = np.asarray(stats.source_to_dnr_offset, dtype=np.float32).reshape(1, 4, 1, 1)
        source_window = raw_window - self.black_source + offset
        sorted_window = np.sort(source_window, axis=0)
        proxy = sorted_window[1:-1].mean(axis=0)
        temporal_range = (
            source_window.max(axis=0) - source_window.min(axis=0)
        ).mean(axis=0, keepdims=True)
        center = self.temporal_radius
        motion = self.md_streams[sequence_index][
            frame_index,
            packed_top : packed_top + self.packed_crop_size,
            packed_left : packed_left + self.packed_crop_size,
        ]
        source_luma = (raw_window[center] - self.black_source).mean(axis=0, keepdims=True)
        valid_signal = (
            (source_luma > 8.0)
            & (raw_window[center].max(axis=0, keepdims=True) < 4090.0)
        ).astype(np.float32)

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

        return {
            "source": tensor(source_window[center]),
            "source_prev": tensor(source_window[center - 1]),
            "source_next": tensor(source_window[center + 1]),
            "denoised": tensor(denoised - self.black_dnr),
            "fused": tensor(fused - self.black_dnr),
            "proxy": tensor(proxy),
            "temporal_range": tensor(temporal_range),
            "motion": tensor((motion > 0).astype(np.float32)[None]),
            "valid_signal": tensor(valid_signal),
            "noise_sigma": tensor(
                np.asarray(stats.noise_sigma, dtype=np.float32).reshape(4, 1, 1)
            ),
            "sequence_id": sequence.sequence_id,
            "frame_index": frame_index,
            "sampling_mode": sampling_mode,
        }

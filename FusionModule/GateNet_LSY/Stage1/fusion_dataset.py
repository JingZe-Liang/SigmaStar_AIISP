"""Backward-compatible single-sequence loader built on the strict RAW reader.

Use ``dataset_io.PairedFusionDataset`` to read both paired sequences together.
"""

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from dataset_io import (
    LEFT_ALIGNED_12BIT,
    RIGHT_ALIGNED_12BIT,
    RawStreamReader,
    RawStreamSpec,
    _prepare_tensor,
)


class FusionDataset(Dataset):
    def __init__(
        self,
        source_path,
        denoised_path,
        fused_path,
        width=1920,
        height=1080,
        crop_size=256,
        black_source_12bit=252,
        black_denoised_12bit=300,
        max_frames=None,
        normalize=True,
    ):
        super().__init__()
        if (width, height) != (1920, 1080):
            raise ValueError("This dataset is fixed at 1920x1080")
        if crop_size <= 0 or crop_size % 2:
            raise ValueError("crop_size must be a positive even integer")
        if crop_size > min(width, height):
            raise ValueError("crop_size exceeds the RAW frame dimensions")

        source_path = Path(source_path)
        denoised_path = Path(denoised_path)
        fused_path = Path(fused_path)
        self.source_spec = RawStreamSpec.from_path(
            source_path,
            role="source",
            encoding=LEFT_ALIGNED_12BIT,
            category="paired",
            condition=source_path.parent.name,
        )
        self.denoised_spec = RawStreamSpec.from_path(
            denoised_path,
            role="2dnr",
            encoding=RIGHT_ALIGNED_12BIT,
            category="paired",
            condition=source_path.parent.name,
            cfa_pattern=self.source_spec.cfa_pattern,
        )
        self.fused_spec = RawStreamSpec.from_path(
            fused_path,
            role="3dnr",
            encoding=RIGHT_ALIGNED_12BIT,
            category="paired",
            condition=source_path.parent.name,
            cfa_pattern=self.source_spec.cfa_pattern,
        )
        frame_counts = {
            self.source_spec.frame_count,
            self.denoised_spec.frame_count,
            self.fused_spec.frame_count,
        }
        if len(frame_counts) != 1:
            raise ValueError(f"source/2DNR/3DNR frame counts differ: {frame_counts}")

        available_frames = self.source_spec.frame_count
        self.num_frames = (
            available_frames if max_frames is None else min(available_frames, max_frames)
        )
        if self.num_frames < 3:
            raise ValueError("At least 3 frames are required for prev/current/next")

        self.crop_size = crop_size
        self.black_source = black_source_12bit
        self.black_dnr = black_denoised_12bit
        self.normalize = normalize
        self.valid_indices = tuple(range(1, self.num_frames - 1))
        self.readers = {
            "source": RawStreamReader(self.source_spec),
            "denoised": RawStreamReader(self.denoised_spec),
            "fused": RawStreamReader(self.fused_spec),
        }

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index) -> dict[str, Any]:
        frame_index = self.valid_indices[index]
        top = int(np.random.randint(0, 1080 - self.crop_size + 1))
        left = int(np.random.randint(0, 1920 - self.crop_size + 1))
        crop = (top, left, self.crop_size, self.crop_size)
        origin = (top, left)

        def source_tensor(source_index: int):
            mosaic = self.readers["source"].read_frame(source_index, crop=crop)
            return _prepare_tensor(
                mosaic,
                cfa_pattern=self.source_spec.cfa_pattern,
                origin=origin,
                black_level=self.black_source,
                normalize=self.normalize,
            )

        denoised = self.readers["denoised"].read_frame(frame_index, crop=crop)
        fused = self.readers["fused"].read_frame(frame_index, crop=crop)
        return {
            "source": source_tensor(frame_index),
            "source_prev": source_tensor(frame_index - 1),
            "source_next": source_tensor(frame_index + 1),
            "denoised": _prepare_tensor(
                denoised,
                cfa_pattern=self.denoised_spec.cfa_pattern,
                origin=origin,
                black_level=self.black_dnr,
                normalize=self.normalize,
            ),
            "fused": _prepare_tensor(
                fused,
                cfa_pattern=self.fused_spec.cfa_pattern,
                origin=origin,
                black_level=self.black_dnr,
                normalize=self.normalize,
            ),
            "frame_index": frame_index,
            "crop_origin": origin,
        }

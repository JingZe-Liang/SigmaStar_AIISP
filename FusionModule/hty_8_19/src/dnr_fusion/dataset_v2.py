from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .confidence import safety_confidence
from .dataset_fast import (
    load_packed_crop,
    open_scene_streams,
    scene_safety_params,
)
from .features_v2 import build_threshold_normalized_features


class ThresholdNormalizedPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        config: dict[str, Any],
        scene_name: str,
        frame_range: tuple[int, int] | list[int],
        *,
        patch_size: int,
        samples: int,
        seed: int,
        augment: bool,
    ) -> None:
        self.config = config
        self.scene_name = scene_name
        self.streams = open_scene_streams(config, scene_name)
        self.frame_start, self.frame_end = map(int, frame_range)
        if self.frame_start < 1 or self.frame_end >= self.streams.frame_count:
            raise ValueError(
                f"Frame range must be within [1, {self.streams.frame_count - 1}]"
            )
        self.patch_size = int(patch_size)
        self.samples = int(samples)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0
        self.safety = scene_safety_params(config, scene_name)
        self.packed_height = int(config["raw"]["height"]) // 2
        self.packed_width = int(config["raw"]["width"]) // 2

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    @staticmethod
    def _augment(
        tensors: list[torch.Tensor], rng: np.random.Generator
    ) -> list[torch.Tensor]:
        if rng.random() < 0.5:
            tensors = [torch.flip(item, dims=(-1,)) for item in tensors]
        if rng.random() < 0.5:
            tensors = [torch.flip(item, dims=(-2,)) for item in tensors]
        rotations = int(rng.integers(0, 4))
        if rotations:
            tensors = [
                torch.rot90(item, rotations, dims=(-2, -1)) for item in tensors
            ]
        return tensors

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(
            self.seed + self.epoch * self.samples + int(index)
        )
        frame_index = int(rng.integers(self.frame_start, self.frame_end + 1))
        top = int(rng.integers(0, self.packed_height - self.patch_size + 1))
        left = int(rng.integers(0, self.packed_width - self.patch_size + 1))
        rows = slice(top, top + self.patch_size)
        cols = slice(left, left + self.patch_size)

        source, denoised, fused = load_packed_crop(
            self.streams, frame_index, self.config["raw"], rows, cols
        )
        _, previous, _ = load_packed_crop(
            self.streams, frame_index - 1, self.config["raw"], rows, cols
        )
        current_source = torch.from_numpy(source.copy()).unsqueeze(0)
        current_2dnr = torch.from_numpy(denoised.copy()).unsqueeze(0)
        current_3dnr = torch.from_numpy(fused.copy()).unsqueeze(0)
        previous_2dnr = torch.from_numpy(previous.copy()).unsqueeze(0)
        target, diagnostics = safety_confidence(
            current_2dnr, previous_2dnr, current_3dnr, self.safety
        )
        features = build_threshold_normalized_features(
            current_source,
            current_2dnr,
            current_3dnr,
            previous_2dnr,
            self.safety,
        )
        tensors = [features.squeeze(0), target.squeeze(0), diagnostics["hard_mask"].squeeze(0)]
        if self.augment:
            tensors = self._augment(tensors, rng)
        features, target, hard_mask = tensors
        return {
            "features": features.contiguous(),
            "target_gate": target.contiguous(),
            "hard_mask": hard_mask.contiguous(),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
        }

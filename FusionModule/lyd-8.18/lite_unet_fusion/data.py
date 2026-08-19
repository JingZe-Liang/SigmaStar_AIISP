from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .raw import FusionStream


class RandomPatchDataset(Dataset):
    def __init__(self, stream: FusionStream, frames: list[int], patch_size: int, samples: int, seed: int) -> None:
        if patch_size % 8:
            raise ValueError("patch_size must be divisible by 8 for the three downsampling stages")
        self.stream, self.frames, self.patch_size, self.samples, self.seed = stream, frames, patch_size, samples, seed
        if not frames:
            raise ValueError("At least one frame is required")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        frame = self.frames[int(rng.integers(0, len(self.frames)))]
        inputs, values = self.stream.network_input(frame)
        height, width = inputs.shape[-2:]
        top = int(rng.integers(0, height - self.patch_size + 1))
        left = int(rng.integers(0, width - self.patch_size + 1))
        region = np.s_[..., top : top + self.patch_size, left : left + self.patch_size]
        return {
            "input": torch.from_numpy(inputs[region].copy()),
            "teacher": torch.from_numpy(values["teacher"][region].copy()),
            "motion": torch.from_numpy(values["motion"][region].copy()),
            "flatness": torch.from_numpy(values["flatness"][region].copy()),
        }

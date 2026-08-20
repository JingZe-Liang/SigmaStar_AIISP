from __future__ import annotations

import numpy as np
import torch

from .dataset_v2 import ThresholdNormalizedPatchDataset
from .packed_augment import augment_packed_tensors


class BayerAwarePatchDataset(ThresholdNormalizedPatchDataset):
    """v2 dataset with CFA-correct geometric augmentation."""

    @staticmethod
    def _augment(
        tensors: list[torch.Tensor], rng: np.random.Generator
    ) -> list[torch.Tensor]:
        return augment_packed_tensors(tensors, rng)

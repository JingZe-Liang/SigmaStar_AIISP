from __future__ import annotations

import random

import torch


class SameCFAMasker:
    """同步遮挡当前帧全部候选路径，替换源严格保持 Bayer 相位。"""

    def __init__(self, points_per_patch: int, min_distance: int, seed: int):
        self.points_per_patch = points_per_patch
        self.min_distance = min_distance
        self.random = random.Random(seed)
        self.offsets = tuple((dy, dx) for dy in (-4, -2, 0, 2, 4) for dx in (-4, -2, 0, 2, 4) if (dy, dx) != (0, 0))

    def mask(self, image_2dnr: torch.Tensor, image_3dnr: torch.Tensor, noisy_current: torch.Tensor):
        if image_2dnr.shape != image_3dnr.shape or image_2dnr.shape != noisy_current.shape:
            raise ValueError("需要大小相同的 2DNR、3DNR 与 noisy_t")
        batch_size, channels, height, width = noisy_current.shape
        if channels != 1 or height < 10 or width < 10:
            raise ValueError("mask 输入必须为足够大的单通道 Bayer patch")
        masked = [image.clone() for image in (image_2dnr, image_3dnr, noisy_current)]
        supervised = torch.zeros_like(noisy_current, dtype=torch.bool)
        for batch_index in range(batch_size):
            centers: list[tuple[int, int]] = []
            attempts = 0
            while len(centers) < self.points_per_patch and attempts < self.points_per_patch * 100:
                attempts += 1
                row = self.random.randrange(4, height - 4)
                column = self.random.randrange(4, width - 4)
                if all(max(abs(row - old_row), abs(column - old_column)) >= self.min_distance for old_row, old_column in centers):
                    centers.append((row, column))
            if len(centers) != self.points_per_patch:
                raise RuntimeError("当前 patch 无法容纳要求数量的间隔 mask 点")
            for row, column in centers:
                offset_row, offset_column = self.offsets[self.random.randrange(len(self.offsets))]
                donor_row, donor_column = row + offset_row, column + offset_column
                for image in masked:
                    image[batch_index, 0, row, column] = image[batch_index, 0, donor_row, donor_column]
                supervised[batch_index, 0, row, column] = True
        return (*masked, supervised)

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def _channel_transform(item: torch.Tensor, permutation: tuple[int, int, int, int]) -> torch.Tensor:
    """Apply a packed RGGB spatial transform and its CFA channel permutation."""

    if item.shape[0] == 1 or item.shape[0] % 4:
        return item
    groups = item.reshape(item.shape[0] // 4, 4, *item.shape[-2:])
    groups = groups[:, list(permutation)]
    return groups.reshape_as(item)


def _horizontal(item: torch.Tensor) -> torch.Tensor:
    return _channel_transform(torch.flip(item, dims=(-1,)), (1, 0, 3, 2))


def _vertical(item: torch.Tensor) -> torch.Tensor:
    return _channel_transform(torch.flip(item, dims=(-2,)), (3, 2, 1, 0))


def _rotate(item: torch.Tensor, turns: int) -> torch.Tensor:
    permutations = {
        1: (1, 2, 3, 0),
        2: (2, 3, 0, 1),
        3: (3, 0, 1, 2),
    }
    rotated = torch.rot90(item, turns, dims=(-2, -1))
    return _channel_transform(rotated, permutations[turns])


def augment_packed_tensors(
    tensors: list[torch.Tensor], rng: np.random.Generator
) -> list[torch.Tensor]:
    transform: Callable[[torch.Tensor], torch.Tensor] = lambda item: item
    if rng.random() < 0.5:
        previous = transform
        transform = lambda item, previous=previous: _horizontal(previous(item))
    if rng.random() < 0.5:
        previous = transform
        transform = lambda item, previous=previous: _vertical(previous(item))
    turns = int(rng.integers(0, 4))
    if turns:
        previous = transform
        transform = lambda item, previous=previous, turns=turns: _rotate(previous(item), turns)
    return [transform(item) for item in tensors]

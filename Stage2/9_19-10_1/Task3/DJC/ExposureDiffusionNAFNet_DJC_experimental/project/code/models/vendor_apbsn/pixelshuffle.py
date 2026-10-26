"""AP-BSN's channel-preserving pixel shuffle, extracted from src/util/util.py."""

import torch
import torch.nn.functional as F


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0,
                                pad_value: float = 0.0) -> torch.Tensor:
    b, c, h, w = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad:
        unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    return unshuffled.view(b, c, f, f, h // f + 2 * pad, w // f + 2 * pad).permute(
        0, 1, 2, 4, 3, 5).reshape(b, c, h + 2 * f * pad, w + 2 * f * pad)


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0) -> torch.Tensor:
    b, c, h, w = x.shape
    before_shuffle = x.view(b, c, f, h // f, f, w // f).permute(
        0, 1, 2, 4, 3, 5).reshape(b, c * f * f, h // f, w // f)
    if pad:
        before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)

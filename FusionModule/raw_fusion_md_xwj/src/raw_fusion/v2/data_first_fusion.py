"""Shared safe composition for data-first training and inference."""
from __future__ import annotations

import torch
from torch import Tensor


def limit_q_to_raw_range(denoised: Tensor, delta: Tensor, q: Tensor) -> Tensor:
    """Limit a non-negative q map so denoised + q * delta remains in [0, 1]."""
    if denoised.shape != delta.shape or denoised.ndim != 4 or q.shape[0] != denoised.shape[0] or q.shape[-2:] != denoised.shape[-2:]:
        raise ValueError("safe q composition shapes are incompatible")
    positive = delta > 0.0
    negative = delta < 0.0
    infinity = torch.full_like(delta, float("inf"))
    upper = torch.where(positive, (1.0 - denoised) / delta, infinity)
    lower = torch.where(negative, denoised / (-delta), infinity)
    cap = torch.minimum(upper, lower).amin(dim=1, keepdim=True).clamp(0.0, 0.5)
    return torch.minimum(q.clamp_min(0.0), cap)


__all__ = ["limit_q_to_raw_range"]

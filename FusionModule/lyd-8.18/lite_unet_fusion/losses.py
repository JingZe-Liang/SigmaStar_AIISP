from __future__ import annotations

import torch
from torch.nn import functional as F


def total_variation(value: torch.Tensor) -> torch.Tensor:
    return (value[:, :, 1:] - value[:, :, :-1]).abs().mean() + (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()


def fusion_loss(beta: torch.Tensor, teacher: torch.Tensor, motion: torch.Tensor, flatness: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    effective = 0.35 * beta * (1.0 - motion) * flatness
    distill = F.smooth_l1_loss(effective, teacher)
    motion_penalty = (beta * motion).mean()
    smoothness = total_variation(beta)
    loss = distill + 0.10 * motion_penalty + 0.01 * smoothness
    return loss, {"distill": float(distill.detach()), "motion": float(motion_penalty.detach()), "tv": float(smoothness.detach())}

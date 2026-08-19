"""Pseudo-GT, noisy-RAW, and base-preservation losses for FGRF-Net."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

def charbonnier(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value * value + epsilon * epsilon)


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = mask.expand_as(value)
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator


def raw_data_consistency_loss(output: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    """RAW observation consistency against the current noisy frame."""
    return charbonnier(output - noisy).mean()


def pseudo_gt_mse_loss(output: torch.Tensor, pseudo_gt: torch.Tensor) -> torch.Tensor:
    """Supervise the fused RAW against the external denoised pseudo target."""
    return F.mse_loss(output, pseudo_gt)


def base_constraint_loss(output: torch.Tensor, base: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Keep the low-frequency content of 2DNR unchanged."""
    low_output = F.avg_pool2d(output, kernel_size, stride=1, padding=kernel_size // 2)
    low_base = F.avg_pool2d(base, kernel_size, stride=1, padding=kernel_size // 2)
    return charbonnier(low_output - low_base).mean()


def total_loss(
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    pseudo_gt_weight: float = 1.0,
    raw_weight: float = 0.25,
    base_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    output = prediction["output"]
    raw = raw_data_consistency_loss(output, batch["noisy"])
    pseudo_gt = pseudo_gt_mse_loss(output, batch["pseudo_gt"])
    base = base_constraint_loss(output, batch["base"])
    total = raw_weight * raw + pseudo_gt_weight * pseudo_gt + base_weight * base
    return {"total": total, "raw": raw, "pseudo_gt": pseudo_gt, "base": base}

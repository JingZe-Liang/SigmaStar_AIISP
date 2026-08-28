from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import Mamba2DPriorOutput


def _edge_aware_tv(beta: Tensor, source: Tensor) -> Tensor:
    beta_x = (beta[..., :, 1:] - beta[..., :, :-1]).abs()
    beta_y = (beta[..., 1:, :] - beta[..., :-1, :]).abs()
    image_x = (source[..., :, 1:] - source[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    image_y = (source[..., 1:, :] - source[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    return (beta_x * torch.exp(-10.0 * image_x)).mean() + (beta_y * torch.exp(-10.0 * image_y)).mean()


def compute_pseudolabel_loss(output: Mamba2DPriorOutput, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
    distill = F.smooth_l1_loss(output.weight_3d, batch["teacher_w3"])
    motion_leak = (output.weight_3d * output.motion).mean()
    smoothness = _edge_aware_tv(output.beta, batch["curr4"])
    mean_correction = output.weight_3d.mean()
    loss = distill + 0.10 * motion_leak + 0.01 * smoothness + 0.01 * mean_correction
    plane_mean = output.weight_3d.detach().mean(dim=(0, 2, 3))
    return loss, {
        "loss": float(loss.detach()),
        "distill": float(distill.detach()),
        "motion_leak": float(motion_leak.detach()),
        "tv": float(smoothness.detach()),
        "mean_w3": float(mean_correction.detach()),
        "max_w3": float(output.weight_3d.detach().amax()),
        "mean_w3_r": float(plane_mean[0]),
        "mean_w3_g1": float(plane_mean[1]),
        "mean_w3_g2": float(plane_mean[2]),
        "mean_w3_b": float(plane_mean[3]),
    }

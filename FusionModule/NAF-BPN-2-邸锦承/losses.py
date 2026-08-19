from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(error: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(error.square() + epsilon * epsilon)


def masked_noisy_loss(prediction: torch.Tensor, target: torch.Tensor, supervised: torch.Tensor, epsilon: float) -> torch.Tensor:
    valid = supervised & target.lt(1.0)
    weight = valid.to(prediction.dtype)
    return (charbonnier(prediction - target, epsilon) * weight).sum() / weight.sum().clamp_min(1.0)


def _same_cfa_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[..., :, 2:] - image[..., :, :-2], image[..., 2:, :] - image[..., :-2, :]


def candidate_gradient_loss(
    prediction: torch.Tensor,
    image_2dnr: torch.Tensor,
    image_3dnr: torch.Tensor,
    motion_mask: torch.Tensor,
    algorithm_threshold: float,
    algorithm_transition: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同 CFA 梯度：运动区跟随 2DNR，静止区跟随 3DNR，分歧区软偏向 2DNR。"""
    disagreement = F.avg_pool2d(torch.abs(image_2dnr - image_3dnr), 5, stride=1, padding=2)
    algorithm_2d_weight = torch.sigmoid((disagreement - algorithm_threshold) / algorithm_transition)
    candidate_2d_weight = torch.maximum(motion_mask, algorithm_2d_weight).detach()
    pred_x, pred_y = _same_cfa_gradients(prediction)
    dnr2_x, dnr2_y = _same_cfa_gradients(image_2dnr)
    dnr3_x, dnr3_y = _same_cfa_gradients(image_3dnr)
    weight_x = candidate_2d_weight[..., :, 1:-1]
    weight_y = candidate_2d_weight[..., 1:-1, :]
    target_x = weight_x * dnr2_x + (1.0 - weight_x) * dnr3_x
    target_y = weight_y * dnr2_y + (1.0 - weight_y) * dnr3_y
    loss_x = charbonnier(pred_x - target_x, epsilon).mean()
    loss_y = charbonnier(pred_y - target_y, epsilon).mean()
    return (loss_x + loss_y) * 0.5, candidate_2d_weight.mean()

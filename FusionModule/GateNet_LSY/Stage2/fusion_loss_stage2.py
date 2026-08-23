from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2FusionLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, *, motion_weight: float = 0.2):
        super().__init__()
        if motion_weight < 0:
            raise ValueError("motion_weight must be non-negative")
        self.base_criterion = base_criterion
        self.motion_weight = motion_weight

    def forward(
        self,
        alpha: torch.Tensor,
        motion_logit: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fusion_loss, metrics = self.base_criterion(alpha, batch)
        target = batch["motion"]
        valid = batch["valid_signal"]

        positive = (target * valid).sum()
        negative = ((1.0 - target) * valid).sum()
        pos_weight = torch.clamp(
            negative / torch.clamp(positive, min=1.0), 1.0, 20.0
        ).detach()
        bce = F.binary_cross_entropy_with_logits(
            motion_logit, target, reduction="none", pos_weight=pos_weight
        )
        motion_bce = (bce * valid).sum() / torch.clamp(valid.sum(), min=1.0)

        probability = torch.sigmoid(motion_logit)
        intersection = (probability * target * valid).sum()
        denominator = ((probability + target) * valid).sum()
        motion_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        motion_loss = 0.5 * (motion_bce + motion_dice)
        total = fusion_loss + self.motion_weight * motion_loss

        with torch.no_grad():
            prediction = probability >= 0.5
            truth = target >= 0.5
            valid_bool = valid >= 0.5
            true_positive = (prediction & truth & valid_bool).sum().float()
            false_positive = (prediction & ~truth & valid_bool).sum().float()
            false_negative = (~prediction & truth & valid_bool).sum().float()
            precision = true_positive / torch.clamp(
                true_positive + false_positive, min=1.0
            )
            recall = true_positive / torch.clamp(
                true_positive + false_negative, min=1.0
            )

        metrics = dict(metrics)
        metrics["fusion_total"] = metrics["total"]
        metrics["total"] = total.detach()
        metrics["motion_aux"] = motion_loss.detach()
        metrics["motion_bce"] = motion_bce.detach()
        metrics["motion_dice"] = motion_dice.detach()
        metrics["motion_precision"] = precision.detach()
        metrics["motion_recall"] = recall.detach()
        metrics["predicted_motion_fraction"] = probability.mean().detach()
        return total, metrics

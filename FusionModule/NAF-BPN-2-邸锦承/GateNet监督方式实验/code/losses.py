from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def charbonnier(error: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(error.square() + epsilon * epsilon)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _gradient_pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., 1:] - value[..., :-1], value[..., 1:, :] - value[..., :-1, :]


def masked_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    prediction_x, prediction_y = _gradient_pair(prediction)
    target_x, target_y = _gradient_pair(target)
    mask_x = mask[..., 1:]
    mask_y = mask[..., 1:, :]
    return 0.5 * (
        _masked_mean(charbonnier(prediction_x - target_x, epsilon), mask_x)
        + _masked_mean(charbonnier(prediction_y - target_y, epsilon), mask_y)
    )


@dataclass(frozen=True)
class WeakLossWeights:
    proxy_static: float = 1.0
    gradient_static: float = 0.25
    motion_anchor: float = 0.10
    candidate_stability: float = 0.10
    masked_noisy: float = 0.05


class NAFBPNWeakFusionLoss(nn.Module):
    """GateNet-style weak supervision for the original NAF-BPN output.

    There is no clean target and no alpha target here.  The seven-frame
    trimmed mean is a static-region proxy; offline motion is only a loss mask.
    """

    def __init__(
        self,
        weights: WeakLossWeights = WeakLossWeights(),
        *,
        static_temporal_threshold: float = 0.015,
        static_range_threshold: float = 0.035,
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.static_temporal_threshold = static_temporal_threshold
        self.static_range_threshold = static_range_threshold
        self.epsilon = epsilon

    def forward(
        self,
        prediction: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_2dnr = batch["image_2dnr"]
        image_3dnr = batch["image_3dnr"]
        proxy = batch["proxy"]
        motion = batch["motion_target"].clamp(0.0, 1.0)
        valid = batch["valid_signal"].clamp(0.0, 1.0)
        temporal_difference = batch["temporal_difference"]
        temporal_range = batch["temporal_range"]
        sigma = batch["noise_sigma"].clamp_min(1e-4)

        static = (
            (1.0 - motion)
            * (temporal_difference < self.static_temporal_threshold).float()
            * (temporal_range < self.static_range_threshold).float()
            * valid
        )
        moving = motion * valid

        disagreement = (image_3dnr - image_2dnr).abs().mean(dim=1, keepdim=True)
        disagreement = (disagreement / (3.0 * sigma)).clamp(0.0, 1.0)
        candidate_confidence = 1.0 - disagreement

        proxy_error = charbonnier(prediction - proxy, self.epsilon).mean(dim=1, keepdim=True)
        proxy_loss = _masked_mean(proxy_error, static)
        gradient_loss = masked_gradient_loss(prediction, proxy, static, self.epsilon)

        # A soft D2 anchor reduces likely 3DNR trails without forcing motion
        # regions to discard 3DNR completely.
        motion_weight = moving * (0.25 + 0.75 * candidate_confidence)
        motion_anchor = _masked_mean(
            charbonnier(prediction - image_2dnr, self.epsilon).mean(dim=1, keepdim=True),
            motion_weight,
        )
        stability_weight = valid * candidate_confidence
        candidate_stability = _masked_mean(
            charbonnier(prediction - image_2dnr, self.epsilon).mean(dim=1, keepdim=True),
            stability_weight,
        )

        # This is retained only as a low-weight stability term.  D2/D3 have
        # already seen the centre noisy sample, so it is not strict J-invariance.
        noisy_mask = static + 0.25 * moving
        masked_noisy = _masked_mean(
            charbonnier(prediction - batch["noisy_current"], self.epsilon).mean(dim=1, keepdim=True),
            noisy_mask,
        )

        total = (
            self.weights.proxy_static * proxy_loss
            + self.weights.gradient_static * gradient_loss
            + self.weights.motion_anchor * motion_anchor
            + self.weights.candidate_stability * candidate_stability
            + self.weights.masked_noisy * masked_noisy
        )

        with torch.no_grad():
            d2_error = charbonnier(image_2dnr - proxy, self.epsilon).mean(dim=1, keepdim=True)
            d3_error = charbonnier(image_3dnr - proxy, self.epsilon).mean(dim=1, keepdim=True)
            metrics = {
                "total": total.detach(),
                "proxy_static": proxy_loss.detach(),
                "gradient_static": gradient_loss.detach(),
                "motion_anchor": motion_anchor.detach(),
                "candidate_stability": candidate_stability.detach(),
                "masked_noisy": masked_noisy.detach(),
                "static_fraction": static.mean().detach(),
                "motion_fraction": moving.mean().detach(),
                "proxy_output": _masked_mean(proxy_error, static).detach(),
                "proxy_d2": _masked_mean(d2_error, static).detach(),
                "proxy_d3": _masked_mean(d3_error, static).detach(),
                "candidate_disagreement": disagreement.mean().detach(),
            }
        return total, metrics


def stage1_supervised_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Original clean-GT pretraining objective kept separate from Stage 2."""
    base = charbonnier(prediction - target, epsilon).mean()
    gradient = masked_gradient_loss(
        prediction,
        target,
        torch.ones_like(target),
        epsilon,
    )
    total = base + 0.5 * gradient
    return total, {"total": total.detach(), "charbonnier": base.detach(), "gradient": gradient.detach()}

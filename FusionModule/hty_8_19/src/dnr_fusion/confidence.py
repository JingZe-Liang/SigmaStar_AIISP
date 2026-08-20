from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SafetyParams:
    motion_threshold_dn: float
    disagreement_threshold_dn: float
    dynamic_range_dn: float
    local_kernel: int = 5
    dilation_kernel: int = 9
    soft_scale: float = 0.35
    hard_motion_scale: float = 1.0
    hard_joint_motion_scale: float = 0.55
    hard_joint_disagreement_scale: float = 1.0

    @property
    def motion_threshold(self) -> float:
        return self.motion_threshold_dn / self.dynamic_range_dn

    @property
    def disagreement_threshold(self) -> float:
        return self.disagreement_threshold_dn / self.dynamic_range_dn


def _odd(value: int, name: str) -> int:
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return value


def safety_confidence(
    current_2dnr: torch.Tensor,
    previous_2dnr: torch.Tensor,
    current_3dnr: torch.Tensor,
    params: SafetyParams,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a shared CFA gate ceiling and diagnostic risk maps.

    All inputs are normalized packed Bayer tensors with shape Bx4xHxW. A zero
    confidence is a hard forward-pass fallback to 2DNR, not merely a loss term.
    """

    if current_2dnr.shape != previous_2dnr.shape or current_2dnr.shape != current_3dnr.shape:
        raise ValueError("2DNR/3DNR tensors must have identical shapes")
    if current_2dnr.ndim != 4 or current_2dnr.shape[1] != 4:
        raise ValueError(f"Expected Bx4xHxW packed Bayer, got {current_2dnr.shape}")

    local_kernel = _odd(params.local_kernel, "local_kernel")
    dilation_kernel = _odd(params.dilation_kernel, "dilation_kernel")
    local_pad = local_kernel // 2
    dilation_pad = dilation_kernel // 2

    motion = torch.mean(torch.abs(current_2dnr - previous_2dnr), dim=1, keepdim=True)
    disagreement = torch.mean(torch.abs(current_3dnr - current_2dnr), dim=1, keepdim=True)
    motion = F.avg_pool2d(motion, local_kernel, stride=1, padding=local_pad)
    disagreement = F.avg_pool2d(disagreement, local_kernel, stride=1, padding=local_pad)

    motion_threshold = max(params.motion_threshold, 1e-8)
    disagreement_threshold = max(params.disagreement_threshold, 1e-8)
    motion_risk = torch.sigmoid(
        (motion - motion_threshold) / (params.soft_scale * motion_threshold)
    )
    motion_risk = F.max_pool2d(
        motion_risk, dilation_kernel, stride=1, padding=dilation_pad
    )

    disagreement_risk = torch.sigmoid(
        (disagreement - disagreement_threshold)
        / (params.soft_scale * disagreement_threshold)
    )
    joint_risk = disagreement_risk * torch.sigmoid(
        (motion - params.hard_joint_motion_scale * motion_threshold)
        / (params.soft_scale * motion_threshold)
    )
    joint_risk = F.max_pool2d(
        joint_risk, dilation_kernel, stride=1, padding=dilation_pad
    )

    soft_risk = torch.maximum(motion_risk, 0.5 * joint_risk)
    confidence = torch.clamp(1.0 - soft_risk, 0.0, 1.0)

    hard_motion = motion >= params.hard_motion_scale * motion_threshold
    hard_joint = torch.logical_and(
        motion >= params.hard_joint_motion_scale * motion_threshold,
        disagreement >= params.hard_joint_disagreement_scale * disagreement_threshold,
    )
    hard_mask = torch.logical_or(hard_motion, hard_joint).to(current_2dnr.dtype)
    hard_mask = F.max_pool2d(
        hard_mask, dilation_kernel, stride=1, padding=dilation_pad
    )
    confidence = torch.where(hard_mask > 0, torch.zeros_like(confidence), confidence)

    diagnostics = {
        "motion": motion,
        "disagreement": disagreement,
        "motion_risk": motion_risk,
        "joint_risk": joint_risk,
        "hard_mask": hard_mask,
    }
    return confidence, diagnostics


def conservative_gate(predicted_gate: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    if predicted_gate.shape != confidence.shape:
        raise ValueError(
            f"Gate and confidence shapes differ: {predicted_gate.shape} vs {confidence.shape}"
        )
    return torch.minimum(torch.clamp(predicted_gate, 0.0, 1.0), confidence)


def fuse_candidates(
    current_2dnr: torch.Tensor,
    current_3dnr: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if gate.ndim != 4 or gate.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW gate, got {gate.shape}")
    return current_2dnr + gate * (current_3dnr - current_2dnr)


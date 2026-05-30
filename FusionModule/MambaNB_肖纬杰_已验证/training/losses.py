from __future__ import annotations

"""Loss functions for staged RAW fusion training."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .tensor_utils import ensure_image3d, ensure_image4d


@dataclass(frozen=True)
class LossConfig:
    charbonnier_eps: float = 1e-3
    lambda_tv: float = 0.005
    lambda_plane: float = 0.0
    lambda_oracle: float = 0.0
    lambda_diversity: float = 0.0
    motion_weight_alpha: float = 0.0
    edge_scale: float = 10.0
    oracle_valid_threshold: float = 5e-3
    oracle_loss_mode: str = "analytic"
    soft_oracle_tau: float = 3e-5
    soft_oracle_margin: float = 1e-4
    soft_oracle_min_delta: float = 1e-6
    soft_oracle_2dnr_weight: float = 2.0
    soft_oracle_3dnr_weight: float = 0.5


def compute_loss(batch: dict[str, Tensor], output: Any, config: LossConfig) -> tuple[Tensor, dict[str, float]]:
    pred = ensure_image3d(output.prediction)
    clean = ensure_image3d(batch["clean"])

    pixel_weight = None
    if config.motion_weight_alpha > 0:
        pixel_weight = build_motion_pixel_weight(
            batch["motion_prior"],
            target_hw=clean.shape[-2:],
            alpha=config.motion_weight_alpha,
        )

    loss_rec = charbonnier_loss(pred, clean, eps=config.charbonnier_eps, weight=pixel_weight)
    loss_tv = edge_aware_tv_loss(output.packed_weight, batch["curr4"], edge_scale=config.edge_scale)
    loss_plane = (
        plane_consistency_loss(output.packed_weight)
        if config.lambda_plane > 0
        else output.packed_weight.new_zeros(())
    )

    loss_oracle = output.weight.new_zeros(())
    oracle_metrics = zero_oracle_metrics()
    if config.lambda_oracle > 0:
        if config.oracle_loss_mode == "analytic":
            loss_oracle = oracle_weight_loss(
                output.weight,
                batch["dnr2"],
                batch["dnr3"],
                batch["clean"],
                eps=config.charbonnier_eps,
                valid_threshold=config.oracle_valid_threshold,
            )
        elif config.oracle_loss_mode == "soft_winner":
            loss_oracle, oracle_metrics = masked_soft_winner_loss(
                output.weight,
                batch["dnr2"],
                batch["dnr3"],
                batch["clean"],
                eps=config.charbonnier_eps,
                tau=config.soft_oracle_tau,
                margin=config.soft_oracle_margin,
                min_delta=config.soft_oracle_min_delta,
                dnr2_better_weight=config.soft_oracle_2dnr_weight,
                dnr3_better_weight=config.soft_oracle_3dnr_weight,
            )
        else:
            raise ValueError(f"Unsupported oracle_loss_mode: {config.oracle_loss_mode}")

    loss_diversity = diversity_loss(output.weight) if config.lambda_diversity > 0 else output.weight.new_zeros(())

    loss = (
        loss_rec
        + config.lambda_tv * loss_tv
        + config.lambda_plane * loss_plane
        + config.lambda_oracle * loss_oracle
        + config.lambda_diversity * loss_diversity
    )

    return loss, {
        "loss": float(loss.detach().item()),
        "loss_rec": float(loss_rec.detach().item()),
        "loss_tv": float(loss_tv.detach().item()),
        "loss_plane": float(loss_plane.detach().item()),
        "loss_oracle": float(loss_oracle.detach().item()),
        "loss_diversity": float(loss_diversity.detach().item()),
        **oracle_metrics,
    }


def zero_oracle_metrics() -> dict[str, float]:
    return {
        "oracle_valid_frac": 0.0,
        "oracle_confidence_mean": 0.0,
        "oracle_weighted_confidence_mean": 0.0,
        "oracle_target_mean": 0.0,
        "oracle_target_std": 0.0,
        "oracle_dnr2_better_frac": 0.0,
        "oracle_dnr3_better_frac": 0.0,
        "oracle_dnr2_weighted_frac": 0.0,
        "oracle_w_minus_target_abs": 0.0,
    }


def masked_soft_winner_loss(
    weight: Tensor,
    dnr2: Tensor,
    dnr3: Tensor,
    clean: Tensor,
    eps: float,
    tau: float,
    margin: float,
    min_delta: float,
    dnr2_better_weight: float,
    dnr3_better_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Masked soft winner supervision for the 3DNR fusion weight."""
    if tau <= 0.0:
        raise ValueError("soft oracle tau must be > 0.")
    if margin <= 0.0:
        raise ValueError("soft oracle margin must be > 0.")
    if min_delta < 0.0:
        raise ValueError("soft oracle min_delta must be >= 0.")

    weight = ensure_image4d(weight).float()
    dnr2 = ensure_image4d(dnr2).float()
    dnr3 = ensure_image4d(dnr3).float()
    clean = ensure_image4d(clean).float()

    err2 = (dnr2 - clean) ** 2
    err3 = (dnr3 - clean) ** 2
    delta = err2 - err3
    abs_delta = delta.abs()
    valid = abs_delta > min_delta

    # Positive delta means 3DNR has lower squared error, so W should approach 1.
    target = torch.sigmoid(torch.clamp(delta / tau, min=-60.0, max=60.0)).detach()
    confidence = torch.clamp(abs_delta / margin, min=0.0, max=1.0)
    direction_weight = torch.where(
        delta < 0.0,
        torch.as_tensor(dnr2_better_weight, device=delta.device, dtype=delta.dtype),
        torch.as_tensor(dnr3_better_weight, device=delta.device, dtype=delta.dtype),
    )
    pixel_weight = confidence * direction_weight * valid.float()
    pixel_weight_sum = pixel_weight.sum()

    loss_map = torch.sqrt((weight - target) ** 2 + eps**2)
    loss = (loss_map * pixel_weight).sum() / pixel_weight_sum.clamp_min(1e-12)

    with torch.no_grad():
        dnr2_better = delta < 0.0
        dnr3_better = delta > 0.0
        safe_sum = pixel_weight_sum.clamp_min(1e-12)
        weighted_dnr2 = (pixel_weight * dnr2_better.float()).sum()
        weighted_abs_error = ((weight.detach() - target).abs() * pixel_weight).sum()
        metrics = {
            "oracle_valid_frac": float(valid.float().mean().item()),
            "oracle_confidence_mean": float(confidence.mean().item()),
            "oracle_weighted_confidence_mean": float(pixel_weight.mean().item()),
            "oracle_target_mean": float(target.mean().item()),
            "oracle_target_std": float(target.std(unbiased=False).item()),
            "oracle_dnr2_better_frac": float(dnr2_better.float().mean().item()),
            "oracle_dnr3_better_frac": float(dnr3_better.float().mean().item()),
            "oracle_dnr2_weighted_frac": float((weighted_dnr2 / safe_sum).item()),
            "oracle_w_minus_target_abs": float((weighted_abs_error / safe_sum).item()),
        }

    return loss, metrics


def charbonnier_loss(pred: Tensor, target: Tensor, eps: float = 1e-3, weight: Tensor | None = None) -> Tensor:
    loss_map = torch.sqrt((pred - target) ** 2 + eps**2)
    if weight is not None:
        loss_map = loss_map * weight
    return loss_map.mean()


def edge_aware_tv_loss(packed_weight: Tensor, curr4: Tensor, edge_scale: float = 10.0) -> Tensor:
    grad_w_x = torch.abs(packed_weight[..., :, 1:] - packed_weight[..., :, :-1])
    grad_w_y = torch.abs(packed_weight[..., 1:, :] - packed_weight[..., :-1, :])

    grad_img_x = torch.abs(curr4[..., :, 1:] - curr4[..., :, :-1])
    grad_img_y = torch.abs(curr4[..., 1:, :] - curr4[..., :-1, :])
    if packed_weight.shape[1] != curr4.shape[1]:
        grad_img_x = grad_img_x.mean(dim=1, keepdim=True)
        grad_img_y = grad_img_y.mean(dim=1, keepdim=True)

    gate_x = torch.exp(-edge_scale * grad_img_x.detach())
    gate_y = torch.exp(-edge_scale * grad_img_y.detach())
    return (gate_x * grad_w_x).mean() + (gate_y * grad_w_y).mean()


def plane_consistency_loss(packed_weight: Tensor) -> Tensor:
    if packed_weight.shape[1] != 4:
        return packed_weight.new_zeros(())
    mean_weight = packed_weight.mean(dim=1, keepdim=True)
    return ((packed_weight - mean_weight) ** 2).mean()


def oracle_weight_loss(
    weight: Tensor,
    dnr2: Tensor,
    dnr3: Tensor,
    clean: Tensor,
    eps: float,
    valid_threshold: float,
) -> Tensor:
    dnr2 = ensure_image4d(dnr2)
    dnr3 = ensure_image4d(dnr3)
    clean = ensure_image4d(clean)

    denom = dnr3 - dnr2
    valid = torch.abs(denom) > valid_threshold
    if not valid.any():
        return weight.new_zeros(())

    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    oracle = torch.clamp((clean - dnr2) / safe_denom, 0.0, 1.0)
    loss_map = torch.sqrt((weight - oracle) ** 2 + eps**2)
    return loss_map[valid].mean()


def diversity_loss(weight: Tensor, eps: float = 1e-6) -> Tensor:
    weight = torch.clamp(weight, eps, 1.0 - eps)
    return -(weight * torch.log(weight) + (1.0 - weight) * torch.log(1.0 - weight)).mean()


def build_motion_pixel_weight(motion_prior: Tensor, target_hw: tuple[int, int], alpha: float) -> Tensor:
    motion = motion_prior.mean(dim=1)
    flat = motion.flatten(1)
    q50 = torch.quantile(flat.detach(), 0.50, dim=1).view(-1, 1, 1)
    q95 = torch.quantile(flat.detach(), 0.95, dim=1).view(-1, 1, 1)
    motion = ((motion - q50) / (q95 - q50 + 1e-6)).clamp(0.0, 1.0)
    motion = motion.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    motion = motion[..., : target_hw[0], : target_hw[1]]
    return 1.0 + alpha * motion.detach()

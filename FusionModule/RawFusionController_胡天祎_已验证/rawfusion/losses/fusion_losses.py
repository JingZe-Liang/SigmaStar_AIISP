from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def image_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pdx, pdy = image_gradients(pred)
    tdx, tdy = image_gradients(target)
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


def tv_loss(alpha: torch.Tensor) -> torch.Tensor:
    dx, dy = image_gradients(alpha)
    return dx.abs().mean() + dy.abs().mean()


def oracle_alpha3d_target(dnr2: torch.Tensor, dnr3: torch.Tensor, clean: torch.Tensor, tau: float = 0.015) -> torch.Tensor:
    """Soft target: 1 when 3DNR is closer to clean, 0 when 2DNR is closer."""
    e2 = (dnr2 - clean).abs()
    e3 = (dnr3 - clean).abs()
    # positive => 3dnr better. tau controls softness in normalized RAW units.
    logits = (e2 - e3) / max(float(tau), 1e-6)
    return torch.sigmoid(logits).detach()


def motion_alpha3d_target(x: torch.Tensor) -> torch.Tensor:
    """From input feature channel 2 = |curr-prev|. Static -> alpha3d high; motion -> alpha3d low."""
    if x.shape[1] < 3:
        return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    motion = x[:, 2:3]
    denom = motion.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    m = (motion / denom).clamp(0.0, 1.0)
    return (1.0 - m).detach()


def fusion_loss(
    fused: torch.Tensor,
    clean: torch.Tensor,
    alpha3d: torch.Tensor,
    dnr2: torch.Tensor,
    dnr3: torch.Tensor,
    x: torch.Tensor,
    lam_grad: float = 0.15,
    lam_tv: float = 0.002,
    lam_motion: float = 0.01,
    lam_oracle: float = 0.03,
    oracle_tau: float = 0.015,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rec = charbonnier(fused - clean).mean()
    grad = gradient_loss(fused, clean)
    tv = tv_loss(alpha3d)
    motion_t = motion_alpha3d_target(x)
    motion = F.l1_loss(alpha3d, motion_t)
    oracle_t = oracle_alpha3d_target(dnr2, dnr3, clean, tau=oracle_tau)
    with torch.amp.autocast("cuda", enabled=False):
        oracle = F.binary_cross_entropy(
            alpha3d.float().clamp(1e-4, 1 - 1e-4),
            oracle_t.float()
        )
    total = rec + lam_grad * grad + lam_tv * tv + lam_motion * motion + lam_oracle * oracle
    metrics = {
        "loss": float(total.detach().cpu().item()),
        "rec": float(rec.detach().cpu().item()),
        "grad": float(grad.detach().cpu().item()),
        "tv": float(tv.detach().cpu().item()),
        "motion": float(motion.detach().cpu().item()),
        "oracle": float(oracle.detach().cpu().item()),
    }
    return total, metrics

"""Frame metrics and fixed candidate baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def baseline_predictions(denoised: torch.Tensor, fused: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the two candidates and their fixed 0.5 average."""
    _validate_packed_tensor("denoised", denoised)
    _validate_packed_tensor("fused", fused)
    if denoised.shape != fused.shape:
        raise ValueError("denoised and fused must have the same shape")
    if denoised.dtype != fused.dtype:
        raise ValueError("denoised and fused must have the same dtype")
    if denoised.device != fused.device:
        raise ValueError("denoised and fused must be on the same device")
    return {
        "denoised": denoised,
        "fused": fused,
        "average": 0.5 * (denoised + fused),
    }


def compute_frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute masked MAE/MSE/PSNR and packed-gradient MAE."""
    _validate_packed_tensor("prediction", prediction)
    _validate_packed_tensor("target", target)
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid mask must be a torch.Tensor")
    if prediction.shape != target.shape or prediction.shape != valid_mask.shape:
        raise ValueError("prediction, target, and valid mask must have the same shape")
    if prediction.dtype != target.dtype:
        raise ValueError("prediction and target must have the same dtype")
    if prediction.device != target.device or prediction.device != valid_mask.device:
        raise ValueError("prediction, target, and valid mask must be on the same device")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid mask must be boolean")
    if not bool(valid_mask.any()):
        raise ValueError("valid mask is empty")
    if not bool(torch.isfinite(prediction[valid_mask]).all()) or not bool(
        torch.isfinite(target[valid_mask]).all()
    ):
        raise ValueError("valid metric pixels must be finite")

    error = prediction - target
    mae = float(error.abs()[valid_mask].mean().item())
    mse_value = float(error.square()[valid_mask].mean().item())
    psnr = math.inf if mse_value == 0.0 else -10.0 * math.log10(mse_value)

    horizontal_mask = valid_mask[..., :, 1:] & valid_mask[..., :, :-1]
    vertical_mask = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]
    gradient_terms: list[torch.Tensor] = []
    if bool(horizontal_mask.any()):
        prediction_gradient = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_gradient = target[..., :, 1:] - target[..., :, :-1]
        gradient_terms.append((prediction_gradient - target_gradient).abs()[horizontal_mask])
    if bool(vertical_mask.any()):
        prediction_gradient = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_gradient = target[..., 1:, :] - target[..., :-1, :]
        gradient_terms.append((prediction_gradient - target_gradient).abs()[vertical_mask])
    gradient_mae = float(torch.cat(gradient_terms).mean().item()) if gradient_terms else 0.0
    return {"mae": mae, "mse": mse_value, "psnr": psnr, "gradient_mae": gradient_mae}


def _validate_packed_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 4:
        raise ValueError(f"{name} must have shape [B, 4, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating-point")


class MetricAccumulator:
    """Collect per-frame metric dictionaries and compute their means."""

    def __init__(self) -> None:
        self.frames: list[dict[str, float]] = []

    @property
    def values(self) -> list[dict[str, float]]:
        return self.frames

    def add(self, metrics: Mapping[str, float]) -> None:
        self.frames.append({name: float(value) for name, value in metrics.items()})

    def compute(self) -> dict[str, float]:
        names = {name for frame in self.frames for name in frame}
        summary: dict[str, float] = {}
        for name in sorted(names):
            values = [frame[name] for frame in self.frames if name in frame]
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                summary[name] = sum(finite) / len(finite)
            elif any(value == math.inf for value in values):
                summary[name] = math.inf
            elif any(value == -math.inf for value in values):
                summary[name] = -math.inf
            else:
                summary[name] = math.nan
        return summary

    summary = compute

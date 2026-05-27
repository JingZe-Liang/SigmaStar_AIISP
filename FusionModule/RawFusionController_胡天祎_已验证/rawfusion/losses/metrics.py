from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class SSEMetric:
    sse_norm: float = 0.0
    n: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        err = (pred.detach() - target.detach()).float()
        self.sse_norm += float((err * err).sum().cpu().item())
        self.n += int(err.numel())

    def psnr(self, max_i_code: float = 4095.0, storage_max: float = 4095.0) -> float:
        if self.n <= 0:
            return 99.0
        mse_norm = self.sse_norm / float(self.n)
        if mse_norm <= 1e-20:
            return 99.0
        mse_code = mse_norm * storage_max * storage_max
        return float(10.0 * math.log10((max_i_code * max_i_code) / mse_code))

    def mse_code(self, storage_max: float = 4095.0) -> float:
        if self.n <= 0:
            return float("nan")
        return (self.sse_norm / float(self.n)) * storage_max * storage_max


def psnr_tensor(pred: torch.Tensor, target: torch.Tensor, max_i_code: float = 4095.0, storage_max: float = 4095.0) -> float:
    metric = SSEMetric()
    metric.update(pred, target)
    return metric.psnr(max_i_code=max_i_code, storage_max=storage_max)


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"SSIM expects NCHW tensor, got ndim={pred.ndim}")
    c = pred.shape[1]
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.expand(c, 1, window_size, window_size).contiguous()
    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(target, kernel, padding=window_size // 2, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=c) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    out = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return float(out.mean().detach().cpu().item())


def summarize_metrics(metrics: Dict[str, SSEMetric], max_i_code: float, storage_max: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, m in metrics.items():
        out[f"{name}_psnr"] = m.psnr(max_i_code=max_i_code, storage_max=storage_max)
        out[f"{name}_mse"] = m.mse_code(storage_max=storage_max)
    return out

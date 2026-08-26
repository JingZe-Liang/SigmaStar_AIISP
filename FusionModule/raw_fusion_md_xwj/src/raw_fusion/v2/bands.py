"""Fixed packed-domain frequency kernels."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


H1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
H2 = torch.tensor([1.0, 0.0, 4.0, 0.0, 6.0, 0.0, 4.0, 0.0, 1.0], dtype=torch.float32) / 16.0


def _separable_reflect_conv(x: Tensor, kernel_1d: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError("frequency kernels require [N,C,H,W] tensors")
    value = x.to(dtype=torch.float32)
    channels = value.shape[1]
    kernel = torch.outer(kernel_1d, kernel_1d).to(device=value.device, dtype=value.dtype)
    weight = kernel[None, None].expand(channels, 1, *kernel.shape).contiguous()
    radius = kernel.shape[-1] // 2
    padded = F.pad(value, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, weight, groups=channels)


def a1(x: Tensor) -> Tensor:
    return _separable_reflect_conv(x, H1)


def a2(x: Tensor) -> Tensor:
    return _separable_reflect_conv(a1(x), H2)


def low_pass(x: Tensor) -> Tensor:
    return a2(x)


def b2(x: Tensor) -> Tensor:
    first = a1(x)
    return first - _separable_reflect_conv(first, H2)

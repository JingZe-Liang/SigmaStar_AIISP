from __future__ import annotations

"""Small tensor/runtime helpers shared by training modules."""

import random
from typing import Any

import numpy as np
import torch
from torch import Tensor


def ensure_image3d(x: Tensor) -> Tensor:
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected image tensor [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def ensure_image4d(x: Tensor) -> Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected image tensor [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None) -> torch.amp.autocast_mode.autocast:
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_amp_dtype(amp_arg: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or amp_arg == "none":
        return None
    if amp_arg == "bf16":
        return torch.bfloat16
    if amp_arg == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

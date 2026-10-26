"""Device selection shared by training, fitting and tests."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import torch


def device_info() -> dict[str, Any]:
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": mps_available,
    }


def resolve_device(request: str | None = None) -> torch.device:
    """Resolve explicit device, environment override, or CUDA>MPS>CPU auto mode."""
    requested = (request or os.environ.get("TASK3_DEVICE", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    elif requested in {"cuda", "mps", "cpu"}:
        selected = requested
    else:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")

    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    if selected == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device(selected)


def autocast_context(
    device: torch.device, enabled: bool = True, dtype: torch.dtype | None = None
):
    """Use CUDA AMP with a wide-range dtype when the device supports it."""
    if enabled and device.type == "cuda":
        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_torch_rng_state(device: torch.device):
    """Restore global Torch RNG streams after deterministic validation.

    Validation deliberately seeds its synthetic observations so every
    checkpoint sees the same scene-9 noise.  Without this guard, that seed
    also restarts the following training interval's augmentation/noise stream.
    MPS has its own generator state, so saving CPU state alone is insufficient
    on the local backend.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = (
        torch.mps.get_rng_state()
        if device.type == "mps" and hasattr(torch.mps, "get_rng_state")
        else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)

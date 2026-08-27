"""Deterministic RAW packing and signal normalization for V2."""
from __future__ import annotations

import operator
from typing import Any

import numpy as np


def _levels(black_level: int, white_level: int) -> tuple[int, int]:
    black = operator.index(black_level)
    white = operator.index(white_level)
    if black < 0 or white <= black or white > np.iinfo(np.uint16).max:
        raise ValueError("black_level and white_level must satisfy 0 <= black < white <= 65535")
    return black, white


def pack_rggb_v2(mosaic: Any) -> np.ndarray:
    """Pack an even RGGB mosaic as ``[R, Gr, B, Gb]``.

    The blue plane deliberately precedes Gb.  This differs from a few legacy
    helpers in the repository and is part of the V2 raw contract.
    """
    array = np.asarray(mosaic)
    if array.ndim != 2:
        raise ValueError("mosaic must be two-dimensional")
    height, width = array.shape
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("mosaic dimensions must be positive and even")
    return np.ascontiguousarray(
        np.stack(
            (array[0::2, 0::2], array[0::2, 1::2], array[1::2, 1::2], array[1::2, 0::2]),
            axis=0,
        )
    )


def normalize_signal(
    signal: Any, black_level: int, white_level: int, *, clip: bool = True
) -> Any:
    """Black-subtract and scale a signal to float32 normalized RAW.

    ``clip=False`` is used by the causal noise estimator, which must retain
    out-of-range residuals for its validity accounting.
    """
    black, white = _levels(black_level, white_level)
    try:
        import torch
    except ImportError:  # pragma: no cover - numpy is the supported baseline
        torch = None
    if torch is not None and isinstance(signal, torch.Tensor):
        value = signal.to(dtype=torch.float32)
        result = (value - float(black)) / float(white - black)
        return result.clamp(0.0, 1.0) if clip else result
    value = np.asarray(signal, dtype=np.float32)
    result = (value - np.float32(black)) / np.float32(white - black)
    if clip:
        result = np.clip(result, 0.0, 1.0)
    return np.ascontiguousarray(result, dtype=np.float32)

from __future__ import annotations

import cv2
import numpy as np


def apply_white_balance_rggb(mosaic: np.ndarray, gains: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(gains) != 3:
        raise ValueError("White-balance gains must be [R, G, B]")
    balanced = mosaic.copy()
    balanced[0::2, 0::2] *= float(gains[0])
    balanced[0::2, 1::2] *= float(gains[1])
    balanced[1::2, 0::2] *= float(gains[1])
    balanced[1::2, 1::2] *= float(gains[2])
    return balanced


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def simple_isp(
    raw: np.ndarray,
    *,
    black: float,
    white: float,
    wb: list[float] | tuple[float, ...],
    exposure: float,
) -> np.ndarray:
    """Render RGGB uint16 RAW through one deterministic comparison ISP."""

    if raw.ndim != 2:
        raise ValueError(f"Expected HxW RAW mosaic, got {raw.shape}")
    linear = (raw.astype(np.float32) - float(black)) / float(white - black)
    linear = apply_white_balance_rggb(linear, wb)
    linear = np.clip(linear * float(exposure), 0.0, 1.0)
    mosaic_u16 = np.round(linear * 65535.0).astype(np.uint16)

    # OpenCV's BG enum maps a top-left R RGGB mosaic to BGR output.
    bgr_u16 = cv2.cvtColor(mosaic_u16, cv2.COLOR_BayerBG2BGR)
    bgr = bgr_u16.astype(np.float32) / 65535.0
    bgr = linear_to_srgb(bgr)
    return np.round(np.clip(bgr, 0.0, 1.0) * 255.0).astype(np.uint8)


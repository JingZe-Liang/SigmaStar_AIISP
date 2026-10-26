"""Team-standard OpenCV ISP for RAW previews and RGB metrics."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from noise_model.raw_io import unpack_bayer


def _mosaic_to_rgb_float(raw_dn: np.ndarray, wb_gains: tuple[float, float] | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Apply the supplied TeamStandardISP, returning RGB float in [0, 1]."""
    if raw_dn.ndim != 3:
        raise ValueError("raw_dn must have shape [N,H,W]")
    values = (raw_dn.astype(np.float32) - 16.0) / (4095.0 - 16.0)
    values = np.clip(values, 0.0, 1.0)
    r_slice = (slice(1, None, 2), slice(0, None, 2))
    b_slice = (slice(0, None, 2), slice(1, None, 2))
    g1_slice = (slice(0, None, 2), slice(0, None, 2))
    g2_slice = (slice(1, None, 2), slice(1, None, 2))
    if wb_gains is None:
        green_mean = (float(values[0][g1_slice].mean()) + float(values[0][g2_slice].mean())) / 2.0
        r_gain = green_mean / (float(values[0][r_slice].mean()) + 1e-6)
        b_gain = green_mean / (float(values[0][b_slice].mean()) + 1e-6)
    else:
        r_gain, b_gain = wb_gains
    values[:, r_slice[0], r_slice[1]] *= r_gain
    values[:, b_slice[0], b_slice[1]] *= b_gain
    values = np.clip(values, 0.0, 1.0)
    images = []
    for value in values:
        # OpenCV's legacy GB short name aliases GRBG, not physical GBRG.
        # Use the explicit CFA name to preserve canonical RAW R/G/B colors.
        bgr = cv2.cvtColor((value * 65535.0).astype(np.uint16), cv2.COLOR_BayerGBRG2BGR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1].astype(np.float32) / 65535.0)
        images.append(np.power(rgb, 1.0 / 2.2))
    return np.stack(images), (r_gain, b_gain)


def packed_to_linear_rgb(
    packed: torch.Tensor,
    cfa: str,
    wb_gains: tuple[float, float] | None = None,
    return_wb: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[float, float]]:
    """Apply TeamStandardISP to normalized packed R,G1,G2,B tensors."""
    if packed.ndim != 4 or packed.shape[1] != 4:
        raise ValueError("packed RAW must have shape [B,4,H,W]")
    if cfa.upper() != "GBRG":
        raise ValueError("TeamStandardISP is defined for GBRG only")
    packed_dn = np.rint(packed.detach().float().cpu().numpy().clip(0, 1) * 4095.0).astype(np.uint16)
    mosaic = unpack_bayer(packed_dn, cfa.upper())
    rgb, gains = _mosaic_to_rgb_float(mosaic, wb_gains)
    result = torch.from_numpy(np.moveaxis(rgb, -1, 1)).to(packed.device)
    return (result, gains) if return_wb else result


def linear_to_srgb(rgb: torch.Tensor, red_gain: float = 1.0, blue_gain: float = 1.0) -> torch.Tensor:
    """Compatibility wrapper: ISP already applies AWB and gamma."""
    del red_gain, blue_gain
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB must have shape [B,3,H,W]")
    return rgb.clamp(0.0, 1.0)

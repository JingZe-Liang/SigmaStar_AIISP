"""Motion-aware repeated-frame preprocessing for RAW noise calibration.

Adjacent-frame variance identities are valid only after signal alignment.  A
global phase-correlation shift is estimated on packed RAW luminance; pixels
whose aligned residual is inconsistent with the robust temporal noise scale
are excluded as local motion.  The routine reports the usable fraction so a
calibration cannot silently absorb scene motion into its noise parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _phase_shift(reference: np.ndarray, moving: np.ndarray, max_shift: int) -> tuple[int, int]:
    """Return integer (dy, dx) to roll ``moving`` onto ``reference``."""
    # Keep full packed resolution: a one-pixel motion is already comparable
    # to the measured DN noise and would disappear under coarse sampling.
    ref = reference.astype(np.float64)
    mov = moving.astype(np.float64)
    ref -= np.mean(ref)
    mov -= np.mean(mov)
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shifts = []
    for coordinate, size in zip(peak, correlation.shape):
        value = int(coordinate)
        if value > size // 2:
            value -= size
        shifts.append(int(np.clip(value, -max_shift, max_shift)))
    return shifts[0], shifts[1]


def _valid_mask(height: int, width: int, shift: tuple[int, int]) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    dy, dx = shift
    if dy > 0:
        mask[:dy] = False
    elif dy < 0:
        mask[dy:] = False
    if dx > 0:
        mask[:, :dx] = False
    elif dx < 0:
        mask[:, dx:] = False
    return mask


def align_pair(
    frames: np.ndarray,
    max_shift: int = 32,
    motion_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align two packed frames and return the motion-stable pixel mask.

    ``frames`` is ``[2,4,H,W]`` in decoded DN.  The returned difference is
    still computed by the caller, so this function cannot alter the DN unit.
    The mask is conservative: local changes above a robust noise threshold are
    excluded rather than treated as sensor noise.
    """
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 2 or values.shape[1] != 4:
        raise ValueError("frames must have shape [2,4,H,W]")
    luminance = values.mean(axis=1)
    shift = _phase_shift(luminance[0], luminance[1], max_shift)
    aligned = np.roll(values[1], shift=shift, axis=(-2, -1))
    stable = _valid_mask(values.shape[-2], values.shape[-1], shift)
    residual = luminance[0] - aligned.mean(axis=0)
    valid_residual = residual[stable]
    center = float(np.median(valid_residual)) if valid_residual.size else 0.0
    mad = float(np.median(np.abs(valid_residual - center))) if valid_residual.size else 0.0
    scale = max(1.4826 * mad, 1e-3)
    stable &= np.abs(residual - center) <= float(motion_sigma) * scale
    return aligned, stable, {
        "shift_y_px": int(shift[0]),
        "shift_x_px": int(shift[1]),
        "stable_fraction": float(stable.mean()),
        "motion_rejected_fraction": float(1.0 - stable.mean()),
        "robust_residual_scale_dn": scale,
    }

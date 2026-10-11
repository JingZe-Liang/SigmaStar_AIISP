"""Cross-gain whole-stream registration.

The camera was bumped between captures: shdarkroom 632x/1000x sit ~63 sensor
px left of the other gains (survey: 62.7/62.1 px alignment shifts). Alignment
is measured once per stream on the temporal means, stored in the manifest,
and applied downstream as an integer packed-resolution crop. Streams whose
alignment peak is weak or ambiguous are marked unpaired and discarded for
pairing.

Method: exhaustive integer-shift search maximizing full-overlap NCC via
FFT-based matchTemplate (immune to the spurious near-zero phase-correlation
peak caused by per-capture re-lighting), subpixel refinement by parabolic
interpolation on the response map, and a final NCC verification on the true
overlap crop.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from raw_fusion.finetune.stream_io import CHANNEL_NAMES

# Integer-shift search bounds in packed pixels (sensor px = 2x). The known
# camera bump is ~31 packed px; these bounds leave a wide margin.
_SEARCH_X_PACKED = 80
_SEARCH_Y_PACKED = 8
# Post-alignment NCC on the true overlap must reach this (A0 initial value;
# observed: correctly aligned streams 0.43 (632x, relighting-capped) and 0.80
# (10x), while the best wrong shift scores ~0.22).
MIN_ALIGNMENT_NCC = 0.35


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """Whole-stream alignment of one stream against its scene baseline.

    shift_x/shift_y are the content displacement of the moving stream relative
    to the baseline in SENSOR pixels (survey-comparable: the survey's "+62.7 px
    alignment shift" for 632x is a content displacement of -62.7 here).
    integer_shift_x/y are the packed-half-resolution displacement used
    operationally to crop packed arrays into the baseline frame.
    """

    baseline: str
    shift_x: float
    shift_y: float
    integer_shift_x: int
    integer_shift_y: int
    ncc: float  # normalized cross-correlation after the integer shift (true overlap)
    paired: bool
    reason: str  # "ok" or why the stream was marked unpaired


def _gray_plane(mean: np.ndarray, black: dict[str, float]) -> np.ndarray:
    """Black-subtracted luminance proxy from the two green channels."""

    green = 0.5 * (mean[CHANNEL_NAMES.index("Gr")] + mean[CHANNEL_NAMES.index("Gb")])
    black_level = 0.5 * (black["Gr"] + black["Gb"])
    return np.maximum(green - black_level, 0.0).astype(np.float32)


def register_streams(
    baseline_mean: np.ndarray,
    baseline_black: dict[str, float],
    moving_mean: np.ndarray,
    moving_black: dict[str, float],
    baseline_label: str,
) -> RegistrationResult:
    """Measure the whole-stream content displacement of `moving` vs `baseline`
    from their packed temporal means [4, h, w]."""

    base_gray = _gray_plane(baseline_mean, baseline_black)
    moving_gray = _gray_plane(moving_mean, moving_black)
    int_x, int_y, sub_x, sub_y = _best_integer_shift(base_gray, moving_gray)

    shift_x = 2.0 * (int_x + sub_x)
    shift_y = 2.0 * (int_y + sub_y)
    base_crop, moving_crop = overlap_crops(base_gray, moving_gray, int_x, int_y)
    ncc = normalized_cross_correlation(base_crop, moving_crop)
    paired = ncc >= MIN_ALIGNMENT_NCC
    return RegistrationResult(
        baseline=baseline_label,
        shift_x=float(shift_x),
        shift_y=float(shift_y),
        integer_shift_x=int_x,
        integer_shift_y=int_y,
        ncc=float(ncc),
        paired=paired,
        reason="ok" if paired else f"post-shift NCC {ncc:.3f} < {MIN_ALIGNMENT_NCC}",
    )


def _best_integer_shift(
    base_gray: np.ndarray, moving_gray: np.ndarray
) -> tuple[int, int, float, float]:
    """Search the integer shift maximizing full-overlap NCC, with subpixel
    parabolic refinement.

    Returns (int_x, int_y, sub_x, sub_y); shifts in packed pixels, subpixel
    terms in [-0.5, 0.5].
    """

    padded = np.pad(
        moving_gray,
        ((_SEARCH_Y_PACKED, _SEARCH_Y_PACKED), (_SEARCH_X_PACKED, _SEARCH_X_PACKED)),
        mode="reflect",
    )
    response = cv2.matchTemplate(padded, base_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, _max_val, _min_loc, (peak_x, peak_y) = cv2.minMaxLoc(response)
    sub_x = _parabolic_offset(response[peak_y, :], peak_x)
    sub_y = _parabolic_offset(response[:, peak_x], peak_y)
    return peak_x - _SEARCH_X_PACKED, peak_y - _SEARCH_Y_PACKED, sub_x, sub_y


def _parabolic_offset(profile: np.ndarray, index: int) -> float:
    """Subpixel peak offset along one axis via 3-point parabolic fit."""

    if index <= 0 or index >= profile.size - 1:
        return 0.0
    left, center, right = float(profile[index - 1]), float(profile[index]), float(profile[index + 1])
    denominator = left - 2.0 * center + right
    if denominator == 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def overlap_crops(base: np.ndarray, moving: np.ndarray, shift_x: int, shift_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop two planes to their overlap given the content displacement
    (shift_x, shift_y) of `moving` relative to `base`.

    Baseline content at (y, x) appears in the moving plane at
    (y + shift_y, x + shift_x); both returned crops show the same content.
    """

    height, width = base.shape
    base_y0 = max(0, -shift_y)
    base_y1 = min(height, height - shift_y)
    base_x0 = max(0, -shift_x)
    base_x1 = min(width, width - shift_x)
    base_crop = base[base_y0:base_y1, base_x0:base_x1]
    moving_crop = moving[base_y0 + shift_y : base_y1 + shift_y, base_x0 + shift_x : base_x1 + shift_x]
    if base_crop.size == 0:
        raise ValueError(f"shift ({shift_x}, {shift_y}) leaves no overlap")
    return base_crop, moving_crop


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two same-shape planes."""

    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def aligned_packed_pair(
    baseline_mean: np.ndarray, moving_mean: np.ndarray, shift_x: int, shift_y: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Crop two packed temporal means [4, h, w] to their overlap given the
    content displacement of the moving stream.

    Returns (baseline_crop, moving_crop, y0, x0) where (y0, x0) is the origin
    of the crop inside the baseline frame, so callers can crop auxiliary maps
    (masks, confidence) over the identical region.
    """

    base_channels: list[np.ndarray] = []
    moving_channels: list[np.ndarray] = []
    y0 = max(0, -shift_y)
    x0 = max(0, -shift_x)
    for c in range(baseline_mean.shape[0]):
        base_crop, moving_crop = overlap_crops(baseline_mean[c], moving_mean[c], shift_x, shift_y)
        base_channels.append(base_crop)
        moving_channels.append(moving_crop)
    return np.stack(base_channels), np.stack(moving_channels), y0, x0

"""Static-region per-channel exposure normalization fits.

Every re-light between gain captures changed the scene brightness (survey:
10x chart signal 1091 DN vs 645x 194 DN), so each (noisy, reference) pair
needs a per-channel linear map ref ~ slope * noisy + intercept fitted on
long-term static, non-saturated pixels before the noisy stream can be
supervised by the reference.

The residual distribution is honestly recorded: cross-capture relighting
moves shadows and changes the spectral mix, so beyond the robust-fit inliers
there is an irreducible per-pixel disagreement that no linear map removes.
The manifest therefore carries inlier residuals plus the inlier fraction,
and downstream stages weigh pairings accordingly (A1 prefers the same-shutter
10x/64x pairs precisely because their residual is far tighter).
"""

from dataclasses import dataclass

import numpy as np

from raw_fusion.finetune.adapters import WHITE_CLIP_DN
from raw_fusion.finetune.stream_io import CHANNEL_NAMES

# Fit only on pixels comfortably above black and below white on both sides.
SIGNAL_MARGIN_DN = 20.0
SATURATION_MARGIN_DN = 295.0  # WHITE_CLIP_DN - margin = 3800 DN cap
_MAX_FIT_PIXELS = 400_000
_OUTLIER_ROUNDS = 3
_OUTLIER_SIGMAS = 3.0


@dataclass(frozen=True, slots=True)
class ExposureFit:
    reference: str  # gain label of the reference stream
    slopes: dict[str, float]
    intercepts: dict[str, float]
    residual_median_dn: dict[str, float]
    residual_p90_dn: dict[str, float]
    inlier_fraction: dict[str, float]  # fraction of the fit population within the final trim
    n_pixels: dict[str, int]


def fit_exposure_pair(
    noisy_mean: np.ndarray,
    noisy_black: dict[str, float],
    reference_mean: np.ndarray,
    reference_black: dict[str, float],
    static_region: np.ndarray,
    reference_label: str,
    max_pixels: int = _MAX_FIT_PIXELS,
) -> ExposureFit:
    """Fit ref ~ slope * noisy + intercept per canonical channel with iterative
    sigma-trimmed least squares; residuals are reported over the final inliers.

    `noisy_mean` / `reference_mean` are packed temporal means [4, h, w] already
    cropped to the same content; `static_region` is the boolean packed mask of
    long-term static pixels (intersection of both streams).
    """

    if noisy_mean.shape != reference_mean.shape:
        raise ValueError(f"shape mismatch {noisy_mean.shape} vs {reference_mean.shape}")
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    residual_median: dict[str, float] = {}
    residual_p90: dict[str, float] = {}
    inlier_fraction: dict[str, float] = {}
    n_pixels: dict[str, int] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        noisy = noisy_mean[channel].astype(np.float64)
        ref = reference_mean[channel].astype(np.float64)
        selected = (
            static_region
            & (noisy > noisy_black[name] + SIGNAL_MARGIN_DN)
            & (ref > reference_black[name] + SIGNAL_MARGIN_DN)
            & (noisy < WHITE_CLIP_DN - SATURATION_MARGIN_DN)
            & (ref < WHITE_CLIP_DN - SATURATION_MARGIN_DN)
        )
        x = noisy[selected]
        y = ref[selected]
        if x.size < 1000:
            raise ValueError(
                f"channel {name}: only {x.size} usable pixels for the exposure fit; "
                "static region or signal level too small"
            )
        if x.size > max_pixels:
            indices = np.linspace(0, x.size - 1, max_pixels).astype(np.intp)
            x = x[indices]
            y = y[indices]
        slope, intercept, inliers = _trimmed_line_fit(x, y)
        if slope <= 0:
            raise ValueError(f"channel {name}: exposure fit produced non-positive slope {slope:.4f}")
        residual = y[inliers] - (slope * x[inliers] + intercept)
        slopes[name] = slope
        intercepts[name] = intercept
        residual_median[name] = float(np.median(residual))
        residual_p90[name] = float(np.percentile(residual, 90))
        inlier_fraction[name] = float(inliers.mean())
        n_pixels[name] = int(x.size)
    return ExposureFit(
        reference=reference_label,
        slopes=slopes,
        intercepts=intercepts,
        residual_median_dn=residual_median,
        residual_p90_dn=residual_p90,
        inlier_fraction=inlier_fraction,
        n_pixels=n_pixels,
    )


def _trimmed_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least squares with iterated 3-sigma outlier rejection (3 rounds).

    Returns (slope, intercept, inlier_mask) where the mask marks the pixels
    kept by the final trimming round.
    """

    design = np.stack([x, np.ones_like(x)], axis=1)
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    inliers = np.ones(x.size, dtype=bool)
    for _ in range(_OUTLIER_ROUNDS):
        residual = y - (slope * x + intercept)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        inliers = np.abs(residual) <= _OUTLIER_SIGMAS * 1.4826 * mad
        if inliers.all() or int(inliers.sum()) < 1000:
            break
        slope, intercept = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
    return float(slope), float(intercept), inliers

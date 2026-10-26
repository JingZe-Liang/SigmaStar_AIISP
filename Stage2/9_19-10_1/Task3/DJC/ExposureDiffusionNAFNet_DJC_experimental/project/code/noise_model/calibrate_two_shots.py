"""Calibrate the CVPR 2026 dark prior and shot coefficient on MIS20S1 RAW."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .raw_io import RawSpec, count_frames, pack_bayer, read_frames
from .two_shots import SpectralDarkModel


GAINS = (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)
SPEC = RawSpec(width=1920, height=1080, bit_shift=4, valid_bits=12, cfa="GRBG")


def one_raw(root: Path, kind: str, gain: int) -> Path:
    paths = sorted((root / kind / str(gain)).glob("*.raw"))
    if len(paths) != 1:
        raise ValueError(f"expected one {kind} RAW for gain {gain}, found {len(paths)}")
    if count_frames(paths[0], SPEC) < 50:
        raise ValueError(f"expected at least 50 frames in {paths[0]}")
    return paths[0]


def exposure_us(path: Path) -> int:
    match = re.search(r"(?:^|,)US=(\d+)", path.name)
    if match is None:
        raise ValueError(f"US exposure metadata missing: {path}")
    return int(match.group(1))


def sampled_frame(path: Path, index: int, stride: int = 8) -> np.ndarray:
    frame = read_frames(path, SPEC, max_frames=1, start_frame=index)
    return pack_bayer(frame, SPEC.cfa)[0, :, ::stride, ::stride]


def temporal_bins(path: Path, black_level: float, pairs: int = 20,
                  bin_width: int = 32, start_pair: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variance from adjacent-frame differences of a static calibration scene.

    This repeated-frame estimator is an adaptation of Lu et al.'s Eq. (17).
    It removes static scene texture, unlike their single-frame 3x3 shortcut.
    """
    n_bins = 4096 // bin_width
    count = np.zeros(n_bins, dtype=np.int64)
    sums = np.zeros(n_bins, dtype=np.float64)
    sums_sq = np.zeros(n_bins, dtype=np.float64)
    for i in range(start_pair, start_pair + pairs):
        a = sampled_frame(path, 2 * i)
        b = sampled_frame(path, 2 * i + 1)
        signal = (a + b) * 0.5 - black_level
        delta = a - b
        # Correct small global exposure / bias drift channel-wise.
        delta = delta - np.median(delta, axis=(1, 2), keepdims=True)
        variance = 0.5 * delta**2
        valid = (signal >= 0) & (signal < 3800) & (variance < 1e6)
        indices = np.floor(signal[valid] / bin_width).astype(np.int32)
        count += np.bincount(indices, minlength=n_bins)
        sums += np.bincount(indices, weights=variance[valid], minlength=n_bins)
        sums_sq += np.bincount(indices, weights=variance[valid]**2, minlength=n_bins)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    stderr = np.sqrt(np.maximum(sums_sq / np.maximum(count, 1) - mean**2, 0)
                     / np.maximum(count, 1))
    return count, mean, stderr


def fit_shot_gain(color: Path, dark: Path, black_level: float) -> dict:
    # Keep five pairs for a final check.  The first fifteen pairs select the
    # intercept policy on a separate five-pair internal split.  The previous
    # implementation fixed the 83 ms dark-frame variance as the intercept for
    # every color exposure, which is not justified when the color exposure is
    # as short as 326 us.
    n, v, se = temporal_bins(color, black_level, pairs=15, start_pair=0)
    dark_n, dark_v, _ = temporal_bins(dark, black_level)
    x = (np.arange(len(n)) + 0.5) * 32.0
    # Low bins can be affected by clipping; high bins by saturation. Require
    # adequate population and use inverse uncertainty weights.
    use = (n >= 300) & (x >= 32) & (x <= 3000) & (v > 0)
    if use.sum() < 12:
        return {"shot_gain_dn": None, "reason": "too few populated signal bins"}
    weights = n[use] / np.maximum(v[use]**2, 1)
    weights /= weights.max()
    coeff = np.polyfit(x[use], v[use], deg=1, w=np.sqrt(weights))
    baseline = float(dark_v[dark_n > 300].mean()) if (dark_n > 300).any() else None
    if baseline is None:
        return {"shot_gain_dn": None, "reason": "dark variance unavailable"}
    fixed_gain = float(np.sum(weights * x[use] * (v[use] - baseline)) /
                       np.sum(weights * x[use]**2))
    weighted_mean = np.average(v[use], weights=weights)
    r2 = 1 - np.sum(weights * (v[use] - (baseline + fixed_gain * x[use]))**2) / np.sum(
        weights * (v[use] - weighted_mean)**2
    )
    free_slope, free_intercept = float(coeff[0]), float(coeff[1])
    zero_gain = float(np.sum(weights * x[use] * v[use]) /
                      np.sum(weights * x[use]**2))
    candidates = {
        "fixed_dark_intercept": (baseline, fixed_gain),
        "nonnegative_intercept": (
            max(free_intercept, 0.0), free_slope if free_slope > 0 else zero_gain
        ),
        "zero_intercept": (0.0, zero_gain),
    }

    def relative_error(start_pair: int, pair_count: int,
                       candidate: tuple[float, float], min_count: int) -> float:
        hold_n, hold_v, _ = temporal_bins(
            color, black_level, pairs=pair_count, start_pair=start_pair
        )
        hold_use = (hold_n >= min_count) & (x >= 32) & (x <= 3000) & (hold_v > 0)
        if hold_use.sum() < 12:
            return float("nan")
        hold_weights = hold_n[hold_use] / np.maximum(hold_v[hold_use] ** 2, 1)
        hold_weights /= hold_weights.max()
        intercept, slope = candidate
        prediction = intercept + slope * x[hold_use]
        return float(np.sqrt(
            np.sum(hold_weights * ((hold_v[hold_use] - prediction) /
                                   np.maximum(hold_v[hold_use], 1)) ** 2) /
            np.sum(hold_weights)
        ))

    internal_errors = {
        name: relative_error(15, 5, candidate, 75)
        for name, candidate in candidates.items()
    }
    valid_internal = {k: v for k, v in internal_errors.items() if np.isfinite(v)}
    selected_name = min(valid_internal, key=valid_internal.get)
    selected_intercept, selected_gain = candidates[selected_name]
    final_errors = {
        name: relative_error(20, 5, candidate, 75)
        for name, candidate in candidates.items()
    }
    final_n, _, _ = temporal_bins(color, black_level, pairs=5, start_pair=20)
    final_use = (final_n >= 75) & (x >= 32) & (x <= 3000)
    result = {
        "shot_gain_dn": selected_gain if selected_gain > 0 else None,
        "selected_intercept_variance_dn2": selected_intercept,
        "selected_candidate": selected_name,
        "fixed_dark_intercept_shot_gain_dn": fixed_gain if fixed_gain > 0 else None,
        "fixed_dark_intercept_variance_dn2": baseline,
        "free_fit_slope_dn": free_slope,
        "free_fit_intercept_variance_dn2": free_intercept,
        "zero_intercept_shot_gain_dn": zero_gain if zero_gain > 0 else None,
        "dark_pair_variance_dn2": baseline,
        "weighted_r2": float(r2),
        "signal_bins": int(use.sum()),
        "internal_selection_pairs": [0, 14],
        "internal_validation_pairs": [15, 19],
        "final_validation_pairs": [20, 24],
        "candidate_internal_relative_errors": internal_errors,
        "candidate_final_relative_errors": final_errors,
        "fit_signal_min_dn": float(x[use].min()),
        "fit_signal_max_dn": float(x[use].max()),
        "final_signal_min_dn": float(x[final_use].min()) if final_use.any() else None,
        "final_signal_max_dn": float(x[final_use].max()) if final_use.any() else None,
        "method": "temporal frame-pair photon-transfer fit; constrained intercept selection; Eq.17 adaptation",
    }
    result["holdout_weighted_relative_error"] = final_errors[selected_name]
    result["holdout_signal_bins"] = int(final_use.sum())
    result["status"] = (
        "provisional" if result.get("holdout_weighted_relative_error", 1) > 0.1
        or free_intercept < 0 else "heldout_checked"
    )
    if result["shot_gain_dn"] is None:
        result["reason"] = "nonpositive variance slope"
    return result


def calibrate(root: Path, output: Path, gains: tuple[int, ...],
              fit_shot: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "paper": "Lu et al., 2-Shots in the Dark, CVPR 2026",
        "method": "one dark frame per gain, sigma=50 packed pixels, K=10",
        "raw_spec": vars(SPEC),
        "fit_frame": 0,
        "holdout_frames": list(range(40, 50)),
        "gains": {},
    }
    for gain in gains:
        dark = one_raw(root, "black", gain)
        reference = pack_bayer(read_frames(dark, SPEC, max_frames=1))[0]
        model = SpectralDarkModel.fit(reference)
        model_file = output / f"mis20s1_gain{gain}_2shots.npz"
        np.savez_compressed(model_file, reference_dark_dn=reference.astype(np.uint16),
                            sigma=np.float32(50), iterations=np.int32(10))
        black_level = float(reference.mean())
        info = {
            "model_file": str(model_file.resolve()),
            "dark_path": str(dark.resolve()),
            "dark_exposure_us": exposure_us(dark),
            "reference_frame_sha256": hashlib.sha256(
                reference.astype(np.uint16).tobytes()
            ).hexdigest(),
            "reference_black_dn": black_level,
            "reference_channel_mean_dn": reference.mean(axis=(1, 2)).round(5).tolist(),
            "residual_channel_std_dn": model.reference_residual.std(axis=(1, 2)).round(5).tolist(),
            "shading_channel_range_dn": [
                [round(float(np.min(c)), 4), round(float(np.max(c)), 4)]
                for c in model.shading
            ],
        }
        if fit_shot:
            color = one_raw(root, "color", gain)
            info["color_path"] = str(color.resolve())
            info["color_exposure_us"] = exposure_us(color)
            info["shot"] = fit_shot_gain(color, dark, black_level)
        report["gains"][str(gain)] = info
        (output / "calibration.json").write_text(json.dumps(report, indent=2))
        print(f"gain {gain}: dark std {info['residual_channel_std_dn']}; "
              f"shot {info.get('shot', {}).get('shot_gain_dn')}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", nargs="+", type=int, default=GAINS)
    parser.add_argument("--no-shot", action="store_true")
    args = parser.parse_args()
    calibrate(args.calibration_root, args.output, tuple(args.gains),
              fit_shot=not args.no_shot)


if __name__ == "__main__":
    main()

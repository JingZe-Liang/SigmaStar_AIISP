"""Baseline metrics for data-first outputs against denoised frames."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    aggregate: Mapping[str, float]
    per_condition: Mapping[str, Mapping[str, float]]


def _metrics(prediction: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if pred.shape != base.shape:
        raise ValueError("prediction and denoised arrays must have equal shape")
    diff = pred - base
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    psnr = 99.0 if rmse == 0.0 else float(20.0 * np.log10(4095.0 / rmse))
    # A dependency-free global SSIM approximation, sufficient for baseline
    # comparisons when skimage is not installed.
    mean_p, mean_b = float(pred.mean()), float(base.mean())
    var_p, var_b = float(pred.var()), float(base.var())
    cov = float(np.mean((pred - mean_p) * (base - mean_b)))
    c1, c2 = 6.5025, 58.5225
    ssim = ((2 * mean_p * mean_b + c1) * (2 * cov + c2)) / ((mean_p**2 + mean_b**2 + c1) * (var_p + var_b + c2))
    temporal_prediction = float(np.mean(np.abs(np.diff(pred, axis=0)))) if pred.shape[0] > 1 else 0.0
    temporal_baseline = float(np.mean(np.abs(np.diff(base, axis=0)))) if base.shape[0] > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "psnr": psnr, "ssim": float(ssim), "temporal_prediction": temporal_prediction, "temporal_baseline": temporal_baseline, "high_frequency_delta": float(np.mean(np.abs(diff))), "fallback_fraction": 0.0}


def compare_against_denoised(prediction_root: Path, denoised: Mapping[str, np.ndarray], output_path: Path) -> ComparisonResult:
    per_condition: dict[str, dict[str, float]] = {}
    for condition, baseline in denoised.items():
        prediction = np.load(Path(prediction_root) / f"inference_{condition}.npy")
        per_condition[condition] = _metrics(prediction, baseline)
    if not per_condition:
        raise ValueError("at least one condition is required")
    keys = tuple(next(iter(per_condition.values())).keys())
    aggregate = {key: float(np.mean([values[key] for values in per_condition.values()])) for key in keys}
    payload = {"protocol": "raw_fusion_v2_data_first_comparison", "aggregate": aggregate, "per_condition": per_condition}
    Path(output_path).write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="ascii")
    return ComparisonResult(aggregate, per_condition)


__all__ = ["ComparisonResult", "compare_against_denoised"]

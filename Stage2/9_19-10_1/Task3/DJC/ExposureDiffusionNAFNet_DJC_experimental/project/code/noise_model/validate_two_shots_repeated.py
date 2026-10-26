"""One-frame fit versus all remaining 49 dark frames, per gain.

The saved CVPR 2026 prior contains ONLY frame 0.  This script reads frames
1..49 as evaluation data and samples fresh frames from the unchanged prior.
No validation statistic is used to select sigma, iterations, or the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from .calibrate_two_shots import SPEC
from .raw_io import count_frames, pack_bayer, read_frames
from .two_shots import load_calibrated_model
from .validate_two_shots import radial_power


REAL_PSD_FRAMES = {1, 5, 10, 20, 30, 40, 49}


def frame_metrics(frame: np.ndarray, shading: np.ndarray,
                  residual_mean: np.ndarray, include_psd: bool) -> dict:
    residual = frame - shading - residual_mean
    residual = residual - residual.mean(axis=(1, 2), keepdims=True)
    result = {
        "mean": frame.mean(axis=(1, 2)),
        "std": residual.std(axis=(1, 2)),
        "row_std": residual.mean(axis=2).std(axis=1),
        "col_std": residual.mean(axis=1).std(axis=1),
        "cross_corr": np.corrcoef(residual.reshape(4, -1)),
        "values": residual[:, ::8, ::8].copy(),
    }
    adc_residual = np.rint(frame) - shading - residual_mean
    adc_residual -= adc_residual.mean(axis=(1, 2), keepdims=True)
    result["adc_values"] = adc_residual[:, ::8, ::8].copy()
    if include_psd:
        result["psd"] = radial_power(residual)
    return result


def collect_real(path: Path, model) -> tuple[dict, np.ndarray, np.ndarray]:
    if count_frames(path, SPEC) != 50:
        raise ValueError(f"expected exactly 50 dark frames in {path}")
    summary = {"mean": [], "std": [], "row_std": [], "col_std": [],
               "cross_corr": [], "values": [], "adc_values": [], "psd": []}
    image_sum = np.zeros_like(model.shading, dtype=np.float64)
    image_sumsq = np.zeros_like(model.shading, dtype=np.float64)
    for index in range(1, 50):
        frame = pack_bayer(read_frames(path, SPEC, max_frames=1,
                                       start_frame=index))[0]
        stats = frame_metrics(frame, model.shading, model.residual_mean,
                              include_psd=index in REAL_PSD_FRAMES)
        for key, value in stats.items():
            summary[key].append(value)
        image_sum += frame
        image_sumsq += frame.astype(np.float64)**2
    mean_map = image_sum / 49
    temporal_var = np.maximum(image_sumsq / 49 - mean_map**2, 0) * 49 / 48
    return summary, mean_map, np.sqrt(temporal_var)


def collect_synthetic(model, gain: int, sample_count: int) -> tuple[dict, np.ndarray, np.ndarray]:
    summary = {"mean": [], "std": [], "row_std": [], "col_std": [],
               "cross_corr": [], "values": [], "adc_values": [], "psd": []}
    image_sum = np.zeros_like(model.shading, dtype=np.float64)
    image_sumsq = np.zeros_like(model.shading, dtype=np.float64)
    for index in range(sample_count):
        frame = model.sample(seed=20260927 + gain * 100 + index)
        stats = frame_metrics(frame, model.shading, model.residual_mean,
                              include_psd=True)
        for key, value in stats.items():
            summary[key].append(value)
        image_sum += frame
        image_sumsq += frame.astype(np.float64)**2
    mean_map = image_sum / sample_count
    temporal_var = np.maximum(image_sumsq / sample_count - mean_map**2, 0)
    temporal_var *= sample_count / (sample_count - 1)
    return summary, mean_map, np.sqrt(temporal_var)


def aggregate(summary: dict) -> dict:
    result = {}
    for key in ("mean", "std", "row_std", "col_std", "cross_corr", "psd"):
        values = np.asarray(summary[key])
        result[key + "_mean"] = values.mean(axis=0).round(5).tolist()
        result[key + "_sd_across_frames"] = values.std(axis=0).round(5).tolist()
    return result


def compare(real: dict, fake: dict, real_mean: np.ndarray,
            fake_mean: np.ndarray, real_temporal: np.ndarray,
            fake_temporal: np.ndarray, model) -> dict:
    real_values = np.concatenate(real["values"], axis=1).reshape(4, -1)
    fake_values = np.concatenate(fake["values"], axis=1).reshape(4, -1)
    real_adc = np.concatenate(real["adc_values"], axis=1).reshape(4, -1)
    fake_adc = np.concatenate(fake["adc_values"], axis=1).reshape(4, -1)
    real_fixed = real_mean - model.shading - model.residual_mean
    fake_fixed = fake_mean - model.shading - model.residual_mean
    real_fixed -= real_fixed.mean(axis=(1, 2), keepdims=True)
    fake_fixed -= fake_fixed.mean(axis=(1, 2), keepdims=True)
    psd_real = np.mean(real["psd"], axis=0)
    psd_fake = np.mean(fake["psd"], axis=0)
    real_frame_mean = np.array(real["mean"])
    fake_frame_mean = np.array(fake["mean"])
    return {
        "residual_wasserstein_dn": [
            round(float(wasserstein_distance(real_values[c], fake_values[c])), 4)
            for c in range(4)
        ],
        "adc_quantized_residual_wasserstein_dn": [
            round(float(wasserstein_distance(real_adc[c], fake_adc[c])), 4)
            for c in range(4)
        ],
        "residual_quantiles_real_dn": np.quantile(
            real_values, [0.001, 0.01, 0.5, 0.99, 0.999], axis=1).T.round(4).tolist(),
        "residual_quantiles_synthetic_dn": np.quantile(
            fake_values, [0.001, 0.01, 0.5, 0.99, 0.999], axis=1).T.round(4).tolist(),
        "temporal_pixel_std_real_dn": real_temporal.mean(axis=(1, 2)).round(4).tolist(),
        "temporal_pixel_std_synthetic_dn": fake_temporal.mean(axis=(1, 2)).round(4).tolist(),
        "temporal_pixel_rms_std_real_dn": np.sqrt(
            np.mean(real_temporal**2, axis=(1, 2))).round(4).tolist(),
        "temporal_pixel_rms_std_synthetic_dn": np.sqrt(
            np.mean(fake_temporal**2, axis=(1, 2))).round(4).tolist(),
        "unmodeled_fixed_pattern_std_dn": real_fixed.std(axis=(1, 2)).round(4).tolist(),
        "reference_vs_holdout_mean_residual_correlation": [
            round(float(np.corrcoef(model.reference_residual[c].ravel(),
                                    real_fixed[c].ravel())[0, 1]), 5)
            for c in range(4)
        ],
        "synthetic_sample_mean_pattern_std_dn": fake_fixed.std(axis=(1, 2)).round(4).tolist(),
        "real_frame_mean_sd_dn": real_frame_mean.std(axis=0).round(5).tolist(),
        "synthetic_frame_mean_sd_dn": fake_frame_mean.std(axis=0).round(5).tolist(),
        "radial_psd_synthetic_over_real":
            (psd_fake / np.maximum(psd_real, 1e-20)).round(4).tolist(),
    }


def validate(calibration: Path, gains: list[int], sample_count: int,
             output: Path) -> dict:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    fitted = json.loads(calibration.read_text())
    report = {
        "protocol": "frame 0 fit; all remaining frames 1..49 held out; fresh synthetic frames",
        "real_frames": 49,
        "synthetic_frames": sample_count,
        "real_psd_frames": sorted(REAL_PSD_FRAMES),
        "histogram_pixel_stride": 8,
        "sigma_and_iterations_unchanged": True,
        "gains": {},
    }
    for gain in gains:
        info = fitted["gains"][str(gain)]
        with np.load(info["model_file"]) as archive:
            reference = archive["reference_dark_dn"]
        source_reference = pack_bayer(read_frames(info["dark_path"], SPEC,
                                                   max_frames=1))[0].astype(np.uint16)
        digest = hashlib.sha256(reference.tobytes()).hexdigest()
        if digest != info["reference_frame_sha256"] or not np.array_equal(
            reference, source_reference
        ):
            raise ValueError(f"model reference is not raw frame 0 for gain {gain}")
        model = load_calibrated_model(info["model_file"])
        real, real_mean, real_temporal = collect_real(Path(info["dark_path"]), model)
        fake, fake_mean, fake_temporal = collect_synthetic(model, gain, sample_count)
        report["gains"][str(gain)] = {
            "real": aggregate(real),
            "synthetic": aggregate(fake),
            "comparison": compare(real, fake, real_mean, fake_mean,
                                  real_temporal, fake_temporal, model),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"validated {gain}: 49 real, {sample_count} synthetic", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", nargs="+", type=int,
                        default=[100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600])
    parser.add_argument("--synthetic-frames", type=int, default=8)
    args = parser.parse_args()
    validate(args.calibration, args.gains, args.synthetic_frames, args.output)


if __name__ == "__main__":
    main()

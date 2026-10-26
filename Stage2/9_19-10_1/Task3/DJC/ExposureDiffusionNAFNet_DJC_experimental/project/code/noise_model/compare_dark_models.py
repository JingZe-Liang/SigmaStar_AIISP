"""Fair holdout comparison: one-frame CVPR model vs multi-frame decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from .calibrate_two_shots import SPEC
from .pnnp_decomposition import DecomposedDarkModel
from .raw_io import pack_bayer, read_frames
from .two_shots import load_calibrated_model
from .validate_two_shots import radial_power


def diagnostics(stack: np.ndarray, fixed_map: np.ndarray,
                target_mean_map: np.ndarray | None = None) -> dict:
    """All groups use the same ten-frame statistics and 8x pixel grid."""
    n = len(stack)
    means = stack.mean(axis=(2, 3), dtype=np.float64)
    mean_map = stack.mean(axis=0, dtype=np.float64)
    centered = stack - mean_map[None]
    temporal_var = centered.astype(np.float64).__pow__(2).sum(axis=0) / (n - 1)
    temporal_rms_std = np.sqrt(temporal_var.mean(axis=(1, 2)))
    fixed_difference = (mean_map if target_mean_map is None else target_mean_map) - fixed_map
    fixed_rmse = np.sqrt(np.mean(fixed_difference**2, axis=(1, 2)))
    spatial_difference = fixed_difference - fixed_difference.mean(
        axis=(1, 2), keepdims=True
    )
    spatial_fixed_rmse = np.sqrt(np.mean(spatial_difference**2, axis=(1, 2)))
    psd = np.mean([radial_power(centered[i]) for i in range(n)], axis=0)
    row_std = np.mean([centered[i].mean(axis=2).std(axis=1) for i in range(n)], axis=0)
    col_std = np.mean([centered[i].mean(axis=1).std(axis=1) for i in range(n)], axis=0)
    return {
        "frame_mean_sd_dn": means.std(axis=0).round(5).tolist(),
        "pixel_temporal_rms_std_dn": temporal_rms_std.round(5).tolist(),
        "fixed_map_rmse_to_group_mean_dn": fixed_rmse.round(5).tolist(),
        "spatial_fixed_map_rmse_dn": spatial_fixed_rmse.round(5).tolist(),
        "row_std_dn": row_std.round(5).tolist(),
        "col_std_dn": col_std.round(5).tolist(),
        "psd": psd,
        "raw_sample": stack[:, :, ::8, ::8].transpose(1, 0, 2, 3).reshape(4, -1),
    }


def compare_to_real(real: dict, synthetic: dict) -> dict:
    real_psd = np.asarray(real["psd"])
    synthetic_psd = np.asarray(synthetic["psd"])
    ratio = synthetic_psd / np.maximum(real_psd, 1e-20)
    real_sigma = np.asarray(real["pixel_temporal_rms_std_dn"])
    synthetic_sigma = np.asarray(synthetic["pixel_temporal_rms_std_dn"])
    return {
        "pixel_temporal_std_error_pct":
            ((synthetic_sigma / real_sigma - 1) * 100).round(3).tolist(),
        "psd_ratio": ratio.round(4).tolist(),
        "mean_abs_psd_log_ratio_pct": float(np.abs(np.log(ratio)).mean() * 100),
        "pooled_raw_wasserstein_dn": [
            round(float(wasserstein_distance(real["raw_sample"][c],
                                             synthetic["raw_sample"][c])), 4)
            for c in range(4)
        ],
        "frame_mean_sd_dn": synthetic["frame_mean_sd_dn"],
        "row_std_dn": synthetic["row_std_dn"],
        "col_std_dn": synthetic["col_std_dn"],
    }


def evaluate(pnnp_report: Path, two_shots_report: Path, output: Path,
             gains: list[int]) -> dict:
    fit = json.loads(pnnp_report.read_text())
    baseline = json.loads(two_shots_report.read_text())
    if fit["fit_indices"] != list(range(40)) or fit["holdout_indices"] != list(range(40, 50)):
        raise ValueError("expected 40/10 split")
    report = {"fit_indices_new_model": list(range(40)),
              "fit_index_one_frame_baseline": 0,
              "shared_holdout_indices": list(range(40, 50)),
              "synthetic_frames_each": 10,
              "gains": {}}
    for gain in gains:
        info = fit["gains"][str(gain)]
        old_info = baseline["gains"][str(gain)]
        if info["dark_path"] != old_info["dark_path"]:
            raise ValueError(f"different RAW files for gain {gain}")
        real = pack_bayer(read_frames(info["dark_path"], SPEC,
                                      start_frame=40, max_frames=10), SPEC.cfa)
        new_model = DecomposedDarkModel.load(info["model_file"])
        old_model = load_calibrated_model(old_info["model_file"])
        new = np.stack([new_model.sample(seed=501000 + gain * 100 + i,
                                         residual_mode="spectral_marginal")
                        for i in range(10)])
        independent = np.stack([new_model.sample(
            seed=501000 + gain * 100 + i, residual_mode="iid")
            for i in range(10)])
        old = np.stack([new_model.quantize(old_model.sample(
            seed=301000 + gain * 100 + i))
            for i in range(10)]).astype(np.float32)
        real_stats = diagnostics(real, real.mean(axis=0))
        heldout_mean_map = real.mean(axis=0, dtype=np.float64)
        new_stats = diagnostics(new, new_model.fixed_map, heldout_mean_map)
        iid_stats = diagnostics(independent, new_model.fixed_map, heldout_mean_map)
        old_stats = diagnostics(old, old_model.shading + old_model.residual_mean,
                                heldout_mean_map)
        new_compare = compare_to_real(real_stats, new_stats)
        iid_compare = compare_to_real(real_stats, iid_stats)
        old_compare = compare_to_real(real_stats, old_stats)
        report["gains"][str(gain)] = {
            "real": {k: v for k, v in real_stats.items()
                     if k not in ("psd", "raw_sample")},
            "pnnp_spectral_marginal": {
                **{k: v for k, v in new_stats.items()
                   if k not in ("psd", "raw_sample")},
                "comparison": new_compare,
            },
            "pnnp_iid": {
                **{k: v for k, v in iid_stats.items()
                   if k not in ("psd", "raw_sample")},
                "comparison": iid_compare,
            },
            "one_frame_cvpr": {
                **{k: v for k, v in old_stats.items()
                   if k not in ("psd", "raw_sample")},
                "comparison": old_compare,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"compared gain {gain}: spectral+marginal W1 "
              f"{np.mean(new_compare['pooled_raw_wasserstein_dn']):.3f}, "
              f"iid W1 {np.mean(iid_compare['pooled_raw_wasserstein_dn']):.3f}, "
              f"one-frame W1 {np.mean(old_compare['pooled_raw_wasserstein_dn']):.3f}",
              flush=True)
        del real, new, independent, old, new_model, old_model
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnp-calibration", type=Path, required=True)
    parser.add_argument("--two-shots-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", nargs="+", type=int,
                        default=[100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600])
    args = parser.parse_args()
    evaluate(args.pnnp_calibration, args.two_shots_calibration,
             args.output, args.gains)


if __name__ == "__main__":
    main()

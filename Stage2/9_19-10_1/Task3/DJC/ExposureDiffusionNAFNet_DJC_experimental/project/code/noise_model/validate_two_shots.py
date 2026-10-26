"""Held-out statistical check of the CVPR 2026 dark synthesizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftfreq
from scipy.stats import wasserstein_distance

from .calibrate_two_shots import SPEC
from .raw_io import pack_bayer, read_frames
from .two_shots import SpectralDarkModel, load_calibrated_model


def radial_power(residual: np.ndarray) -> np.ndarray:
    _, height, width = residual.shape
    fy = fftfreq(height)[:, None]
    fx = fftfreq(width)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    bands = np.array([0.01, 0.04, 0.1, 0.2, 0.35, 0.5, 0.71])
    power = np.abs(fft2(residual, axes=(-2, -1)))**2
    return np.array([[float(power[c, (radius >= low) & (radius < high)].mean())
                      for low, high in zip(bands[:-1], bands[1:])]
                     for c in range(4)])


def summarize(real: np.ndarray, synthetic: np.ndarray,
              model: SpectralDarkModel) -> dict:
    real_res = real - model.shading
    fake_res = synthetic - model.shading
    real_res -= real_res.mean(axis=(-2, -1), keepdims=True)
    fake_res -= fake_res.mean(axis=(-2, -1), keepdims=True)
    real_power = radial_power(real_res)
    fake_power = radial_power(fake_res)
    # Deterministic sampling gives reproducible, bounded memory distance.
    real_points = real_res[:, ::3, ::3].reshape(4, -1)
    fake_points = fake_res[:, ::3, ::3].reshape(4, -1)
    return {
        "holdout_mean_dn": real.mean(axis=(1, 2)).round(4).tolist(),
        "synthetic_mean_dn": synthetic.mean(axis=(1, 2)).round(4).tolist(),
        "holdout_residual_std_dn": real_res.std(axis=(1, 2)).round(4).tolist(),
        "synthetic_residual_std_dn": fake_res.std(axis=(1, 2)).round(4).tolist(),
        "residual_wasserstein_dn": [
            round(float(wasserstein_distance(real_points[c], fake_points[c])), 4)
            for c in range(4)
        ],
        "radial_psd_ratio_synth_over_holdout":
            (fake_power / np.maximum(real_power, 1e-20)).round(3).tolist(),
        "real_channel_correlation": np.corrcoef(real_res.reshape(4, -1)).round(4).tolist(),
        "synthetic_channel_correlation": np.corrcoef(fake_res.reshape(4, -1)).round(4).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gains", nargs="+", type=int, default=[100, 1600, 25600])
    parser.add_argument("--holdout-frame", type=int, default=45)
    args = parser.parse_args()
    report = json.loads(args.calibration.read_text())
    validation = {"holdout_frame": args.holdout_frame, "gains": {}}
    for gain in args.gains:
        info = report["gains"][str(gain)]
        model = load_calibrated_model(info["model_file"])
        real = pack_bayer(read_frames(info["dark_path"], SPEC, max_frames=1,
                                      start_frame=args.holdout_frame))[0]
        synthetic = model.sample(seed=2026 + gain)
        validation["gains"][str(gain)] = summarize(real, synthetic, model)
        print(f"validated gain {gain}", flush=True)
    target = args.calibration.parent / "validation.json"
    target.write_text(json.dumps(validation, indent=2))
    print(target)


if __name__ == "__main__":
    main()

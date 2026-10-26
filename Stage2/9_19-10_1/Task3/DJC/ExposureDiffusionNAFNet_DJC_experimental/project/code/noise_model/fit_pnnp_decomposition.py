"""Fit the transparent PNNP-style physical decomposition on MIS20S1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrate_two_shots import GAINS, SPEC, exposure_us, one_raw
from .pnnp_decomposition import DecomposedDarkModel
from .raw_io import pack_bayer, read_frames


def fit_all(root: Path, output: Path, gains: list[int],
            fit_frames: int = 40) -> dict:
    if fit_frames < 3 or fit_frames >= 50:
        raise ValueError("fit_frames must lie in [3,49]")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "PNNP TPAMI 2026 physics-guided decomposition; empirical frame offsets and pixel bootstrap are project adaptations",
        "raw_spec": vars(SPEC),
        "fit_indices": list(range(fit_frames)),
        "holdout_indices": list(range(fit_frames, 50)),
        "gains": {},
    }
    for gain in gains:
        dark_path = one_raw(root, "black", gain)
        frames = pack_bayer(read_frames(dark_path, SPEC,
                                         max_frames=fit_frames), SPEC.cfa)
        model = DecomposedDarkModel.fit(frames, seed=2026 + gain)
        model_path = output / f"mis20s1_gain{gain}_pnnp_decomp.npz"
        model.save(model_path)
        report["gains"][str(gain)] = {
            "model_file": str(model_path.resolve()),
            "dark_path": str(dark_path.resolve()),
            "exposure_us": exposure_us(dark_path),
            "fixed_map_std_dn": model.fixed_map.std(axis=(1, 2)).round(5).tolist(),
            "high_frequency_fpn_shrink": model.fpn_shrink.round(5).tolist(),
            "adc_step_dn": model.adc_step_dn,
            "adc_offset_dn": model.adc_offset_dn,
            "adc_parity_counts_train": np.bincount(
                model.adc_offsets_per_frame, minlength=2
            ).tolist(),
            "frame_offset_std_dn": model.frame_offsets.std(axis=0).round(5).tolist(),
            "row_std_dn": np.sqrt(np.diag(model.row_cov)).round(5).tolist(),
            "col_std_dn": np.sqrt(np.diag(model.col_cov)).round(5).tolist(),
            "pixel_std_dn": model.pixel_pool.std(axis=1).round(5).tolist(),
        }
        (output / "calibration.json").write_text(json.dumps(report, indent=2))
        print(f"fitted gain {gain}: frame offset SD "
              f"{np.mean(model.frame_offsets.std(axis=0)):.3f} DN", flush=True)
        del frames, model
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", nargs="+", type=int, default=GAINS)
    args = parser.parse_args()
    fit_all(args.calibration_root, args.output, args.gains)


if __name__ == "__main__":
    main()

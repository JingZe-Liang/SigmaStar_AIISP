"""Compare the quantized likelihood on generated dark samples and real holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fit_quantized_likelihood import _score_dark_holdout
from .pnnp_decomposition import DecomposedDarkModel


def validate(calibration: Path, likelihood_dir: Path, output: Path,
             gains: list[int], samples: int = 3) -> dict:
    report = {"samples": samples, "gains": {}}
    calibration_data = json.loads(calibration.read_text())["gains"]
    for gain in gains:
        info = calibration_data[str(gain)]
        model_path = likelihood_dir / f"mis20s1_gain{gain}_quantized_likelihood.npz"
        with np.load(model_path) as archive:
            mixture = {
                "weights": archive["mixture_weights"],
                "means_dn": archive["mixture_means_dn"],
                "sigmas_dn": archive["mixture_sigmas_dn"],
            }
        dark_model = DecomposedDarkModel.load(
            likelihood_dir.parent / "pnnp_decomposition" /
            f"mis20s1_gain{gain}_pnnp_decomp.npz"
        )
        synthetic = np.stack([
            dark_model.sample(
                seed=800000 + gain * 10 + index,
                residual_mode="spectral_marginal",
            )
            for index in range(samples)
        ])
        synthetic_score = _score_dark_holdout(synthetic, dark_model, mixture)
        report["gains"][str(gain)] = {
            "real_holdout_nll_nats_per_pixel": info["holdout"]["nll_nats_per_pixel"],
            "synthetic_nll_nats_per_pixel": synthetic_score["nll_nats_per_pixel"],
            "synthetic_channel_nll_nats_per_pixel": synthetic_score[
                "channel_nll_nats_per_pixel"
            ],
        }
        print(
            f"gain {gain}: real={info['holdout']['nll_nats_per_pixel']:.4f} "
            f"synthetic={synthetic_score['nll_nats_per_pixel']:.4f}",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--likelihood-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", nargs="+", type=int, default=[100, 800, 1600, 6400, 12800, 25600])
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    validate(args.calibration, args.likelihood_dir, args.output, args.gains, args.samples)


if __name__ == "__main__":
    main()

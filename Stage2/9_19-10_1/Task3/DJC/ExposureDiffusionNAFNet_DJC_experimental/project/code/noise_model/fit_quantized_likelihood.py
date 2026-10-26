"""Fit and validate the first usable RAW observation likelihood.

The dark stack is split in time, not in pixels.  A 30/10 internal split
selects the residual-mixture complexity, then a 40/10 split is kept for the
reported holdout.  The resulting likelihood is a marginal per-pixel model:
the fitted fixed dark map is retained, while frame and row/column components
are integrated into the residual variance.  This is deliberately explicit;
it is suitable for the first R1 ablation, not a claim that shared band
variables have disappeared from the sensor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .calibrate_two_shots import GAINS, SPEC, one_raw
from .pnnp_decomposition import DecomposedDarkModel
from .quantized_likelihood import fit_gaussian_mixture_1d, quantized_poisson_gaussian_log_prob
from .raw_io import pack_bayer, read_frames


def _fit_channel_mixtures(model: DecomposedDarkModel, components: int) -> dict[str, np.ndarray]:
    """Fit pixel residual mixtures and marginalize fitted band/frame variance."""
    weights, means, sigmas = [], [], []
    band_var = np.diag(model.row_cov) + np.diag(model.col_cov)
    frame_var = model.frame_offsets.var(axis=0, ddof=1)
    for channel in range(4):
        w, m, s = fit_gaussian_mixture_1d(
            model.pixel_pool[channel], components=components, seed=2026 + channel
        )
        weights.append(w)
        means.append(m)
        sigmas.append(np.sqrt(s * s + band_var[channel] + frame_var[channel]))
    return {
        "weights": np.asarray(weights, dtype=np.float32),
        "means_dn": np.asarray(means, dtype=np.float32),
        "sigmas_dn": np.asarray(sigmas, dtype=np.float32),
    }


def _score_dark_holdout(
    frames: np.ndarray,
    model: DecomposedDarkModel,
    mixture: dict[str, np.ndarray],
    stride: int = 8,
) -> dict[str, object]:
    """Score quantized dark observations in DN, with finite differentiable p."""
    observations = frames[:, :, ::stride, ::stride].transpose(1, 0, 2, 3).reshape(4, -1)
    base_offset = model.fixed_map[:, ::stride, ::stride]
    offsets = np.broadcast_to(
        base_offset[:, None], (4, len(frames), *base_offset.shape[1:])
    ).reshape(4, -1)
    nll = []
    quantiles = []
    step = float(model.adc_step_dn)
    for channel in range(4):
        y = torch.from_numpy(observations[channel])
        offset = torch.from_numpy(offsets[channel])
        logp = quantized_poisson_gaussian_log_prob(
            y,
            torch.zeros_like(y),
            gain_dn_per_electron=1.0,
            mixture_weights=torch.from_numpy(mixture["weights"][channel]),
            mixture_means_dn=torch.from_numpy(mixture["means_dn"][channel]),
            mixture_sigmas_dn=torch.from_numpy(mixture["sigmas_dn"][channel]),
            dark_offset_dn=offset,
            quantization_step_dn=step,
        )
        if not torch.isfinite(logp).all():
            raise ValueError("non-finite quantized dark likelihood")
        residual = observations[channel] - offsets[channel]
        nll.append(float(-logp.mean()))
        quantiles.append(np.quantile(residual, [0.01, 0.5, 0.99]).round(5).tolist())
    return {
        "nll_nats_per_pixel": round(float(np.mean(nll)), 6),
        "channel_nll_nats_per_pixel": np.round(nll, 6).tolist(),
        "observed_residual_quantiles_dn": quantiles,
        "num_pixels": int(observations.shape[1]),
        "quantization_step_dn": step,
    }


def _fit_one_gain(dark_path: Path, output_dir: Path, gain: int,
                  gain_dn: float | None = None) -> dict:
    all_frames = pack_bayer(read_frames(dark_path, SPEC, max_frames=50), SPEC.cfa)
    internal_model = DecomposedDarkModel.fit(all_frames[:30], seed=3000 + gain)
    internal_scores = {}
    for components in (1, 3, 5):
        mixture = _fit_channel_mixtures(internal_model, components)
        internal_scores[str(components)] = _score_dark_holdout(
            all_frames[30:40], internal_model, mixture
        )
    selected = min(
        internal_scores,
        key=lambda key: internal_scores[key]["nll_nats_per_pixel"],
    )
    final_model = DecomposedDarkModel.fit(all_frames[:40], seed=4000 + gain)
    final_mixture = _fit_channel_mixtures(final_model, int(selected))
    holdout = _score_dark_holdout(all_frames[40:50], final_model, final_mixture)
    model_path = output_dir / f"mis20s1_gain{gain}_quantized_likelihood.npz"
    np.savez_compressed(
        model_path,
        fixed_map=final_model.fixed_map,
        mixture_weights=final_mixture["weights"],
        mixture_means_dn=final_mixture["means_dn"],
        mixture_sigmas_dn=final_mixture["sigmas_dn"],
        row_cov=final_model.row_cov,
        col_cov=final_model.col_cov,
        frame_offsets=final_model.frame_offsets,
        gain_dn=np.float32(gain_dn if gain_dn is not None else np.nan),
        adc_step_dn=np.int32(final_model.adc_step_dn),
        fit_frames=np.int32(final_model.fit_frames),
    )
    return {
        "model_file": str(model_path.resolve()),
        "dark_path": str(dark_path.resolve()),
        "fit_indices": list(range(40)),
        "internal_selection_indices": list(range(30, 40)),
        "holdout_indices": list(range(40, 50)),
        "selected_components": int(selected),
        "internal_scores": internal_scores,
        "holdout": holdout,
        "fixed_map_std_dn": np.round(final_model.fixed_map.std(axis=(1, 2)), 5).tolist(),
        "mixture_sigma_dn": np.round(final_mixture["sigmas_dn"], 5).tolist(),
    }


def fit_all(calibration_root: Path, output: Path, gains: tuple[int, ...] = GAINS,
            shot_calibration: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": (
            "quantized Poisson-Gaussian likelihood; PNNP-style dark decomposition; "
            "1/3/5-component residual mixture selected on frames 30-39"
        ),
        "raw_spec": vars(SPEC),
        "gains": {},
    }
    shot_report = {}
    if shot_calibration is not None:
        shot_report = json.loads(shot_calibration.read_text()).get("gains", {})
    for gain in gains:
        dark = one_raw(calibration_root, "black", gain)
        shot_gain = shot_report.get(str(gain), {}).get("shot", {}).get("shot_gain_dn")
        report["gains"][str(gain)] = _fit_one_gain(
            dark, output, gain, float(shot_gain) if shot_gain is not None else None
        )
        report["gains"][str(gain)]["shot_gain_dn"] = shot_gain
        (output / "calibration.json").write_text(json.dumps(report, indent=2))
        result = report["gains"][str(gain)]
        print(
            f"gain {gain}: components={result['selected_components']} "
            f"internal_nll={result['internal_scores'][str(result['selected_components'])]['nll_nats_per_pixel']:.4f} "
            f"holdout_nll={result['holdout']['nll_nats_per_pixel']:.4f}",
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot-calibration", type=Path, default=None)
    parser.add_argument("--gains", nargs="+", type=int, default=GAINS)
    args = parser.parse_args()
    fit_all(args.calibration_root, args.output, tuple(args.gains), args.shot_calibration)


if __name__ == "__main__":
    main()

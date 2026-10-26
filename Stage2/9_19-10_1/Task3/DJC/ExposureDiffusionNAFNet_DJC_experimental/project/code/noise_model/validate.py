from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

from .likelihood import empirical_residual_log_prob
from .raw_io import RawSpec, read_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    fit_meta = json.loads(Path(config["fit_metadata"]).read_text())
    fit_arrays = np.load(fit_meta["npz"])
    spec = RawSpec(**config["spec"])
    holdout = read_frames(
        config["holdout_path"],
        spec,
        max_frames=config.get("holdout_max_frames", config.get("max_frames", 1)),
        start_frame=config.get("holdout_start_frame", 0),
    )
    dark = fit_arrays["dark_field"]
    residual = holdout - dark[None]
    samples = torch.from_numpy(fit_arrays["pixel_residual_samples"])
    residual = torch.from_numpy((holdout - dark[None]).reshape(-1)[:64])
    x = torch.zeros_like(residual, requires_grad=True)
    logp = empirical_residual_log_prob(
        residual,
        x,
        samples,
        bandwidth_dn=float(config.get("empirical_bandwidth_dn", 0.75)),
    )
    grad = torch.autograd.grad(logp.mean(), x)[0]
    holdout_residual = (holdout - dark[None]).reshape(-1)
    reference_residual = fit_arrays["pixel_residual_samples"]
    ks = ks_2samp(holdout_residual, reference_residual)
    report = {
        "holdout_mean_dn": float(residual.mean()),
        "holdout_std_dn": float(residual.std()),
        "fit_sigma_dn": float(fit_meta["sigma_dn"]),
        "residual_ks_stat": float(ks.statistic),
        "residual_ks_pvalue": float(ks.pvalue),
        "residual_wasserstein_dn": float(wasserstein_distance(holdout_residual, reference_residual)),
        "empirical_likelihood_finite": bool(torch.isfinite(logp).all()),
        "gradient_finite": bool(torch.isfinite(grad).all()),
    }
    output = Path(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

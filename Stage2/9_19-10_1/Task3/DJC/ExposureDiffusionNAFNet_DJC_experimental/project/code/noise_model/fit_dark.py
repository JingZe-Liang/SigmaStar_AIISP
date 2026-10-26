from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .raw_io import RawSpec, read_frames


def fit_dark(frames: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit fixed field, zero-mean row/column terms and residual sigma."""
    fixed = frames.mean(axis=0).astype(np.float32)
    residual = frames - fixed[None]
    frame_bias = residual.mean(axis=(1, 2)).astype(np.float32)
    residual = residual - frame_bias[:, None, None]
    row_bias = residual.mean(axis=(0, 2)).astype(np.float32)
    row_bias -= row_bias.mean()
    residual = residual - row_bias[None, :, None]
    col_bias = residual.mean(axis=(0, 1)).astype(np.float32)
    col_bias -= col_bias.mean()
    residual = residual - col_bias[None, None, :]
    sample_count = min(65536, residual.size)
    rng = np.random.default_rng(7)
    indices = rng.choice(residual.size, size=sample_count, replace=False)
    return {
        "dark_field": fixed,
        "frame_bias": frame_bias,
        "row_bias": row_bias,
        "col_bias": col_bias,
        "pixel_residual_samples": residual.reshape(-1)[indices].astype(np.float32),
        "sigma_dn": float(residual.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    spec = RawSpec(**config["spec"])
    frames = read_frames(
        config["dark_path"],
        spec,
        max_frames=config.get("fit_max_frames", config.get("max_frames")),
        start_frame=config.get("fit_start_frame", 0),
    )
    result = fit_dark(frames)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    metadata = {"npz": str(output.with_suffix(".npz")), "sigma_dn": result["sigma_dn"], "spec": config["spec"]}
    output.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

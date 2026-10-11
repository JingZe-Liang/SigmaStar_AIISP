#!/usr/bin/env python3
"""Validate eight-frame dynamic Noise2Noise windows without training."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64, copy=False)
    second = second.astype(np.float64, copy=False)
    first -= np.mean(first)
    second -= np.mean(second)
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    return float(np.sum(first * second) / denominator) if denominator > 1e-8 else 0.0


def _highpass(values: np.ndarray) -> np.ndarray:
    return values - np.stack([cv2.blur(frame, (3, 3)) for frame in values])


def validate_probe(manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window_results: list[dict[str, object]] = []
    for record in manifest["windows"]:
        sample_path = Path(record["sample"])
        sample = np.load(sample_path)
        source = sample["source"].astype(np.float32, copy=False)
        target = sample["target"].astype(np.float32, copy=False)
        confidence = sample["confidence"].astype(bool, copy=False)
        if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 4:
            raise ValueError(f"invalid source/target shape in {sample_path}: {source.shape}")
        if confidence.shape != (source.shape[0], *source.shape[2:]):
            raise ValueError(f"invalid confidence shape in {sample_path}: {confidence.shape}")
        correlations: list[float] = []
        difference_means: list[float] = []
        for channel in range(4):
            source_highpass = _highpass(source[:, channel])[confidence]
            target_highpass = _highpass(target[:, channel])[confidence]
            correlations.append(_correlation(source_highpass, target_highpass))
            difference_means.append(float(np.mean(np.abs(source[:, channel][confidence] - target[:, channel][confidence]))))
        window_results.append({
            "start_frame": record["start_frame"],
            "confidence_pixels": int(np.count_nonzero(confidence)),
            "confidence_fraction": float(confidence.mean()),
            "highpass_correlation_per_channel": dict(zip(_CHANNEL_NAMES, correlations, strict=True)),
            "highpass_correlation_mean": float(np.mean(correlations)),
            "source_target_abs_difference_mean_per_channel": dict(zip(_CHANNEL_NAMES, difference_means, strict=True)),
        })

    result: dict[str, object] = {
        "protocol": "eight-frame dynamic Noise2Noise window validation",
        "probe_manifest": str(manifest_path.resolve()),
        "target_semantics": manifest["target_semantics"],
        "confidence_definition": manifest.get("confidence_definition", {}),
        "windows": window_results,
        "interpretation": {
            "highpass_correlation_is_a_diagnostic": True,
            "low_correlation_supports_but_does_not_prove_noise_independence": True,
            "no_clean_reference_metric_is_available_for_motion_pixels": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_probe(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()

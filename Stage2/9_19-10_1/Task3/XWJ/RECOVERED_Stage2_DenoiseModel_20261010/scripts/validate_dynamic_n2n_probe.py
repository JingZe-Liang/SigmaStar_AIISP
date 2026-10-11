#!/usr/bin/env python3
"""Validate saved dynamic Noise2Noise probe samples without training."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")


def _highpass(values: np.ndarray) -> np.ndarray:
    return values - np.stack([cv2.blur(frame, (3, 3)) for frame in values])


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.sqrt(np.sum(first_centered * first_centered) * np.sum(second_centered * second_centered)))
    return float(np.sum(first_centered * second_centered) / denominator) if denominator > 1e-8 else 0.0


def validate_probe(manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_results: list[dict[str, object]] = []
    for record in manifest["pairs"]:
        sample_path = Path(record["sample"])
        sample = np.load(sample_path)
        source = sample["source"].astype(np.float32, copy=False)
        target = sample["target"].astype(np.float32, copy=False)
        confidence = sample["confidence"].astype(bool, copy=False)
        if source.shape != target.shape or source.shape[0] != 4 or confidence.shape != source.shape[1:]:
            raise ValueError(f"invalid sample shapes in {sample_path}")
        if not np.any(confidence):
            raise ValueError(f"empty confidence mask in {sample_path}")

        correlations: list[float] = []
        difference_means: list[float] = []
        difference_p95s: list[float] = []
        for channel in range(4):
            source_highpass = _highpass(source[channel:channel + 1])[0][confidence]
            target_highpass = _highpass(target[channel:channel + 1])[0][confidence]
            correlations.append(_correlation(source_highpass, target_highpass))
            difference = np.abs(source[channel][confidence] - target[channel][confidence])
            difference_means.append(float(np.mean(difference)))
            difference_p95s.append(float(np.percentile(difference, 95)))

        pair_results.append({
            "pair": record["pair"],
            "confidence_pixels": int(np.count_nonzero(confidence)),
            "highpass_correlation_per_channel": dict(zip(_CHANNEL_NAMES, correlations, strict=True)),
            "highpass_correlation_mean": float(np.mean(correlations)),
            "source_target_abs_difference_mean_per_channel": dict(zip(_CHANNEL_NAMES, difference_means, strict=True)),
            "source_target_abs_difference_p95_per_channel": dict(zip(_CHANNEL_NAMES, difference_p95s, strict=True)),
        })

    result: dict[str, object] = {
        "protocol": "dynamic Noise2Noise probe validation",
        "probe_manifest": str(manifest_path.resolve()),
        "target_semantics": manifest["target_semantics"],
        "pairs": pair_results,
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
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/dynamic_n2n_probe_645x/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/dynamic_n2n_probe_645x/validation.json"))
    args = parser.parse_args()
    print(json.dumps(validate_probe(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()

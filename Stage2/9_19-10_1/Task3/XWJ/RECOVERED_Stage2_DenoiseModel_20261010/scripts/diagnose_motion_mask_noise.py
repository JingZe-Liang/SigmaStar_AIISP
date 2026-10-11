#!/usr/bin/env python3
"""Audit whether dynamic-mask pixels have measurable motion or only noise.

This is a diagnostic only.  It does not modify supervision masks or training
data.  For selected frame pairs it measures how much of the current dynamic
confidence mask has near-zero optical flow, which is a useful warning signal
for static-background false positives.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from build_dynamic_n2n_probe import (  # noqa: E402
    _aligned_motion_mask,
    _flow_confidence,
    _read_aligned_frame,
)
from raw_fusion.finetune.a3 import _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.adapters import resolve_adapter  # noqa: E402
from raw_fusion.finetune.stream_io import open_stream  # noqa: E402


def _parse_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        start_text, end_text = item.strip().split(":")
        start, end = int(start_text), int(end_text)
        if start < 0 or end <= start:
            raise ValueError(f"invalid pair {item!r}")
        pairs.append((start, end))
    if not pairs:
        raise ValueError("at least one pair is required")
    return pairs


def _pair_report(
    source: np.ndarray,
    target: np.ndarray,
    source_motion: np.ndarray,
    target_motion: np.ndarray,
) -> dict[str, float]:
    _, flow, flow_error, photometric_error, flow_magnitude = _flow_confidence(source, target)
    height, width = flow_magnitude.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    target_motion_warped = cv2.remap(
        target_motion,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1.0,
    )
    dynamic = (source_motion >= 0.5) | (target_motion_warped >= 0.5)
    valid = (flow_error < 1.5) & (photometric_error < 0.08) & (flow_magnitude < 30.0)
    confidence = dynamic & valid

    def fraction(condition: np.ndarray, population: np.ndarray) -> float:
        return float(condition[population].mean()) if population.any() else 0.0

    report = {
        "dynamic_fraction": float(dynamic.mean()),
        "confidence_fraction": float(confidence.mean()),
        "confidence_fraction_of_dynamic": fraction(confidence, dynamic),
        "dynamic_flow_lt_0_5_fraction": fraction(flow_magnitude < 0.5, dynamic),
        "dynamic_flow_lt_1_fraction": fraction(flow_magnitude < 1.0, dynamic),
        "confidence_flow_lt_0_5_fraction": fraction(flow_magnitude < 0.5, confidence),
        "confidence_flow_lt_1_fraction": fraction(flow_magnitude < 1.0, confidence),
        "confidence_flow_median": float(np.median(flow_magnitude[confidence])) if confidence.any() else 0.0,
        "confidence_flow_p95": float(np.percentile(flow_magnitude[confidence], 95)) if confidence.any() else 0.0,
        "confidence_photometric_median": float(np.median(photometric_error[confidence])) if confidence.any() else 0.0,
        "confidence_area_with_flow_ge_1": float((confidence & (flow_magnitude >= 1.0)).mean()),
        "confidence_area_with_flow_ge_2": float((confidence & (flow_magnitude >= 2.0)).mean()),
    }
    return report


def run(
    *,
    supervision_dir: Path,
    gain: str,
    pairs: list[tuple[int, int]],
    output_path: Path,
) -> dict[str, object]:
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    reference_black = _channel_vector(report["black_levels"]).astype(np.float32)
    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames = sorted({frame for pair in pairs for frame in pair})
    loaded = {
        frame: _read_aligned_frame(reader, adapter, manifest, reference_black, frame)
        for frame in frames
    }
    masks = {frame: _aligned_motion_mask(manifest, frame) for frame in frames}
    pair_reports = []
    for start, end in pairs:
        pair_reports.append({
            "pair": [start, end],
            "stats": _pair_report(loaded[start], loaded[end], masks[start], masks[end]),
        })
    result: dict[str, object] = {
        "protocol": "motion-mask noise/low-flow audit",
        "gain": gain,
        "pairs": pair_reports,
        "interpretation": {
            "warning": "confidence pixels with flow below 1 px are not evidence of object motion; they may be static-background false positives from sensor noise, texture, or registration residuals",
            "recommended_next_gate": "rebuild the mask or confidence rule with a minimum-flow/temporal-consistency check before using dynamic targets for training",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--gain", default="645x")
    parser.add_argument("--pairs", default="30:34,46:50,140:144")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/finetune_mis20s1/evaluation/motion_mask_noise_diagnostic_645x.json"),
    )
    args = parser.parse_args()
    result = run(
        supervision_dir=args.supervision_dir,
        gain=args.gain,
        pairs=_parse_pairs(args.pairs),
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

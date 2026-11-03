#!/usr/bin/env python3
"""Render globally brightened canonical packed fused RAW variants with fixed ISP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from batch_fourchannel_fused_rgb import packed_permutation
from opencv_source_match.fair_compare import canonical_packed_to_mosaic, render_fixed
from opencv_source_match.pipeline import cfa_name, crop_default, load_metadata_file, save_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-packed", required=True, type=Path)
    parser.add_argument("--inference-report", required=True, type=Path)
    parser.add_argument("--metadata-under", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--factor", action="append", type=float, required=True, help="Linear signal factor; repeat")
    return parser.parse_args()


def label_for_factor(factor: float) -> str:
    increase_percent = round((factor - 1.0) * 100.0)
    if increase_percent <= 0:
        raise ValueError("Factor must be greater than 1.0")
    return "bright_{:02d}".format(increase_percent)


def load_cfa_aligned_packed(fused_path: Path, report_path: Path, metadata_cfa: str) -> tuple[np.ndarray, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_cfa = str(report["bayer_pattern"]).upper()
    packed = np.load(fused_path, allow_pickle=False)
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError("Expected canonical packed RAW [4,H,W], got {}".format(packed.shape))
    permutation = packed_permutation(source_cfa, metadata_cfa)
    return np.ascontiguousarray(packed[list(permutation)], dtype=np.float32), {
        "input": str(fused_path.resolve()),
        "inference_report": str(report_path.resolve()),
        "source_packed_cfa": source_cfa,
        "metadata_cfa": metadata_cfa,
        "permutation_to_metadata_canonical": list(permutation),
        "was_reordered": permutation != (0, 1, 2, 3),
    }


def main() -> None:
    args = parse_args()
    if len(set(args.factor)) != len(args.factor):
        raise ValueError("Each --factor must be unique")
    metadata = load_metadata_file(args.metadata_under)
    packed, layout = load_cfa_aligned_packed(args.fused_packed, args.inference_report, cfa_name(metadata["cfa"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for factor in args.factor:
        label = label_for_factor(factor)
        bright = packed * float(factor)
        packed_path = args.output_dir / ("fused_{}_packed_float32.npy".format(label))
        np.save(packed_path, bright)
        mosaic, clipped_fraction = canonical_packed_to_mosaic(bright, metadata["cfa"])
        mosaic = crop_default(mosaic, metadata)
        rgb, details = render_fixed(
            mosaic,
            metadata,
            fixed_ev=0.0,
            enable_false_color_suppression=False,
            enable_highlight_recovery=True,
            tone_map="reinhard",
        )
        rgb_path = args.output_dir / ("fused_{}.png".format(label))
        save_rgb(rgb, rgb_path)
        report = {
            "tool": "opencv-fixed-raw-compare-isp",
            "source_packed": layout,
            "brightness": {
                "linear_signal_factor": float(factor),
                "increase_percent": float((factor - 1.0) * 100.0),
                "operation": "fused_packed_native_signal * factor; no pre-ISP clipping",
                "input_range": [float(packed.min()), float(packed.max())],
                "brightened_range": [float(bright.min()), float(bright.max())],
                "out_of_sensor_range_fraction_before_isp": float(((bright < 0.0) | (bright > 1.0)).mean()),
                "clipped_before_demosaic_by_isp_fraction": float(clipped_fraction),
            },
            "isp": {
                "fixed_ev": 0.0,
                "highlight_recovery": True,
                "tone_map": "reinhard",
                "false_color_suppression": False,
                "camera_tint": False,
                "shared_metadata": str(args.metadata_under.resolve()),
            },
            "packed_output": str(packed_path.resolve()),
            "rgb_output": str(rgb_path.resolve()),
            "render_details": details,
        }
        report_path = args.output_dir / ("fused_{}_isp_report.json".format(label))
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(report)
    (args.output_dir / "fused_brightness_variants_manifest.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Rendered {} variants".format(len(results)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render geometric-mean-reference under/over RAW sources into existing test folders."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import tifffile

from opencv_source_match.fair_compare import render_fixed, validate_shared_metadata
from opencv_source_match.pipeline import crop_default, load_metadata_file, normalize_raw, save_rgb


DEFAULT_FUSED_RAW_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_emvd_bf16_geomean\u2fusion_raw_emvd_bf16_geomean\outputs\raw_mef_fused_geomean_test"
)
DEFAULT_RAW_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_data_mef")
DEFAULT_METADATA_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef")
DEFAULT_OUTPUT_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_emvd_bf16_geomean\u2fusion_raw_emvd_bf16_geomean\outputs\raw_mef_fixed_isp_highlight_recovery_reinhard_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-raw-root", type=Path, default=DEFAULT_FUSED_RAW_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append", default=[], help="Optional scene_xxxx; repeat for several scenes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError("Expected one {} in {}, found {}".format(pattern, directory, len(matches)))
    return matches[0]


def read_reference(scene: str, fused_raw_root: Path) -> dict:
    report_path = fused_raw_root / scene / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("Missing fused RAW inference report: {}".format(report_path))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reference = report["exposure_reference"]
        ratio = float(reference["over_to_under_ratio"])
        under_scale = float(reference["under_to_reference"])
        over_scale = float(reference["over_to_reference"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Invalid exposure reference report {}: {}".format(report_path, error)) from error
    if report.get("reference_domain") != "per-scene geometric-mean exposure":
        raise ValueError("Fused report is not in the expected geometric-mean domain: {}".format(report_path))
    if not math.isfinite(ratio) or ratio <= 0.0 or not math.isfinite(under_scale) or not math.isfinite(over_scale):
        raise ValueError("Non-finite exposure reference in {}".format(report_path))
    if not math.isclose(under_scale, math.sqrt(ratio), rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("under_to_reference does not equal sqrt(over_to_under_ratio) in {}".format(report_path))
    if not math.isclose(over_scale, 1.0 / under_scale, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("over_to_reference does not equal 1/sqrt(ratio) in {}".format(report_path))
    return {
        "report_path": report_path,
        "ratio": ratio,
        "under_scale": under_scale,
        "over_scale": over_scale,
        "estimation_mode": reference.get("estimation_mode"),
        "valid_pixel_count": reference.get("valid_pixel_count"),
    }


def input_paths(scene: str, args: argparse.Namespace) -> dict[str, Path]:
    return {
        "under_raw": exactly_one(args.raw_root / scene / "under", scene + "_unprocessed_*_16bit.tiff"),
        "over_raw": exactly_one(args.raw_root / scene / "over", scene + "_unprocessed_*_16bit.tiff"),
        "under_metadata": exactly_one(args.metadata_root / scene / "under", scene + "_metadata_numeric.json"),
        "over_metadata": exactly_one(args.metadata_root / scene / "over", scene + "_metadata_numeric.json"),
    }


def complete(scene_dir: Path) -> bool:
    return all(
        (scene_dir / filename).is_file()
        for filename in (
            "under_geomean_reference.png",
            "over_geomean_reference.png",
            "geomean_source_isp_report.json",
        )
    )


def render_scene(scene: str, args: argparse.Namespace) -> dict:
    paths = input_paths(scene, args)
    reference = read_reference(scene, args.fused_raw_root)
    under_metadata = load_metadata_file(paths["under_metadata"])
    over_metadata = load_metadata_file(paths["over_metadata"])
    validate_shared_metadata(under_metadata, over_metadata)
    under_native = crop_default(normalize_raw(tifffile.imread(paths["under_raw"]), under_metadata), under_metadata)
    over_native = crop_default(normalize_raw(tifffile.imread(paths["over_raw"]), under_metadata), under_metadata)
    if under_native.shape != over_native.shape:
        raise ValueError("Under/over cropped mosaics differ: {} vs {}".format(under_native.shape, over_native.shape))

    # The radiometric scaling follows the exact ratio used to make the fused RAW.
    under_reference = under_native * reference["under_scale"]
    over_reference = over_native * reference["over_scale"]
    output_dir = args.output_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    under_rgb, under_details = render_fixed(
        under_reference,
        under_metadata,
        fixed_ev=0.0,
        enable_false_color_suppression=False,
        enable_highlight_recovery=True,
        tone_map="reinhard",
    )
    over_rgb, over_details = render_fixed(
        over_reference,
        under_metadata,
        fixed_ev=0.0,
        enable_false_color_suppression=False,
        enable_highlight_recovery=True,
        tone_map="reinhard",
    )
    under_output = output_dir / "under_geomean_reference.png"
    over_output = output_dir / "over_geomean_reference.png"
    save_rgb(under_rgb, under_output)
    save_rgb(over_rgb, over_output)
    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "under_over_geometric_mean_reference_no_preview",
        "scene": scene,
        "fused_reference_report": str(reference["report_path"].resolve()),
        "reference_domain": "per-scene geometric-mean exposure",
        "over_to_under_ratio": reference["ratio"],
        "under_to_reference": reference["under_scale"],
        "over_to_reference": reference["over_scale"],
        "exposure_estimation_mode": reference["estimation_mode"],
        "exposure_valid_pixel_count": reference["valid_pixel_count"],
        "settings": {
            "fixed_ev": 0.0,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "false_color_suppression": False,
            "camera_tint": False,
            "uses_exposure_time_or_iso": False,
            "comparison_preview_generated": False,
        },
        "shared_metadata_reference": str(paths["under_metadata"].resolve()),
        "shared_metadata_validated_against": str(paths["over_metadata"].resolve()),
        "frames": {
            "under_geomean_reference": {
                "input_raw": str(paths["under_raw"].resolve()),
                "native_normalized_mean": float(under_native.mean()),
                "reference_normalized_mean": float(under_reference.mean()),
                "output": str(under_output.resolve()),
                "details": under_details,
            },
            "over_geomean_reference": {
                "input_raw": str(paths["over_raw"].resolve()),
                "native_normalized_mean": float(over_native.mean()),
                "reference_normalized_mean": float(over_reference.mean()),
                "output": str(over_output.resolve()),
                "details": over_details,
            },
        },
    }
    (output_dir / "geomean_source_isp_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    scenes = sorted(directory.name for directory in args.output_root.glob("scene_*") if directory.is_dir())
    if args.scene:
        wanted = set(args.scene)
        scenes = [scene for scene in scenes if scene in wanted]
        missing = wanted - set(scenes)
        if missing:
            raise FileNotFoundError("Requested scenes missing from output-root: {}".format(sorted(missing)))
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes:
        raise FileNotFoundError("No output scene directories under {}".format(args.output_root))
    batch_report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "under_over_geometric_mean_reference_no_preview",
        "output_root": str(args.output_root.resolve()),
        "total_scenes": len(scenes),
        "started_unix_s": time.time(),
        "completed": [],
        "skipped": [],
        "failed": {},
    }
    for index, scene in enumerate(scenes, start=1):
        output_dir = args.output_root / scene
        if not args.force and complete(output_dir):
            print("[{}/{}] {}: skipped".format(index, len(scenes), scene), flush=True)
            batch_report["skipped"].append(scene)
            continue
        try:
            print("[{}/{}] {}: rendering".format(index, len(scenes), scene), flush=True)
            render_scene(scene, args)
            batch_report["completed"].append(scene)
        except Exception as error:
            message = "{}: {}".format(type(error).__name__, error)
            batch_report["failed"][scene] = message
            print("[{}/{}] {}: FAILED: {}".format(index, len(scenes), scene, message), flush=True)
    batch_report["finished_unix_s"] = time.time()
    report_path = args.output_root / "batch_geomean_source_isp_report.json"
    report_path.write_text(json.dumps(batch_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Finished: completed={}, skipped={}, failed={}. Report: {}".format(
        len(batch_report["completed"]), len(batch_report["skipped"]), len(batch_report["failed"]), report_path
    ))
    if batch_report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

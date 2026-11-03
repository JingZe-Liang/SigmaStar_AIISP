#!/usr/bin/env python3
"""Render under RAW after mapping it to the corresponding over-exposure domain."""

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
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError("Expected one {} in {}, found {}".format(pattern, directory, len(matches)))
    return matches[0]


def read_ratio(scene: str, fused_raw_root: Path) -> tuple[float, Path, dict]:
    report_path = fused_raw_root / scene / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reference = report["exposure_reference"]
        ratio = float(reference["over_to_under_ratio"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Invalid fused RAW exposure report {}: {}".format(report_path, error)) from error
    if report.get("reference_domain") != "per-scene geometric-mean exposure":
        raise ValueError("Unexpected fused RAW reference domain in {}".format(report_path))
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("Non-positive or non-finite over_to_under_ratio in {}".format(report_path))
    return ratio, report_path, reference


def complete(scene_dir: Path) -> bool:
    return (scene_dir / "under_over_reference.png").is_file() and (
        scene_dir / "under_over_reference_isp_report.json"
    ).is_file()


def render_scene(scene: str, args: argparse.Namespace) -> dict:
    under_raw = exactly_one(args.raw_root / scene / "under", scene + "_unprocessed_*_16bit.tiff")
    under_metadata_path = exactly_one(args.metadata_root / scene / "under", scene + "_metadata_numeric.json")
    over_metadata_path = exactly_one(args.metadata_root / scene / "over", scene + "_metadata_numeric.json")
    under_metadata = load_metadata_file(under_metadata_path)
    over_metadata = load_metadata_file(over_metadata_path)
    validate_shared_metadata(under_metadata, over_metadata)
    ratio, fused_report_path, reference = read_ratio(scene, args.fused_raw_root)
    under_native = crop_default(normalize_raw(tifffile.imread(under_raw), under_metadata), under_metadata)

    # The output coordinate is explicitly the original over-exposure domain.
    under_over_reference = under_native * ratio
    rgb, details = render_fixed(
        under_over_reference,
        under_metadata,
        fixed_ev=0.0,
        enable_false_color_suppression=False,
        enable_highlight_recovery=True,
        tone_map="reinhard",
    )
    output_dir = args.output_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "under_over_reference.png"
    save_rgb(rgb, output_path)
    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "under_mapped_to_original_over_exposure_domain_no_preview",
        "scene": scene,
        "fused_reference_report": str(fused_report_path.resolve()),
        "output_reference_domain": "original over exposure",
        "over_to_under_ratio": ratio,
        "under_to_output_scale": ratio,
        "reference_exposure_estimation_mode": reference.get("estimation_mode"),
        "reference_valid_pixel_count": reference.get("valid_pixel_count"),
        "settings": {
            "fixed_ev": 0.0,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "false_color_suppression": False,
            "camera_tint": False,
            "uses_exposure_time_or_iso": False,
            "comparison_preview_generated": False,
        },
        "shared_metadata_reference": str(under_metadata_path.resolve()),
        "shared_metadata_validated_against": str(over_metadata_path.resolve()),
        "frame": {
            "input_raw": str(under_raw.resolve()),
            "native_normalized_mean": float(under_native.mean()),
            "over_domain_normalized_mean_before_isp_clipping": float(under_over_reference.mean()),
            "output": str(output_path.resolve()),
            "details": details,
        },
    }
    (output_dir / "under_over_reference_isp_report.json").write_text(
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
        raise FileNotFoundError("No scene directories under {}".format(args.output_root))
    batch_report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "under_mapped_to_original_over_exposure_domain_no_preview",
        "total_scenes": len(scenes),
        "started_unix_s": time.time(),
        "completed": [], "skipped": [], "failed": {},
    }
    for index, scene in enumerate(scenes, start=1):
        if not args.force and complete(args.output_root / scene):
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
    report_path = args.output_root / "batch_under_over_reference_isp_report.json"
    report_path.write_text(json.dumps(batch_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Finished: completed={}, skipped={}, failed={}. Report: {}".format(
        len(batch_report["completed"]), len(batch_report["skipped"]), len(batch_report["failed"]), report_path
    ))
    if batch_report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

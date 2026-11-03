#!/usr/bin/env python3
"""Batch fixed-ISP rendering for all U2Fusion RAW-MEF test scenes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opencv_source_match.fair_compare import render_comparison


DEFAULT_FUSED_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_emvd_bf16\outputs\raw_mef_fused_bf16_test"
)
DEFAULT_RAW_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_data_mef")
DEFAULT_METADATA_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef")
DEFAULT_OUTPUT_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_emvd_bf16\outputs\raw_mef_fixed_isp_strict_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--enable-highlight-recovery",
        action="store_true",
        help="Apply the same fixed clipped-channel reconstruction to under, fused, and over.",
    )
    parser.add_argument(
        "--tone-map",
        choices=("hard-clip", "reinhard"),
        default="hard-clip",
        help="One fixed output tone map for all three frames; default: hard-clip.",
    )
    return parser.parse_args()


def exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one {} in {}, found {}".format(pattern, directory, len(matches))
        )
    return matches[0]


def complete(output_dir: Path) -> bool:
    return all(
        (output_dir / filename).is_file()
        for filename in ("under.png", "fused.png", "over.png", "comparison_preview.png", "fairness_report.json")
    )


def input_paths(scene: str, args: argparse.Namespace) -> dict[str, Path]:
    return {
        "under_raw": exactly_one(args.raw_root / scene / "under", scene + "_unprocessed_*_16bit.tiff"),
        "over_raw": exactly_one(args.raw_root / scene / "over", scene + "_unprocessed_*_16bit.tiff"),
        "fused_packed": args.fused_root / scene / "fused_packed_float32.npy",
        "under_metadata": exactly_one(
            args.metadata_root / scene / "under", scene + "_metadata_numeric.json"
        ),
        "over_metadata": exactly_one(
            args.metadata_root / scene / "over", scene + "_metadata_numeric.json"
        ),
    }


def main() -> None:
    args = parse_args()
    scenes = sorted(
        directory.name
        for directory in args.fused_root.glob("scene_*")
        if (directory / "fused_packed_float32.npy").is_file()
    )
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        raise FileNotFoundError("No fused scenes under {}".format(args.fused_root))

    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "strict_raw_clipping_preserved",
        "settings": {
            "fixed_ev": 0.0,
            "tone_map": args.tone_map,
            "highlight_recovery": args.enable_highlight_recovery,
            "false_color_suppression": False,
            "camera_tint": False,
            "exposure_and_iso_used": False,
            "full_resolution_rgb": True,
            "preview_long_edge": 1200,
        },
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
            report["skipped"].append(scene)
            continue
        try:
            paths = input_paths(scene, args)
            if not paths["fused_packed"].is_file():
                raise FileNotFoundError(paths["fused_packed"])
            print("[{}/{}] {}: {}".format(
                index, len(scenes), scene, "dry run" if args.dry_run else "rendering"
            ), flush=True)
            if not args.dry_run:
                render_comparison(
                    **paths,
                    output_dir=output_dir,
                    fixed_ev=0.0,
                    enable_false_color_suppression=False,
                    enable_highlight_recovery=args.enable_highlight_recovery,
                    tone_map=args.tone_map,
                    max_long_edge=0,
                    preview_long_edge=1200,
                    image_extension=".png",
                )
            report["completed"].append(scene)
        except Exception as error:
            report["failed"][scene] = "{}: {}".format(type(error).__name__, error)
            print("[{}/{}] {}: FAILED: {}".format(
                index, len(scenes), scene, report["failed"][scene]
            ), flush=True)

    report["finished_unix_s"] = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "batch_fairness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "Finished: completed={}, skipped={}, failed={}. Report: {}".format(
            len(report["completed"]), len(report["skipped"]), len(report["failed"]), report_path
        ),
        flush=True,
    )
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

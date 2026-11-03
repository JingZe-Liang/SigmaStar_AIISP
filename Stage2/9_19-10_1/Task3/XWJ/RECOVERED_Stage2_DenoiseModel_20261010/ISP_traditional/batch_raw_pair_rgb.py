#!/usr/bin/env python3
"""Mirror under/over RAW scenes to RGB with one fixed fair OpenCV ISP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from opencv_source_match.fair_compare import (
    _load_raw_tiff,
    render_fixed,
    validate_shared_metadata,
)
from opencv_source_match.pipeline import load_metadata_file, resize_long_edge, save_rgb


DEFAULT_RAW_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_data_mef")
DEFAULT_METADATA_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef")
DEFAULT_OUTPUT_ROOT = Path(r"F:\RAW_MEF\data_RGB\iPhone17Pro\core_data_mef_highlight_reinhard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one {} in {}, found {}".format(pattern, directory, len(matches))
        )
    return matches[0]


def input_paths(scene: str, args: argparse.Namespace) -> dict[str, Path]:
    return {
        "under_raw": exactly_one(args.raw_root / scene / "under", scene + "_unprocessed_*_16bit.tiff"),
        "over_raw": exactly_one(args.raw_root / scene / "over", scene + "_unprocessed_*_16bit.tiff"),
        "under_metadata": exactly_one(
            args.metadata_root / scene / "under", scene + "_metadata_numeric.json"
        ),
        "over_metadata": exactly_one(
            args.metadata_root / scene / "over", scene + "_metadata_numeric.json"
        ),
    }


def output_paths(scene: str, inputs: dict[str, Path], output_root: Path) -> dict[str, Path]:
    scene_root = output_root / scene
    return {
        "under": scene_root / "under" / (inputs["under_raw"].stem + ".png"),
        "over": scene_root / "over" / (inputs["over_raw"].stem + ".png"),
        "report": scene_root / "fair_isp_report.json",
    }


def scene_complete(scene: str, args: argparse.Namespace) -> bool:
    try:
        inputs = input_paths(scene, args)
    except FileNotFoundError:
        return False
    paths = output_paths(scene, inputs, args.output_root)
    return all(path.is_file() for path in paths.values())


def render_scene(scene: str, args: argparse.Namespace) -> dict:
    inputs = input_paths(scene, args)
    under_meta = load_metadata_file(inputs["under_metadata"])
    over_meta = load_metadata_file(inputs["over_metadata"])
    validate_shared_metadata(under_meta, over_meta)
    under = _load_raw_tiff(inputs["under_raw"], under_meta)
    over = _load_raw_tiff(inputs["over_raw"], under_meta)
    if under.shape != over.shape:
        raise ValueError("Cropped under/over RAW dimensions differ: {} vs {}".format(under.shape, over.shape))

    paths = output_paths(scene, inputs, args.output_root)
    frames = {}
    for name, raw01, raw_path in (("under", under, inputs["under_raw"]), ("over", over, inputs["over_raw"])):
        image, details = render_fixed(
            raw01,
            under_meta,
            fixed_ev=0.0,
            enable_false_color_suppression=False,
            enable_highlight_recovery=True,
            tone_map="reinhard",
        )
        image = resize_long_edge(image, 0)
        save_rgb(image, paths[name])
        frames[name] = {
            "input": str(raw_path.resolve()),
            "output": str(paths[name].resolve()),
            "details": details,
        }

    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "fairness_contract": {
            "shared_metadata_reference": str(inputs["under_metadata"].resolve()),
            "shared_metadata_validated_against": str(inputs["over_metadata"].resolve()),
            "fixed_ev_for_all_frames": 0.0,
            "uses_baseline_exposure": False,
            "uses_exposure_time_or_iso": False,
            "uses_role_dependent_ev": False,
            "uses_highlight_reconstruction": True,
            "uses_camera_specific_tint": False,
            "uses_image_adaptive_tone_mapping": False,
            "tone_map": "reinhard",
            "false_color_suppression_enabled": False,
        },
        "shared_metadata": {
            "black_level": under_meta["black"],
            "white_level": under_meta["white"],
            "cfa": under_meta["cfa"],
            "as_shot_neutral": under_meta["neutral"],
        },
        "frames": frames,
    }
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards and num-shards >= 1")
    scenes = sorted(path.name for path in args.raw_root.glob("scene_*") if path.is_dir())
    scenes = scenes[args.shard_index :: args.num_shards]
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        raise FileNotFoundError("No scenes selected from {}".format(args.raw_root))

    report = {
        "tool": "opencv-fixed-raw-pair-isp",
        "storage_layout": "scene_xxxx/under|over/<source_raw_stem>.png",
        "settings": {
            "fixed_ev": 0.0,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "exposure_and_iso_used": False,
            "full_resolution_rgb": True,
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_scenes": len(scenes),
        "started_unix_s": time.time(),
        "completed": [],
        "skipped": [],
        "failed": {},
    }
    for index, scene in enumerate(scenes, start=1):
        if not args.force and scene_complete(scene, args):
            print("[{}/{}] {}: skipped".format(index, len(scenes), scene), flush=True)
            report["skipped"].append(scene)
            continue
        try:
            print("[{}/{}] {}: rendering".format(index, len(scenes), scene), flush=True)
            render_scene(scene, args)
            report["completed"].append(scene)
        except Exception as error:
            report["failed"][scene] = "{}: {}".format(type(error).__name__, error)
            print("[{}/{}] {}: FAILED: {}".format(index, len(scenes), scene, report["failed"][scene]), flush=True)

    report["finished_unix_s"] = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "batch_report_shard_{}_of_{}.json".format(
        args.shard_index, args.num_shards
    )
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

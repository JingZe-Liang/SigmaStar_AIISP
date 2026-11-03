#!/usr/bin/env python3
"""Render final fused RAW TIFFs and copy existing under/over fair RGB files.

Use this entrypoint when inference retained only
``scene_xxxx_fused_<CFA>_16bit.tiff`` rather than packed float32 NPY files.
The fused TIFF is normalized with the shared under/over DNG metadata and is
then rendered with fixed EV, highlight reconstruction, and Reinhard mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import tifffile

from opencv_source_match.fair_compare import render_fixed, validate_shared_metadata
from opencv_source_match.pipeline import cfa_name, crop_default, load_metadata_file, normalize_raw, save_rgb


DEFAULT_SOURCE_RGB_ROOT = Path(r"F:\RAW_MEF\data_RGB\iPhone17Pro\core_data_mef_highlight_reinhard")
DEFAULT_METADATA_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef")
CANONICAL_CHANNELS = ("R", "Gr", "B", "Gb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, required=True)
    parser.add_argument("--source-rgb-root", type=Path, default=DEFAULT_SOURCE_RGB_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[], help="Optional scene_xxxx; repeat to select scenes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def exactly_one(directory: Path, suffixes: tuple[str, ...]) -> Path:
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    if len(matches) != 1:
        raise FileNotFoundError("Expected one {} file in {}, found {}".format("/".join(suffixes), directory, len(matches)))
    return matches[0]


def fused_tiff_path(scene_dir: Path, scene: str) -> Path:
    matches = sorted(scene_dir.glob(scene + "_fused_*_16bit.tif*"))
    if len(matches) != 1:
        raise FileNotFoundError("Expected one fused TIFF in {}, found {}".format(scene_dir, len(matches)))
    return matches[0]


def source_rgb_paths(scene: str, root: Path) -> tuple[Path, Path, Path]:
    scene_root = root / scene
    under = exactly_one(scene_root / "under", (".png", ".jpg", ".jpeg"))
    over = exactly_one(scene_root / "over", (".png", ".jpg", ".jpeg"))
    report = scene_root / "fair_isp_report.json"
    if not report.is_file():
        raise FileNotFoundError("Missing source RGB fairness report: {}".format(report))
    return under, over, report


def validate_source_report(path: Path) -> dict:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        contract = content["fairness_contract"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid source RGB report {}: {}".format(path, error)) from error
    expected = {
        "fixed_ev_for_all_frames": 0.0,
        "uses_highlight_reconstruction": True,
        "tone_map": "reinhard",
        "false_color_suppression_enabled": False,
        "uses_camera_specific_tint": False,
    }
    mismatch = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
    if mismatch:
        raise ValueError("Source RGB ISP settings do not match high-recovery Reinhard mode: {}".format(mismatch))
    return content


def channel_positions(pattern: str) -> dict[str, tuple[int, int]]:
    pattern = pattern.upper()
    if len(pattern) != 4 or set(pattern) - {"R", "G", "B"}:
        raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
    tile = np.asarray(list(pattern)).reshape(2, 2)
    r_row, r_col = np.argwhere(tile == "R")[0]
    b_row, b_col = np.argwhere(tile == "B")[0]
    greens = [tuple(position) for position in np.argwhere(tile == "G")]
    gr = [position for position in greens if position[0] == r_row]
    gb = [position for position in greens if position[0] == b_row]
    if len(gr) != 1 or len(gb) != 1:
        raise ValueError("Cannot assign Gr/Gb locations for {}".format(pattern))
    return {
        "R": (int(r_row), int(r_col)),
        "Gr": (int(gr[0][0]), int(gr[0][1])),
        "B": (int(b_row), int(b_col)),
        "Gb": (int(gb[0][0]), int(gb[0][1])),
    }


def remosaic_to_metadata_cfa(mosaic: np.ndarray, source_cfa: str, metadata_cfa: str) -> np.ndarray:
    """Move each physical CFA sample into the metadata CFA layout when needed."""
    if source_cfa == metadata_cfa:
        return mosaic
    source = channel_positions(source_cfa)
    target = channel_positions(metadata_cfa)
    canonical = {name: mosaic[row::2, col::2] for name, (row, col) in source.items()}
    remapped = np.empty_like(mosaic)
    for name in CANONICAL_CHANNELS:
        row, col = target[name]
        remapped[row::2, col::2] = canonical[name]
    return remapped


def load_fused_mosaic(scene_dir: Path, scene: str, metadata: dict) -> tuple[np.ndarray, dict]:
    report_path = scene_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("Missing inference report: {}".format(report_path))
    try:
        inference_report = json.loads(report_path.read_text(encoding="utf-8"))
        source_cfa = str(inference_report["bayer_pattern"]).upper()
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read fused TIFF CFA from {}: {}".format(report_path, error)) from error
    path = fused_tiff_path(scene_dir, scene)
    raw = tifffile.imread(path)
    if raw.ndim != 2:
        raise ValueError("Expected one-channel fused RAW TIFF: {}".format(path))
    if raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError("Fused RAW TIFF dimensions must be even: {}".format(path))
    metadata_cfa = cfa_name(metadata["cfa"])
    normalized = normalize_raw(raw, metadata)
    normalized = remosaic_to_metadata_cfa(normalized, source_cfa, metadata_cfa)
    return crop_default(normalized, metadata), {
        "input": str(path.resolve()),
        "inference_report": str(report_path.resolve()),
        "source_tiff_cfa": source_cfa,
        "metadata_cfa": metadata_cfa,
        "was_remosaicked_to_metadata_cfa": source_cfa != metadata_cfa,
    }


def complete(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in ("under.png", "fused.png", "over.png", "fused_isp_report.json"))


def render_scene(scene: str, args: argparse.Namespace) -> dict:
    scene_dir = args.fused_root / scene
    source_under, source_over, source_report_path = source_rgb_paths(scene, args.source_rgb_root)
    source_report = validate_source_report(source_report_path)
    under_matches = sorted((args.metadata_root / scene / "under").glob("*_metadata_numeric.json"))
    over_matches = sorted((args.metadata_root / scene / "over").glob("*_metadata_numeric.json"))
    if len(under_matches) != 1 or len(over_matches) != 1:
        raise FileNotFoundError("Expected one numeric metadata JSON for under and over in {}".format(scene))
    under_metadata_path, over_metadata_path = under_matches[0], over_matches[0]
    under_metadata = load_metadata_file(under_metadata_path)
    over_metadata = load_metadata_file(over_metadata_path)
    validate_shared_metadata(under_metadata, over_metadata)
    fused_mosaic, fused_input = load_fused_mosaic(scene_dir, scene, under_metadata)
    fused_rgb, details = render_fixed(
        fused_mosaic,
        under_metadata,
        fixed_ev=0.0,
        enable_false_color_suppression=False,
        enable_highlight_recovery=True,
        tone_map="reinhard",
    )
    output_dir = args.output_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(fused_rgb, output_dir / "fused.png")
    shutil.copy2(source_under, output_dir / "under.png")
    shutil.copy2(source_over, output_dir / "over.png")
    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "fused_tiff_plus_preexisting_highlight_reinhard_sources",
        "scene": scene,
        "fused_input": fused_input,
        "fused_output": str((output_dir / "fused.png").resolve()),
        "fused_settings": {
            "fixed_ev": 0.0,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "false_color_suppression": False,
            "camera_tint": False,
        },
        "shared_metadata_reference": str(under_metadata_path.resolve()),
        "shared_metadata_validated_against": str(over_metadata_path.resolve()),
        "fused_details": details,
        "under_rgb": {
            "source": str(source_under.resolve()),
            "output": str((output_dir / "under.png").resolve()),
            "size_wh": list(Image.open(source_under).convert("RGB").size),
        },
        "over_rgb": {
            "source": str(source_over.resolve()),
            "output": str((output_dir / "over.png").resolve()),
            "size_wh": list(Image.open(source_over).convert("RGB").size),
        },
        "source_rgb_report": str(source_report_path.resolve()),
        "source_rgb_settings": source_report["fairness_contract"],
    }
    (output_dir / "fused_isp_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    scenes = sorted(
        directory.name for directory in args.fused_root.glob("scene_*")
        if (directory / "report.json").is_file()
    )
    if args.scene:
        wanted = set(args.scene)
        scenes = [scene for scene in scenes if scene in wanted]
        missing = wanted - set(scenes)
        if missing:
            raise FileNotFoundError("Requested scenes absent from fused root: {}".format(sorted(missing)))
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes:
        raise FileNotFoundError("No fused RAW scene directories under {}".format(args.fused_root))
    batch = {"total_scenes": len(scenes), "started_unix_s": time.time(), "completed": [], "skipped": [], "failed": {}}
    for index, scene in enumerate(scenes, start=1):
        output_dir = args.output_root / scene
        if not args.force and complete(output_dir):
            print("[{}/{}] {}: skipped".format(index, len(scenes), scene), flush=True)
            batch["skipped"].append(scene)
            continue
        try:
            print("[{}/{}] {}: rendering fused and copying sources".format(index, len(scenes), scene), flush=True)
            render_scene(scene, args)
            batch["completed"].append(scene)
        except Exception as error:
            message = "{}: {}".format(type(error).__name__, error)
            batch["failed"][scene] = message
            print("[{}/{}] {}: FAILED: {}".format(index, len(scenes), scene, message), flush=True)
    batch["finished_unix_s"] = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "batch_fused_isp_report.json"
    report_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Finished: completed={}, skipped={}, failed={}. Report: {}".format(
        len(batch["completed"]), len(batch["skipped"]), len(batch["failed"]), report_path
    ))
    if batch["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render native-DN fused packed RAW and collect existing under/over RGB files.

The four-channel inference code records the Bayer convention used when packing
the input TIFF in each scene's ``report.json``.  That convention may differ
from the camera metadata CFA.  Before demosaicking, this script maps the
packed values by their physical 2x2 Bayer locations, so an NPY is never
silently interpreted using the wrong CFA convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from opencv_source_match.fair_compare import (
    canonical_packed_to_mosaic,
    render_fixed,
    validate_shared_metadata,
)
from opencv_source_match.pipeline import cfa_name, crop_default, load_metadata_file, save_rgb


DEFAULT_FUSED_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_fourchannel_nativedn_snr\outputs\raw_mef_fused_nativedn_snr_test"
)
DEFAULT_SOURCE_RGB_ROOT = Path(r"F:\RAW_MEF\data_RGB\iPhone17Pro\core_data_mef_highlight_reinhard")
DEFAULT_METADATA_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef")
DEFAULT_OUTPUT_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_fourchannel_nativedn_snr\outputs\raw_mef_fixed_isp_nativedn_snr_highlight_reinhard_test"
)

CANONICAL_CHANNELS = ("R", "Gr", "B", "Gb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--source-rgb-root", type=Path, default=DEFAULT_SOURCE_RGB_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append", default=[], help="Optional scene_xxxx; repeat to select scenes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def exactly_one(directory: Path, suffixes: tuple[str, ...]) -> Path:
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    if len(matches) != 1:
        raise FileNotFoundError("Expected one {} file in {}, found {}".format("/".join(suffixes), directory, len(matches)))
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
        raise ValueError("Source RGB ISP settings do not match required high-recovery Reinhard mode: {}".format(mismatch))
    return content


def canonical_channel_positions(pattern: str) -> dict[str, tuple[int, int]]:
    """Return physical 2x2 locations for canonical R, Gr, B, Gb channels."""
    pattern = pattern.upper()
    if len(pattern) != 4 or set(pattern) - {"R", "G", "B"}:
        raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
    tile = np.asarray(list(pattern)).reshape(2, 2)
    r_row, r_col = np.argwhere(tile == "R")[0]
    b_row, b_col = np.argwhere(tile == "B")[0]
    green_positions = [tuple(position) for position in np.argwhere(tile == "G")]
    gr_positions = [position for position in green_positions if position[0] == r_row]
    gb_positions = [position for position in green_positions if position[0] == b_row]
    if len(gr_positions) != 1 or len(gb_positions) != 1:
        raise ValueError("Cannot assign Gr/Gb positions for {}".format(pattern))
    return {
        "R": (int(r_row), int(r_col)),
        "Gr": (int(gr_positions[0][0]), int(gr_positions[0][1])),
        "B": (int(b_row), int(b_col)),
        "Gb": (int(gb_positions[0][0]), int(gb_positions[0][1])),
    }


def packed_permutation(source_cfa: str, target_cfa: str) -> tuple[int, int, int, int]:
    """Map packed canonical data from source-CFA coordinates to target-CFA coordinates."""
    source_positions = canonical_channel_positions(source_cfa)
    target_positions = canonical_channel_positions(target_cfa)
    source_index_at_position = {
        position: index for index, name in enumerate(CANONICAL_CHANNELS)
        for position in (source_positions[name],)
    }
    return tuple(source_index_at_position[target_positions[name]] for name in CANONICAL_CHANNELS)


def load_inference_packed(scene_dir: Path, metadata_cfa: str) -> tuple[np.ndarray, dict]:
    """Load packed inference output and align it to the metadata CFA convention."""
    packed_path = scene_dir / "fused_packed_native_dn_float32.npy"
    inference_report_path = scene_dir / "report.json"
    if not packed_path.is_file():
        raise FileNotFoundError(packed_path)
    if not inference_report_path.is_file():
        raise FileNotFoundError(
            "Missing inference report needed to interpret packed RAW CFA: {}".format(
                inference_report_path
            )
        )
    try:
        inference_report = json.loads(inference_report_path.read_text(encoding="utf-8"))
        source_cfa = str(inference_report["bayer_pattern"]).upper()
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Invalid inference report {}; cannot determine packed RAW CFA: {}".format(
                inference_report_path, error
            )
        ) from error

    packed = np.load(packed_path, allow_pickle=False)
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError("Expected [4,H,W] packed RAW in {}, got {}".format(packed_path, packed.shape))
    permutation = packed_permutation(source_cfa, metadata_cfa)
    return np.ascontiguousarray(packed[list(permutation)]), {
        "input": str(packed_path.resolve()),
        "inference_report": str(inference_report_path.resolve()),
        "source_packed_cfa": source_cfa,
        "metadata_cfa": metadata_cfa,
        "permutation_to_metadata_canonical": list(permutation),
        "was_reordered": permutation != (0, 1, 2, 3),
    }


def complete(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in ("under.png", "fused.png", "over.png", "fused_isp_report.json"))


def render_scene(scene: str, args: argparse.Namespace) -> dict:
    fused_scene_dir = args.fused_root / scene
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

    metadata_cfa = cfa_name(under_metadata["cfa"])
    fused_packed, packed_layout = load_inference_packed(fused_scene_dir, metadata_cfa)
    fused_mosaic, clipped_fraction = canonical_packed_to_mosaic(
        fused_packed, under_metadata["cfa"]
    )
    fused_mosaic = crop_default(fused_mosaic, under_metadata)
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
    fused_output = output_dir / "fused.png"
    save_rgb(fused_rgb, fused_output)
    shutil.copy2(source_under, output_dir / "under.png")
    shutil.copy2(source_over, output_dir / "over.png")

    source_sizes = {
        "under": list(Image.open(source_under).convert("RGB").size),
        "over": list(Image.open(source_over).convert("RGB").size),
    }
    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "mode": "nativedn_fused_plus_preexisting_highlight_reinhard_sources_no_preview_cfa_safe",
        "scene": scene,
        "fused_input": packed_layout["input"],
        "fused_output": str(fused_output.resolve()),
        "fused_packed_contract": (
            "[R, Gr, B, Gb] normalized packed RAW in the inference report CFA; "
            "reordered by physical Bayer position to the metadata CFA before ISP"
        ),
        "fused_packed_layout": packed_layout,
        "fused_input_out_of_range_fraction_clipped_before_demosaic": clipped_fraction,
        "fused_settings": {
            "fixed_ev": 0.0,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "false_color_suppression": False,
            "camera_tint": False,
            "comparison_preview_generated": False,
        },
        "shared_metadata_reference": str(under_metadata_path.resolve()),
        "shared_metadata_validated_against": str(over_metadata_path.resolve()),
        "fused_details": details,
        "under_rgb": {"source": str(source_under.resolve()), "output": str((output_dir / "under.png").resolve()), "size_wh": source_sizes["under"]},
        "over_rgb": {"source": str(source_over.resolve()), "output": str((output_dir / "over.png").resolve()), "size_wh": source_sizes["over"]},
        "source_rgb_report": str(source_report_path.resolve()),
        "source_rgb_settings": source_report["fairness_contract"],
    }
    (output_dir / "fused_isp_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    scenes = sorted(
        directory.name for directory in args.fused_root.glob("scene_*")
        if (directory / "fused_packed_native_dn_float32.npy").is_file()
    )
    if args.scene:
        wanted = set(args.scene)
        scenes = [scene for scene in scenes if scene in wanted]
        missing = wanted - set(scenes)
        if missing:
            raise FileNotFoundError("Requested scenes absent from fused-root: {}".format(sorted(missing)))
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes:
        raise FileNotFoundError("No fused packed RAW scenes under {}".format(args.fused_root))
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

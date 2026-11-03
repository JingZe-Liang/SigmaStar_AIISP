#!/usr/bin/env python3
"""Correct fair-ISP outputs when inference and metadata Bayer layouts differ.

The fused NPY is canonical relative to the source CFA recorded in its
inference report.  Before a metadata-driven ISP pass, its four planes must be
reassigned by physical 2x2 Bayer location when the metadata CFA differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from opencv_source_match.fair_compare import render_comparison
from opencv_source_match.pipeline import cfa_name, load_metadata_file


DEFAULT_FUSED_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_fourchannel_geomean_relative_snr_spatial_chroma_hard_source_chroma_post_morph\outputs\raw_mef_fused_hard_source_chroma_post_morph_from_legacy_tiff_test"
)
DEFAULT_RAW_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_data_mef")
DEFAULT_METADATA_ROOT = Path(r"F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef")
DEFAULT_OUTPUT_ROOT = Path(
    r"F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_fourchannel_geomean_relative_snr_spatial_chroma_hard_source_chroma_post_morph\outputs\raw_mef_fixed_isp_hard_source_chroma_post_morph_highlight_reinhard_from_legacy_tiff_test"
)
CANONICAL_CHANNELS = ("R", "Gr", "B", "Gb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append", default=[], help="Optional scene_xxxx; repeat to select scenes")
    return parser.parse_args()


def exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError("Expected one {} in {}, found {}".format(pattern, directory, len(matches)))
    return matches[0]


def canonical_channel_positions(pattern: str) -> dict[str, tuple[int, int]]:
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


def packed_permutation(source_cfa: str, metadata_cfa: str) -> tuple[int, int, int, int]:
    source_positions = canonical_channel_positions(source_cfa)
    metadata_positions = canonical_channel_positions(metadata_cfa)
    source_index_at_position = {
        source_positions[channel]: index for index, channel in enumerate(CANONICAL_CHANNELS)
    }
    return tuple(source_index_at_position[metadata_positions[channel]] for channel in CANONICAL_CHANNELS)


def input_paths(scene: str, args: argparse.Namespace) -> dict[str, Path]:
    return {
        "fused": args.fused_root / scene / "fused_packed_float32.npy",
        "report": args.fused_root / scene / "report.json",
        "under_raw": exactly_one(args.raw_root / scene / "under", scene + "_unprocessed_*_16bit.tiff"),
        "over_raw": exactly_one(args.raw_root / scene / "over", scene + "_unprocessed_*_16bit.tiff"),
        "under_metadata": exactly_one(args.metadata_root / scene / "under", scene + "_metadata_numeric.json"),
        "over_metadata": exactly_one(args.metadata_root / scene / "over", scene + "_metadata_numeric.json"),
    }


def source_cfa(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["bayer_pattern"]).upper()
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read source Bayer pattern from {}".format(path)) from error


def render_scene(scene: str, args: argparse.Namespace) -> dict:
    paths = input_paths(scene, args)
    source_pattern = source_cfa(paths["report"])
    metadata_pattern = cfa_name(load_metadata_file(paths["under_metadata"])["cfa"])
    if source_pattern == metadata_pattern:
        return {"scene": scene, "status": "skipped_matching_cfa"}
    packed = np.load(paths["fused"], allow_pickle=False)
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError("Expected [4,H,W] packed RAW in {}".format(paths["fused"]))
    permutation = packed_permutation(source_pattern, metadata_pattern)
    metadata_packed = np.ascontiguousarray(packed[list(permutation)])
    metadata_packed_path = paths["fused"].with_name("fused_packed_metadata_cfa_float32.npy")
    np.save(metadata_packed_path, metadata_packed)

    output_dir = args.output_root / scene
    report = render_comparison(
        under_raw=paths["under_raw"],
        over_raw=paths["over_raw"],
        fused_packed=metadata_packed_path,
        under_metadata=paths["under_metadata"],
        over_metadata=paths["over_metadata"],
        output_dir=output_dir,
        fixed_ev=0.0,
        enable_false_color_suppression=False,
        enable_highlight_recovery=True,
        tone_map="reinhard",
        max_long_edge=0,
        preview_long_edge=1200,
        image_extension=".png",
    )
    report["fused_packed_layout"] = {
        "source_input": str(paths["fused"].resolve()),
        "source_cfa": source_pattern,
        "metadata_cfa": metadata_pattern,
        "permutation_to_metadata_canonical": list(permutation),
        "metadata_cfa_reordered_input": str(metadata_packed_path.resolve()),
        "reason": "source and metadata CFA differ; planes reassigned by physical Bayer position before ISP",
    }
    report["cfa_correction_applied"] = True
    report_path = output_dir / "fairness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"scene": scene, "status": "rerendered", "permutation": list(permutation)}


def main() -> None:
    args = parse_args()
    args.fused_root = args.fused_root.resolve()
    args.raw_root = args.raw_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    args.output_root = args.output_root.resolve()
    scenes = sorted(
        directory.name for directory in args.fused_root.glob("scene_*")
        if (directory / "fused_packed_float32.npy").is_file() and (directory / "report.json").is_file()
    )
    if args.scene:
        wanted = set(args.scene)
        scenes = [scene for scene in scenes if scene in wanted]
        missing = wanted - set(scenes)
        if missing:
            raise FileNotFoundError("Requested scenes absent from fused root: {}".format(sorted(missing)))
    manifest = {"started_unix_s": time.time(), "rerendered": [], "skipped_matching_cfa": [], "failed": {}}
    for scene in scenes:
        try:
            result = render_scene(scene, args)
            manifest[result["status"]].append(result["scene"])
        except Exception as error:
            manifest["failed"][scene] = "{}: {}".format(type(error).__name__, error)
    manifest["finished_unix_s"] = time.time()
    manifest_path = args.output_root / "cfa_correction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Finished: rerendered={}, matching_cfa={}, failed={}. Report: {}".format(
        len(manifest["rerendered"]), len(manifest["skipped_matching_cfa"]), len(manifest["failed"]), manifest_path
    ))
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

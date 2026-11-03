from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fair_compare import render_comparison
from .pipeline import EXPOSURES, process_frame, render_scene, save_rgb


def _positive_or_zero(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or positive")
    return parsed


def _add_resize_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-long-edge",
        type=_positive_or_zero,
        default=0,
        help="Resize final RGB output after ISP; 0 keeps full resolution.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traditional OpenCV source-match ISP for DatasetMEF extracted RAW data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    frame = subparsers.add_parser("frame", help="Render one exposure from one scene.")
    frame.add_argument("--scene-dir", required=True, type=Path)
    frame.add_argument("--exposure", required=True, choices=EXPOSURES)
    frame.add_argument("--output", required=True, type=Path)
    frame.add_argument("--report", type=Path)
    _add_resize_argument(frame)

    scene = subparsers.add_parser("scene", help="Render under/normal/over for one scene.")
    scene.add_argument("--scene-dir", required=True, type=Path)
    scene.add_argument("--output-dir", required=True, type=Path)
    scene.add_argument("--format", choices=("png", "jpg", "tiff"), default="png")
    _add_resize_argument(scene)

    dataset = subparsers.add_parser("dataset", help="Render multiple DatasetMEF scenes.")
    dataset.add_argument("--dataset", required=True, type=Path)
    dataset.add_argument("--output-dir", required=True, type=Path)
    dataset.add_argument(
        "--scenes",
        nargs="*",
        help="Optional scene names, for example scene_0001 scene_0245.",
    )
    dataset.add_argument("--limit", type=_positive_or_zero, default=0)
    dataset.add_argument("--format", choices=("png", "jpg", "tiff"), default="png")
    dataset.add_argument("--fail-fast", action="store_true")
    _add_resize_argument(dataset)

    compare = subparsers.add_parser(
        "fair-compare",
        help="Render under/fused/over with one fixed, shared traditional ISP.",
    )
    compare.add_argument("--under-raw", required=True, type=Path)
    compare.add_argument("--over-raw", required=True, type=Path)
    compare.add_argument(
        "--fused-packed",
        required=True,
        type=Path,
        help="U2Fusion packed float32 NPY in canonical [R,Gr,B,Gb] order.",
    )
    compare.add_argument("--metadata-under", required=True, type=Path)
    compare.add_argument("--metadata-over", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    compare.add_argument(
        "--fixed-ev",
        type=float,
        default=0.0,
        help="One constant EV applied to all three frames; default: 0.",
    )
    compare.add_argument(
        "--enable-false-color-suppression",
        action="store_true",
        help="Use the same fixed chroma artifact filter for all frames.",
    )
    compare.add_argument(
        "--enable-highlight-recovery",
        action="store_true",
        help="Use the same fixed clipped-channel reconstruction for all frames.",
    )
    compare.add_argument(
        "--tone-map",
        choices=("hard-clip", "reinhard"),
        default="hard-clip",
        help="One fixed output tone map for all frames; default: hard-clip.",
    )
    compare.add_argument(
        "--preview-long-edge",
        type=_positive_or_zero,
        default=1200,
        help="Resize only comparison_preview.png; 0 keeps its full resolution.",
    )
    compare.add_argument("--format", choices=("png", "jpg", "tiff"), default="png")
    _add_resize_argument(compare)
    return parser


def _run_frame(args: argparse.Namespace) -> int:
    result = process_frame(
        args.scene_dir,
        args.exposure,
        max_long_edge=args.max_long_edge,
    )
    save_rgb(result.image, args.output)
    result.report["output_image"] = str(args.output)
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote image: {args.output}")
    print(f"Wrote report: {report_path}")
    return 0


def _run_scene(args: argparse.Namespace) -> int:
    reports = render_scene(
        args.scene_dir,
        args.output_dir,
        max_long_edge=args.max_long_edge,
        image_extension=f".{args.format}",
    )
    print(f"Rendered {len(reports)} frames to: {args.output_dir}")
    return 0


def _run_fair_compare(args: argparse.Namespace) -> int:
    report = render_comparison(
        under_raw=args.under_raw,
        over_raw=args.over_raw,
        fused_packed=args.fused_packed,
        under_metadata=args.metadata_under,
        over_metadata=args.metadata_over,
        output_dir=args.output_dir,
        fixed_ev=args.fixed_ev,
        enable_false_color_suppression=args.enable_false_color_suppression,
        enable_highlight_recovery=args.enable_highlight_recovery,
        tone_map=args.tone_map,
        max_long_edge=args.max_long_edge,
        preview_long_edge=args.preview_long_edge,
        image_extension="." + args.format,
    )
    print("Wrote fair comparison to: {}".format(args.output_dir))
    print("Wrote fairness report: {}".format(args.output_dir / "fairness_report.json"))
    return 0


def _scene_directories(dataset: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [dataset / name for name in requested]
    return sorted(
        path for path in dataset.iterdir() if path.is_dir() and path.name.startswith("scene_")
    )


def _run_dataset(args: argparse.Namespace) -> int:
    if not args.dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {args.dataset}")

    scenes = _scene_directories(args.dataset, args.scenes)
    if args.limit:
        scenes = scenes[: args.limit]

    batch_report = {
        "tool": "opencv-source-match",
        "dataset": str(args.dataset),
        "requested_scene_count": len(scenes),
        "completed": [],
        "failed": [],
    }
    for index, scene_dir in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene_dir.name}")
        try:
            scene_output = args.output_dir / scene_dir.name
            reports = render_scene(
                scene_dir,
                scene_output,
                max_long_edge=args.max_long_edge,
                image_extension=f".{args.format}",
            )
            batch_report["completed"].append(
                {"scene": scene_dir.name, "frames": len(reports)}
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"ERROR {scene_dir.name}: {message}", file=sys.stderr)
            batch_report["failed"].append(
                {"scene": scene_dir.name, "error": message}
            )
            if args.fail_fast:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "batch_processing_report.json"
    report_path.write_text(
        json.dumps(batch_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote batch report: {report_path}")
    return 1 if batch_report["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "frame":
            return _run_frame(args)
        if args.command == "scene":
            return _run_scene(args)
        if args.command == "fair-compare":
            return _run_fair_compare(args)
        return _run_dataset(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

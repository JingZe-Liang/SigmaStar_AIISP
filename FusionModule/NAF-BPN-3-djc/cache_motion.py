from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from data import FRAME_COUNT, HEIGHT, WIDTH, discover_sequences, read_source
from motion_detection.robust_raw_md import MDConfig, run_motion_detection
from train import config_path, read_config


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training-only Robust MD cache")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cloud.json")
    parser.add_argument("--sequence", choices=("128x", "645x", "all"), default="all")
    parser.add_argument("--force", action="store_true", help="Regenerate an existing complete cache")
    return parser.parse_args()


def cache_complete(cache_root: Path, sequence_name: str) -> bool:
    masks = cache_root / sequence_name / "masks"
    return all((masks / f"{index:04d}.png").is_file() for index in range(FRAME_COUNT))


def validate_masks(cache_root: Path, sequence_name: str) -> dict[str, float | int]:
    masks = cache_root / sequence_name / "masks"
    coverage: list[float] = []
    for index in range(FRAME_COUNT):
        mask = cv2.imread(str(masks / f"{index:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (HEIGHT // 2, WIDTH // 2):
            raise ValueError(f"MD mask 格式错误: {masks / f'{index:04d}.png'}")
        coverage.append(float((mask > 0).mean()))
    return {
        "frames": FRAME_COUNT,
        "mean_motion_coverage": float(np.mean(coverage)),
        "max_motion_coverage": float(np.max(coverage)),
    }


def cache_sequence(sequence, cache_root: Path, force: bool) -> None:
    destination = cache_root / sequence.name
    if cache_complete(cache_root, sequence.name) and not force:
        print(f"{sequence.name}: reuse complete Robust MD cache", flush=True)
    else:
        if destination.exists() and not force:
            raise FileExistsError(f"{destination} is incomplete; rerun with --force to regenerate it")
        destination.mkdir(parents=True, exist_ok=True)
        stack = np.empty((FRAME_COUNT, HEIGHT, WIDTH), dtype=np.uint16)
        for index in range(FRAME_COUNT):
            stack[index] = read_source(sequence, index)
        detector_config = MDConfig(
            mode="robust",
            cfa_pattern=sequence.cfa_pattern,
            black_level=sequence.source_black_level,
            white_level=sequence.white_level,
        )
        run_motion_detection(
            stack,
            str(destination),
            config=detector_config,
            fps=25,
            save_masks=True,
            save_bboxes=True,
            save_video=True,
            verbose=True,
        )
        del stack
    summary = {
        "sequence": sequence.name,
        "detector": "robust_raw_md",
        "cfa_pattern": sequence.cfa_pattern,
        "black_level": sequence.source_black_level,
        "white_level": sequence.white_level,
        **validate_masks(cache_root, sequence.name),
    }
    (destination / "cache_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    sequences = discover_sequences(
        config_path(config, "data_root"),
        tuple(config["sequence_names"]),
        None,
        str(config["cfa_pattern"]),
        int(config["source_black_level"]),
        int(config["dnr_black_level"]),
        int(config["white_level"]),
    )
    if args.sequence != "all":
        sequences = tuple(item for item in sequences if item.name == args.sequence)
    cache_root = config_path(config, "motion_cache_root")
    for sequence in sequences:
        cache_sequence(sequence, cache_root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

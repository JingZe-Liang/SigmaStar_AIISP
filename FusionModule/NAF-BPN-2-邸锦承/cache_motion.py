from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from data import FRAME_COUNT, HEIGHT, WIDTH, SOURCE_BLACK_LEVEL, discover_sequences, read_source


ROOT = Path(__file__).resolve().parent
MD_DIR = Path(r"D:\DeepLearning\VideoDenoising\SigmaStar_AIISP-main\utils\MD\robust_raw_md_肖纬杰_已验证")


def parse_args():
    parser = argparse.ArgumentParser(description="顺序预计算公司 RAW 的 robust MD 缓存")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--sequence", choices=("128x", "645x", "all"), default="all")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的完整 MD 缓存")
    return parser.parse_args()


def md_module():
    if not MD_DIR.is_dir():
        raise FileNotFoundError(f"MD 目录不存在: {MD_DIR}")
    sys.path.insert(0, str(MD_DIR))
    return importlib.import_module("noise_adaptive_flicker_robust_md")


def contact_sheet(sequence, cache_root: Path, indices: list[int]) -> None:
    panels = []
    for index in indices:
        source = read_source(sequence, index)
        display = np.clip((source.astype(np.float32) - SOURCE_BLACK_LEVEL) / (4095 - SOURCE_BLACK_LEVEL), 0.0, 1.0)
        display = np.rint(np.power(display, 1 / 2.2) * 255).astype(np.uint8)
        display = cv2.resize(display, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA)
        mask = cv2.imread(str(cache_root / sequence.name / "masks" / f"{index:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"缺少 MD mask: {sequence.name}/{index:04d}")
        mask = cv2.resize(mask, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_NEAREST)
        panel = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        panel[mask > 0] = (0, 0, 255)
        cv2.putText(panel, f"{sequence.name} frame {index:03d}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    rows = [np.concatenate(panels[row:row + 5], axis=1) for row in range(0, len(panels), 5)]
    cv2.imwrite(str(cache_root / sequence.name / "motion_contact_sheet.png"), np.concatenate(rows, axis=0))


def cache_sequence(sequence, cache_root: Path, module, force: bool) -> None:
    destination = cache_root / sequence.name
    masks = destination / "masks"
    existing = list(masks.glob("*.png")) if masks.is_dir() else []
    if len(existing) == FRAME_COUNT and not force:
        print(f"{sequence.name}: 复用完整 MD 缓存", flush=True)
    else:
        if destination.exists() and not force:
            raise FileExistsError(f"{destination} 存在不完整 MD 缓存；使用 --force 才可覆盖")
        stack = np.empty((FRAME_COUNT, HEIGHT, WIDTH), dtype=np.uint16)
        for index in range(FRAME_COUNT):
            stack[index] = read_source(sequence, index)
        config = module.MDConfig(mode="robust", cfa_pattern="BGGR", black_level=SOURCE_BLACK_LEVEL, white_level=4095)
        module.run_motion_detection(stack, str(destination), config=config, fps=25, save_masks=True, save_bboxes=True, save_video=True, verbose=True)
        del stack
    mask_paths = [masks / f"{index:04d}.png" for index in range(FRAME_COUNT)]
    if not all(path.is_file() for path in mask_paths):
        raise RuntimeError(f"{sequence.name}: MD 缓存未完整产出 200 张 mask")
    coverage = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (HEIGHT // 2, WIDTH // 2):
            raise ValueError(f"MD mask 大小错误: {path}")
        coverage.append(float((mask > 0).mean()))
    indices = np.linspace(0, FRAME_COUNT - 1, 10).round().astype(int).tolist()
    contact_sheet(sequence, cache_root, indices)
    summary = {"sequence": sequence.name, "frames": FRAME_COUNT, "mask_shape": [HEIGHT // 2, WIDTH // 2], "sample_indices": indices, "mean_motion_coverage": float(np.mean(coverage)), "max_motion_coverage": float(np.max(coverage)), "md": {"mode": "robust", "cfa_pattern": "BGGR", "black_level": SOURCE_BLACK_LEVEL, "white_level": 4095}}
    (destination / "cache_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sequences = discover_sequences(Path(config["data_root"]), tuple(config["sequence_names"]))
    if args.sequence != "all":
        sequences = tuple(item for item in sequences if item.name == args.sequence)
    module = md_module()
    root = Path(config["motion_cache_root"])
    for sequence in sequences:
        cache_sequence(sequence, root, module, args.force)


if __name__ == "__main__":
    main()

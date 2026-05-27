from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff

ISO_PATTERN = re.compile(r"ISO\s*[_-]?(\d+)", re.IGNORECASE)
SUPPORTED_EXTS = {".tif", ".tiff"}


@dataclass
class Config:
    clean_root: Path
    dark_root: Path
    output_root: Path
    black_level: float
    white_level: float
    base_iso: float
    qe: float
    seed: int
    save_dark_shading: bool
    max_dark_frames_per_iso: Optional[int]


@dataclass
class DarkGroup:
    iso: int
    files: List[Path]
    dark_shading: np.ndarray


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Realistic RAW noise synthesis: shot noise + sampled dark residual"
    )
    parser.add_argument("--clean_root", type=Path, required=True, help="Clean RAW 根目录")
    parser.add_argument("--dark_root", type=Path, required=True, help="Dark frame 根目录")
    parser.add_argument("--output_root", type=Path, required=True, help="输出根目录")
    parser.add_argument("--black_level", type=float, default=240.0, help="黑电平，默认 240")
    parser.add_argument("--white_level", type=float, default=4095.0, help="白电平，默认 4095")
    parser.add_argument(
        "--base_iso",
        type=float,
        default=400.0,
        help="基础 ISO（对应 AG=1 的近似值），默认 400",
    )
    parser.add_argument(
        "--qe",
        type=float,
        default=0.4,
        help="假设量子效率 QE，默认 0.4；K ≈ (ISO/base_iso) * qe",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--save_dark_shading",
        action="store_true",
        help="是否把每个 ISO 的 dark shading 另存为 tiff",
    )
    parser.add_argument(
        "--max_dark_frames_per_iso",
        type=int,
        default=None,
        help="每个 ISO 最多使用多少张 dark frame；默认全部使用",
    )
    args = parser.parse_args()

    return Config(
        clean_root=args.clean_root,
        dark_root=args.dark_root,
        output_root=args.output_root,
        black_level=args.black_level,
        white_level=args.white_level,
        base_iso=args.base_iso,
        qe=args.qe,
        seed=args.seed,
        save_dark_shading=args.save_dark_shading,
        max_dark_frames_per_iso=args.max_dark_frames_per_iso,
    )


def extract_iso(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = ISO_PATTERN.search(part)
        if match:
            return int(match.group(1))
    match = ISO_PATTERN.search(path.name)
    if match:
        return int(match.group(1))
    return None


def find_tiff_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    files.sort()
    return files


def load_tiff_float32(path: Path) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim != 2:
        raise ValueError(f"目前只支持单通道 2D RAW，文件 {path} 的维度是 {arr.shape}")
    return arr.astype(np.float32)


def compute_dark_shading(files: Sequence[Path]) -> np.ndarray:
    if not files:
        raise ValueError("dark frame 列表为空，无法计算 dark shading")

    mean_arr: Optional[np.ndarray] = None
    count = 0
    for fp in files:
        arr = load_tiff_float32(fp)
        if mean_arr is None:
            mean_arr = np.zeros_like(arr, dtype=np.float64)
        if arr.shape != mean_arr.shape:
            raise ValueError(
                f"同一个 ISO 下 dark frame 尺寸不一致：{files[0]} vs {fp}"
            )
        mean_arr += arr.astype(np.float64)
        count += 1

    assert mean_arr is not None
    return (mean_arr / count).astype(np.float32)


def build_dark_groups(cfg: Config) -> Dict[int, DarkGroup]:
    all_dark_files = find_tiff_files(cfg.dark_root)
    if not all_dark_files:
        raise FileNotFoundError(f"在 {cfg.dark_root} 下没有找到 tif/tiff 文件")

    by_iso: Dict[int, List[Path]] = defaultdict(list)
    for fp in all_dark_files:
        iso = extract_iso(fp)
        if iso is None:
            continue
        by_iso[iso].append(fp)

    if not by_iso:
        raise RuntimeError("没有从 dark_root 中解析出任何 ISO，请检查文件夹命名，例如 ISO1600")

    rng = random.Random(cfg.seed)
    groups: Dict[int, DarkGroup] = {}
    for iso, files in sorted(by_iso.items()):
        files = sorted(files)
        if cfg.max_dark_frames_per_iso is not None and len(files) > cfg.max_dark_frames_per_iso:
            files = rng.sample(files, cfg.max_dark_frames_per_iso)
            files = sorted(files)
        dark_shading = compute_dark_shading(files)
        groups[iso] = DarkGroup(iso=iso, files=files, dark_shading=dark_shading)
    return groups


def system_gain_from_iso(iso: int, base_iso: float, qe: float) -> float:
    return (iso / base_iso) * qe


def synthesize_noisy_raw(
    clean_raw: np.ndarray,
    dark_sample: np.ndarray,
    dark_shading: np.ndarray,
    black_level: float,
    white_level: float,
    k_value: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if clean_raw.shape != dark_sample.shape or clean_raw.shape != dark_shading.shape:
        raise ValueError(
            f"尺寸不一致：clean={clean_raw.shape}, dark_sample={dark_sample.shape}, dark_shading={dark_shading.shape}"
        )
    if k_value <= 0:
        raise ValueError(f"system gain K 必须大于 0，当前 K={k_value}")

    signal = np.clip(clean_raw - black_level, 0.0, None)
    lam = signal / k_value
    shot_signal = rng.poisson(lam).astype(np.float32) * np.float32(k_value)

    dark_residual = dark_sample - dark_shading
    noisy = shot_signal + np.float32(black_level) + dark_residual
    noisy = np.clip(noisy, 0.0, white_level)

    return noisy.astype(np.float32), shot_signal.astype(np.float32), dark_residual.astype(np.float32)


def make_output_path(clean_path: Path, clean_root: Path, output_root: Path) -> Path:
    rel = clean_path.relative_to(clean_root)
    new_name = clean_path.stem + "_noisy" + clean_path.suffix
    return output_root / rel.parent / new_name


def maybe_save_dark_shading(groups: Dict[int, DarkGroup], output_root: Path, white_level: float) -> None:
    shading_root = output_root / "_dark_shading"
    shading_root.mkdir(parents=True, exist_ok=True)
    for iso, group in groups.items():
        out_path = shading_root / f"ISO{iso}_dark_shading.tiff"
        arr = np.clip(np.round(group.dark_shading), 0, white_level).astype(np.uint16)
        tiff.imwrite(out_path, arr)


def write_run_metadata(cfg: Config, groups: Dict[int, DarkGroup]) -> None:
    meta = {
        "pipeline": "clean RAW + shot noise + sampled dark residual",
        "formula": {
            "dark_shading": "DS_iso = mean(D_1, ..., D_N)",
            "dark_residual": "R_k = D_k - DS_iso",
            "system_gain": "K ≈ (ISO / base_iso) * qe",
            "shot_noise": "I_shot = K * Poisson(max(I_clean - BL, 0) / K)",
            "noisy_raw": "I_noisy = clip(I_shot + BL + R_k, 0, WL)",
        },
        "config": {
            "clean_root": str(cfg.clean_root),
            "dark_root": str(cfg.dark_root),
            "output_root": str(cfg.output_root),
            "black_level": cfg.black_level,
            "white_level": cfg.white_level,
            "base_iso": cfg.base_iso,
            "qe": cfg.qe,
            "seed": cfg.seed,
            "save_dark_shading": cfg.save_dark_shading,
            "max_dark_frames_per_iso": cfg.max_dark_frames_per_iso,
        },
        "dark_frames_per_iso": {str(iso): len(group.files) for iso, group in groups.items()},
    }
    with open(cfg.output_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def process_dataset(cfg: Config) -> None:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    rng_py = random.Random(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed)

    dark_groups = build_dark_groups(cfg)
    if cfg.save_dark_shading:
        maybe_save_dark_shading(dark_groups, cfg.output_root, cfg.white_level)

    clean_files = find_tiff_files(cfg.clean_root)
    if not clean_files:
        raise FileNotFoundError(f"在 {cfg.clean_root} 下没有找到 tif/tiff 文件")

    log_path = cfg.output_root / "noise_synthesis_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "clean_path",
                "output_path",
                "iso",
                "k_value",
                "dark_sample_path",
                "min_noisy",
                "max_noisy",
                "mean_noisy",
            ]
        )

        processed = 0
        skipped = 0
        for clean_path in clean_files:
            iso = extract_iso(clean_path)
            if iso is None:
                print(f"[跳过] 无法从路径解析 ISO: {clean_path}")
                skipped += 1
                continue
            if iso not in dark_groups:
                print(f"[跳过] 没有找到 ISO{iso} 对应的 dark frames: {clean_path}")
                skipped += 1
                continue

            clean_raw = load_tiff_float32(clean_path)
            group = dark_groups[iso]
            dark_path = rng_py.choice(group.files)
            dark_sample = load_tiff_float32(dark_path)
            k_value = system_gain_from_iso(iso, cfg.base_iso, cfg.qe)

            noisy, _, _ = synthesize_noisy_raw(
                clean_raw=clean_raw,
                dark_sample=dark_sample,
                dark_shading=group.dark_shading,
                black_level=cfg.black_level,
                white_level=cfg.white_level,
                k_value=k_value,
                rng=rng_np,
            )

            out_path = make_output_path(clean_path, cfg.clean_root, cfg.output_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_arr = np.clip(np.round(noisy), 0, cfg.white_level).astype(np.uint16)
            tiff.imwrite(out_path, save_arr)

            writer.writerow(
                [
                    str(clean_path),
                    str(out_path),
                    iso,
                    f"{k_value:.6f}",
                    str(dark_path),
                    f"{float(noisy.min()):.4f}",
                    f"{float(noisy.max()):.4f}",
                    f"{float(noisy.mean()):.4f}",
                ]
            )
            processed += 1
            if processed % 20 == 0:
                print(f"已处理 {processed} 张")

    write_run_metadata(cfg, dark_groups)
    print(f"完成：processed={processed}, skipped={skipped}")
    print(f"输出目录：{cfg.output_root}")


if __name__ == "__main__":
    config = parse_args()
    process_dataset(config)

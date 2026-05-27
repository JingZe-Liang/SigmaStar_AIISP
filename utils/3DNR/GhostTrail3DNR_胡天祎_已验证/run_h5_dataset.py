from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .ghost_trail_3dnr import (
        GhostTrail3DNRConfig,
        GhostTrail3DNRProcessor,
        box_blur_2d,
        restore_raw_dtype,
    )
    from .run_h5_demo import (
        crop_from_center,
        estimate_gray_world_gains,
        make_panel,
        normalize_for_preview,
        raw_to_luma_preview,
        raw_to_srgb_half,
        rgb_to_uint8_preview,
        save_gif,
    )
except ImportError:
    from ghost_trail_3dnr import (
        GhostTrail3DNRConfig,
        GhostTrail3DNRProcessor,
        box_blur_2d,
        restore_raw_dtype,
    )
    from run_h5_demo import (
        crop_from_center,
        estimate_gray_world_gains,
        make_panel,
        normalize_for_preview,
        raw_to_luma_preview,
        raw_to_srgb_half,
        rgb_to_uint8_preview,
        save_gif,
    )


METHODS = ("noisy", "2dnr", "dataset_3dnr", "ghost3dnr")


@dataclass
class MetricState:
    frame_count: int = 0
    mse_sum: float = 0.0
    psnr_sum: float = 0.0
    ssim_sum: float = 0.0
    static_mse_sum: float = 0.0
    moving_mae_sum: float = 0.0
    static_count: int = 0
    moving_count: int = 0

    def update(self, mse: float, psnr: float, ssim: float, static_mse: float | None, moving_mae: float | None) -> None:
        self.frame_count += 1
        self.mse_sum += mse
        self.psnr_sum += psnr
        self.ssim_sum += ssim
        if static_mse is not None:
            self.static_mse_sum += static_mse
            self.static_count += 1
        if moving_mae is not None:
            self.moving_mae_sum += moving_mae
            self.moving_count += 1

    def row(self, scene: str, method: str) -> dict[str, str]:
        return {
            "scene": scene,
            "method": method,
            "frames": str(self.frame_count),
            "mse": f"{self.mse_sum / max(self.frame_count, 1):.10f}",
            "psnr": f"{self.psnr_sum / max(self.frame_count, 1):.6f}",
            "ssim": f"{self.ssim_sum / max(self.frame_count, 1):.6f}",
            "static_mse": f"{self.static_mse_sum / max(self.static_count, 1):.10f}",
            "moving_mae": f"{self.moving_mae_sum / max(self.moving_count, 1):.6f}",
        }


@dataclass
class ScenePreviewStats:
    y_slice: slice
    x_slice: slice
    raw_low: float
    raw_high: float
    srgb_low: float
    srgb_high: float
    srgb_gains: np.ndarray


@dataclass
class SceneResult:
    scene: str
    raw_video: Path
    srgb_video: Path
    summary_rows: list[dict[str, str]]
    frame_metrics_path: Path
    summary_path: Path


def natural_scene_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("_", maxsplit=1)[-1]
    return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)


def list_scene_dirs(dataset_root: Path, selected_scenes: set[str] | None) -> list[Path]:
    scene_dirs = sorted((path for path in dataset_root.glob("scene_*") if path.is_dir()), key=natural_scene_key)
    if selected_scenes is None:
        return scene_dirs
    return [path for path in scene_dirs if path.name in selected_scenes]


def shard_paths(scene_dir: Path) -> list[Path]:
    paths = sorted(scene_dir.glob("shard_*.h5"), key=lambda path: int(path.stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"{scene_dir} 中没有 shard_*.h5")
    return paths


def iter_shard_frames(scene_dir: Path, noisy_channel: int) -> Iterable[tuple[int, dict[str, np.ndarray]]]:
    global_index = 0
    for shard_path in shard_paths(scene_dir):
        with h5py.File(shard_path, "r") as h5_file:
            frame_count = int(h5_file["clean"].shape[0])
            for local_index in range(frame_count):
                yield (
                    global_index,
                    {
                        "noisy": h5_file["noisy"][local_index, noisy_channel],
                        "2dnr": h5_file["2dnr"][local_index],
                        "dataset_3dnr": h5_file["3dnr"][local_index],
                        "clean": h5_file["clean"][local_index],
                    },
                )
                global_index += 1


def read_scene_metadata(scene_dir: Path) -> dict:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def choose_crop(scene_dir: Path, noisy_channel: int, crop_size: int, sample_stride: int) -> tuple[slice, slice]:
    previous: np.ndarray | None = None
    ys_all: list[np.ndarray] = []
    xs_all: list[np.ndarray] = []
    for frame_index, frame_data in iter_shard_frames(scene_dir, noisy_channel=noisy_channel):
        clean = frame_data["clean"].astype(np.float32)
        if previous is not None and frame_index % sample_stride == 0:
            diff = np.abs(clean - previous)
            threshold = max(50.0, float(np.percentile(diff, 98.5)))
            mask = diff > threshold
            ys, xs = np.where(mask)
            if len(ys) > 0:
                ys_all.append(ys)
                xs_all.append(xs)
        previous = clean

    metadata = read_scene_metadata(scene_dir)
    h, w = metadata["frame_shape"]
    if not ys_all:
        return crop_from_center((h, w), center_y=h // 2, center_x=w // 2, crop_size=crop_size)

    ys_cat = np.concatenate(ys_all)
    xs_cat = np.concatenate(xs_all)
    center_y = int(np.median(ys_cat))
    center_x = int(np.median(xs_cat))
    return crop_from_center((h, w), center_y=center_y, center_x=center_x, crop_size=crop_size)


def collect_preview_stats(
    scene_dir: Path,
    noisy_channel: int,
    y_slice: slice,
    x_slice: slice,
    args: argparse.Namespace,
) -> ScenePreviewStats:
    raw_samples = []
    srgb_samples = []
    for frame_index, frame_data in iter_shard_frames(scene_dir, noisy_channel=noisy_channel):
        if frame_index % args.preview_sample_stride != 0:
            continue
        clean_crop = frame_data["clean"][y_slice, x_slice]
        raw_samples.append(raw_to_luma_preview(clean_crop, black_level=args.black_level, white_level=args.white_level))
        srgb_samples.append(raw_to_srgb_half(clean_crop, black_level=args.black_level, white_level=args.white_level))

    raw_stack = np.stack(raw_samples, axis=0)
    srgb_stack = np.stack(srgb_samples, axis=0)
    gains = estimate_gray_world_gains(srgb_stack)
    balanced = srgb_stack * gains.reshape(1, 1, 1, 3)
    return ScenePreviewStats(
        y_slice=y_slice,
        x_slice=x_slice,
        raw_low=float(np.percentile(raw_stack, 1.0)),
        raw_high=float(np.percentile(raw_stack, 99.6)),
        srgb_low=float(np.percentile(balanced, 1.0)),
        srgb_high=float(np.percentile(balanced, 99.6)),
        srgb_gains=gains,
    )


def compute_scene_static_mask(scene_dir: Path, noisy_channel: int, static_threshold: float) -> np.ndarray:
    clean_min: np.ndarray | None = None
    clean_max: np.ndarray | None = None
    for _, frame_data in iter_shard_frames(scene_dir, noisy_channel=noisy_channel):
        clean = frame_data["clean"]
        if clean_min is None or clean_max is None:
            clean_min = clean.copy()
            clean_max = clean.copy()
        else:
            clean_min = np.minimum(clean_min, clean)
            clean_max = np.maximum(clean_max, clean)
    if clean_min is None or clean_max is None:
        raise ValueError(f"{scene_dir} 没有 clean 帧")
    return (clean_max.astype(np.float32) - clean_min.astype(np.float32)) < static_threshold


def mse_psnr(pred: np.ndarray, clean: np.ndarray, white_level: float) -> tuple[float, float]:
    diff = (pred.astype(np.float32) - clean.astype(np.float32)) / float(white_level)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 1e-12 else -10.0 * math.log10(mse)
    return mse, psnr


def ssim_luma(pred: np.ndarray, clean: np.ndarray, white_level: float, stride: int) -> float:
    pred_luma = raw_to_luma_preview(pred, black_level=0.0, white_level=white_level)[::stride, ::stride] / float(
        white_level
    )
    clean_luma = raw_to_luma_preview(clean, black_level=0.0, white_level=white_level)[::stride, ::stride] / float(
        white_level
    )
    radius = 3
    mu_x = box_blur_2d(pred_luma, radius=radius)
    mu_y = box_blur_2d(clean_luma, radius=radius)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = box_blur_2d(pred_luma * pred_luma, radius=radius) - mu_x2
    sigma_y2 = box_blur_2d(clean_luma * clean_luma, radius=radius) - mu_y2
    sigma_xy = box_blur_2d(pred_luma * clean_luma, radius=radius) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return float(np.mean(ssim_map))


def static_moving_metrics(
    pred: np.ndarray,
    clean: np.ndarray,
    previous_clean: np.ndarray | None,
    scene_static_mask: np.ndarray,
    static_threshold: float,
    moving_threshold: float,
) -> tuple[float | None, float | None]:
    diff = pred.astype(np.float32) - clean.astype(np.float32)
    static_mse = (
        float(np.mean(diff[scene_static_mask] * diff[scene_static_mask])) if np.any(scene_static_mask) else None
    )
    if previous_clean is None:
        return static_mse, None
    motion = np.abs(clean.astype(np.float32) - previous_clean.astype(np.float32))
    moving_mask = motion > moving_threshold
    moving_mae = float(np.mean(np.abs(diff[moving_mask]))) if np.any(moving_mask) else None
    return static_mse, moving_mae


def append_video_frames(
    raw_frames: list[np.ndarray],
    srgb_frames: list[np.ndarray],
    frame_data: dict[str, np.ndarray],
    ghost_raw: np.ndarray,
    frame_index: int,
    preview: ScenePreviewStats,
    args: argparse.Namespace,
) -> None:
    if frame_index % args.video_stride != 0:
        return

    methods = {
        "noisy": frame_data["noisy"],
        "2dnr": frame_data["2dnr"],
        "dataset_3dnr": frame_data["dataset_3dnr"],
        "ghost3dnr": ghost_raw,
        "clean": frame_data["clean"],
    }
    names = ["noisy", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"]
    raw_images = [
        normalize_for_preview(
            raw_to_luma_preview(
                methods[name][preview.y_slice, preview.x_slice],
                black_level=args.black_level,
                white_level=args.white_level,
            ),
            low=preview.raw_low,
            high=preview.raw_high,
        )
        for name in names
    ]
    srgb_images = [
        rgb_to_uint8_preview(
            raw_to_srgb_half(
                methods[name][preview.y_slice, preview.x_slice],
                black_level=args.black_level,
                white_level=args.white_level,
            ),
            gains=preview.srgb_gains,
            low=preview.srgb_low,
            high=preview.srgb_high,
        )
        for name in names
    ]
    titles = [f"noisy f{frame_index:04d}", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"]
    raw_frames.append(make_panel(raw_images, titles))
    srgb_frames.append(make_panel(srgb_images, titles))


def process_scene(scene_dir: Path, output_root: Path, args: argparse.Namespace) -> SceneResult:
    scene_out = output_root / scene_dir.name
    scene_out.mkdir(parents=True, exist_ok=True)
    y_slice, x_slice = choose_crop(
        scene_dir,
        noisy_channel=args.noisy_channel,
        crop_size=args.crop_size,
        sample_stride=args.crop_sample_stride,
    )
    preview = collect_preview_stats(
        scene_dir,
        noisy_channel=args.noisy_channel,
        y_slice=y_slice,
        x_slice=x_slice,
        args=args,
    )
    scene_static_mask = compute_scene_static_mask(
        scene_dir,
        noisy_channel=args.noisy_channel,
        static_threshold=args.scene_static_threshold,
    )

    config = GhostTrail3DNRConfig(
        black_level=args.black_level,
        white_level=args.white_level,
        sigma01=args.sigma01,
        static_history_weight=args.static_history_weight,
        motion_history_weight=args.motion_history_weight,
        motion_current_floor=args.motion_current_floor,
        trail_strength=args.trail_strength,
        trail_decay=args.trail_decay,
        echo_delay=args.echo_delay,
        echo_taps=args.echo_taps,
        background_history_weight=args.background_history_weight,
        motion_threshold_scale=args.motion_threshold_scale,
        motion_softness=args.motion_softness,
        min_motion_threshold=args.min_motion_threshold,
        motion_map_blur_radius=args.motion_map_blur_radius,
    )
    processor = GhostTrail3DNRProcessor(config)
    metric_states: dict[str, MetricState] = {method: MetricState() for method in METHODS}
    frame_rows: list[dict[str, str]] = []
    raw_video_frames: list[np.ndarray] = []
    srgb_video_frames: list[np.ndarray] = []
    previous_clean: np.ndarray | None = None

    for frame_index, frame_data in iter_shard_frames(scene_dir, noisy_channel=args.noisy_channel):
        ghost01, _ = processor.process(frame_data["noisy"])
        ghost_raw = restore_raw_dtype(
            ghost01,
            dtype=frame_data["clean"].dtype,
            black_level=args.black_level,
            white_level=args.white_level,
        )
        predictions = {
            "noisy": frame_data["noisy"],
            "2dnr": frame_data["2dnr"],
            "dataset_3dnr": frame_data["dataset_3dnr"],
            "ghost3dnr": ghost_raw,
        }
        for method, pred in predictions.items():
            mse, psnr = mse_psnr(pred, frame_data["clean"], white_level=args.white_level)
            ssim = ssim_luma(pred, frame_data["clean"], white_level=args.white_level, stride=args.ssim_stride)
            static_mse, moving_mae = static_moving_metrics(
                pred,
                frame_data["clean"],
                previous_clean=previous_clean,
                scene_static_mask=scene_static_mask,
                static_threshold=args.static_threshold,
                moving_threshold=args.moving_threshold,
            )
            metric_states[method].update(mse, psnr, ssim, static_mse, moving_mae)
            frame_rows.append(
                {
                    "scene": scene_dir.name,
                    "frame": str(frame_index),
                    "method": method,
                    "mse": f"{mse:.10f}",
                    "psnr": f"{psnr:.6f}",
                    "ssim": f"{ssim:.6f}",
                    "static_mse": "" if static_mse is None else f"{static_mse:.6f}",
                    "moving_mae": "" if moving_mae is None else f"{moving_mae:.6f}",
                }
            )
        append_video_frames(
            raw_video_frames,
            srgb_video_frames,
            frame_data=frame_data,
            ghost_raw=ghost_raw,
            frame_index=frame_index,
            preview=preview,
            args=args,
        )
        previous_clean = frame_data["clean"].copy()

    frame_metrics_path = scene_out / "metrics_frame.csv"
    write_csv(frame_metrics_path, frame_rows)
    summary_rows = [metric_states[method].row(scene_dir.name, method) for method in METHODS]
    summary_path = scene_out / "metrics_summary.csv"
    write_csv(summary_path, summary_rows)
    raw_video_path = scene_out / f"{scene_dir.name}_raw_video.gif"
    srgb_video_path = scene_out / f"{scene_dir.name}_srgb_video.gif"
    save_gif(raw_video_path, raw_video_frames, fps=args.video_fps)
    save_gif(srgb_video_path, srgb_video_frames, fps=args.video_fps)
    return SceneResult(
        scene=scene_dir.name,
        raw_video=raw_video_path,
        srgb_video=srgb_video_path,
        summary_rows=summary_rows,
        frame_metrics_path=frame_metrics_path,
        summary_path=summary_path,
    )


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_batch_summary(output_root: Path, scene_results: Sequence[SceneResult]) -> Path:
    by_method: dict[str, MetricState] = defaultdict(MetricState)
    all_rows: list[dict[str, str]] = []
    for result in scene_results:
        all_rows.extend(result.summary_rows)
        for row in result.summary_rows:
            method = row["method"]
            frames = int(row["frames"])
            state = by_method[method]
            state.frame_count += frames
            state.mse_sum += float(row["mse"]) * frames
            state.psnr_sum += float(row["psnr"]) * frames
            state.ssim_sum += float(row["ssim"]) * frames
            state.static_mse_sum += float(row["static_mse"]) * frames
            state.static_count += frames
            state.moving_mae_sum += float(row["moving_mae"]) * frames
            state.moving_count += frames

    all_rows.extend(by_method[method].row("ALL", method) for method in METHODS if method in by_method)
    summary_path = output_root / "metrics_summary_all.csv"
    write_csv(summary_path, all_rows)
    return summary_path


def parse_scene_filter(scene_args: Sequence[str]) -> set[str] | None:
    if not scene_args:
        return None
    selected = set()
    for item in scene_args:
        selected.add(item if item.startswith("scene_") else f"scene_{item}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量评估 H5 数据集上的拖影型 3DNR，并导出 RAW/sRGB 视频")
    parser.add_argument("--dataset-root", required=True, help="包含 scene_*/shard_*.h5 的 H5 数据集根目录")
    parser.add_argument("--output-root", default="dataset_results")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--noisy-channel", type=int, default=1)
    parser.add_argument("--black-level", type=float, default=0.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--sigma01", type=float, default=0.012)
    parser.add_argument("--static-history-weight", type=float, default=0.88)
    parser.add_argument("--motion-history-weight", type=float, default=0.94)
    parser.add_argument("--motion-current-floor", type=float, default=0.85)
    parser.add_argument("--trail-strength", type=float, default=1.35)
    parser.add_argument("--trail-decay", type=float, default=0.72)
    parser.add_argument("--echo-delay", type=int, default=2)
    parser.add_argument("--echo-taps", type=int, default=4)
    parser.add_argument("--background-history-weight", type=float, default=0.92)
    parser.add_argument("--motion-threshold-scale", type=float, default=1.0)
    parser.add_argument("--motion-softness", type=float, default=0.7)
    parser.add_argument("--min-motion-threshold", type=float, default=0.008)
    parser.add_argument("--motion-map-blur-radius", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--crop-sample-stride", type=int, default=4)
    parser.add_argument("--preview-sample-stride", type=int, default=8)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--ssim-stride", type=int, default=4)
    parser.add_argument("--scene-static-threshold", type=float, default=40.0)
    parser.add_argument("--static-threshold", type=float, default=40.0)
    parser.add_argument("--moving-threshold", type=float, default=80.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dirs = list_scene_dirs(dataset_root, parse_scene_filter(args.scene))
    if not scene_dirs:
        raise FileNotFoundError(f"没有找到可处理的 scene: {dataset_root}")

    results = []
    for scene_dir in scene_dirs:
        print(f"\n处理 {scene_dir.name} ...", flush=True)
        result = process_scene(scene_dir, output_root=output_root, args=args)
        results.append(result)
        print(f"  指标: {result.summary_path}", flush=True)
        print(f"  RAW 视频: {result.raw_video}", flush=True)
        print(f"  sRGB 视频: {result.srgb_video}", flush=True)

    summary_path = write_batch_summary(output_root, results)
    print(f"\n全数据汇总: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

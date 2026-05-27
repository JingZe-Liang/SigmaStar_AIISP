from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .ghost_trail_3dnr import GhostTrail3DNRConfig, denoise_sequence_array, raw_luma_proxy
except ImportError:
    from ghost_trail_3dnr import GhostTrail3DNRConfig, denoise_sequence_array, raw_luma_proxy


def normalize_for_preview(frame: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    frame32 = frame.astype(np.float32)
    lo = float(np.percentile(frame32, 1) if low is None else low)
    hi = float(np.percentile(frame32, 99.5) if high is None else high)
    if hi <= lo:
        hi = lo + 1.0
    preview = np.clip((frame32 - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(preview * 255.0).astype(np.uint8)


def normalize_raw01(frame: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    denom = float(white_level - black_level)
    if denom <= 0.0:
        raise ValueError("white_level 必须大于 black_level")
    return np.clip((frame.astype(np.float32) - float(black_level)) / denom, 0.0, 1.0)


def raw_to_luma_raw(frame: np.ndarray, white_level: float) -> np.ndarray:
    frame01 = np.clip(frame.astype(np.float32) / float(white_level), 0.0, 1.0)
    return raw_luma_proxy(frame01) * float(white_level)


def raw_to_luma_preview(frame: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    frame01 = normalize_raw01(frame, black_level=black_level, white_level=white_level)
    return raw_luma_proxy(frame01) * float(white_level - black_level)


def raw_to_srgb_half(frame: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    raw01 = normalize_raw01(frame, black_level=black_level, white_level=white_level)
    h_even = raw01.shape[0] - raw01.shape[0] % 2
    w_even = raw01.shape[1] - raw01.shape[1] % 2
    raw01 = raw01[:h_even, :w_even]

    red = raw01[0::2, 0::2]
    green = 0.5 * (raw01[0::2, 1::2] + raw01[1::2, 0::2])
    blue = raw01[1::2, 1::2]
    return np.stack([red, green, blue], axis=2).astype(np.float32)


def estimate_gray_world_gains(rgb_stack: np.ndarray) -> np.ndarray:
    channel_mean = np.mean(rgb_stack, axis=(0, 1, 2))
    target = float(np.mean(channel_mean))
    gains = target / np.maximum(channel_mean, 1e-6)
    return np.clip(gains, 0.25, 4.0).astype(np.float32)


def rgb_to_uint8_preview(
    rgb: np.ndarray,
    gains: np.ndarray,
    low: float,
    high: float,
    gamma: float = 1.0 / 2.2,
) -> np.ndarray:
    balanced = np.clip(rgb.astype(np.float32) * gains.reshape(1, 1, 3), 0.0, 1.0)
    normalized = np.clip((balanced - low) / max(high - low, 1e-6), 0.0, 1.0)
    srgb = normalized**gamma
    return np.rint(srgb * 255.0).astype(np.uint8)


def normalize_signed_residual(residual: np.ndarray, max_abs: float | None = None) -> np.ndarray:
    residual32 = residual.astype(np.float32)
    scale = float(np.percentile(np.abs(residual32), 99.0) if max_abs is None else max_abs)
    if scale <= 1e-6:
        scale = 1.0

    normalized = np.clip(residual32 / scale, -1.0, 1.0)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 2] = np.rint(np.clip(-normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 1] = np.rint((1.0 - np.abs(normalized)) * 80.0).astype(np.uint8)
    return rgb


def read_h5_window(h5_path: str | Path, start: int, count: int, noisy_channel: int) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as h5_file:
        end = min(start + count, h5_file["clean"].shape[0])
        if start < 0 or start >= end:
            raise ValueError(f"非法帧范围: start={start}, count={count}")

        data = {
            "clean": h5_file["clean"][start:end],
            "2dnr": h5_file["2dnr"][start:end],
            "3dnr": h5_file["3dnr"][start:end],
            "noisy": h5_file["noisy"][start:end, noisy_channel],
        }
    return data


def build_motion_masks(clean: np.ndarray, frame_index: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    current = clean[frame_index].astype(np.float32)
    previous = clean[frame_index - 1].astype(np.float32)
    diff = np.abs(current - previous)
    moving = diff > threshold

    older = clean[max(0, frame_index - 4) : frame_index].astype(np.float32)
    history_max = np.max(older, axis=0)
    history_min = np.min(older, axis=0)
    departed = (np.abs(history_max - current) > threshold) | (np.abs(history_min - current) > threshold)
    departed &= ~moving
    return moving, departed


def static_mask_from_clean(clean: np.ndarray, threshold: float) -> np.ndarray:
    diffs = np.abs(np.diff(clean.astype(np.float32), axis=0))
    return np.max(diffs, axis=0) < threshold


def mse_on_mask(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    diff = pred.astype(np.float32)[mask] - target.astype(np.float32)[mask]
    return float(np.mean(diff * diff))


def mean_abs_on_mask(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred.astype(np.float32)[mask] - target.astype(np.float32)[mask])))


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def draw_title(image: np.ndarray, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    rgb = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image
    panel = Image.fromarray(rgb)
    draw = ImageDraw.Draw(panel)
    draw.text((13, 13), title, fill=(0, 0, 0))
    draw.text((12, 12), title, fill=(255, 255, 255))
    return np.asarray(panel)


def make_panel(images: Sequence[np.ndarray], titles: Sequence[str]) -> np.ndarray:
    if len(images) != len(titles):
        raise ValueError("images 和 titles 数量必须一致")

    panels = [draw_title(image, title) for image, title in zip(images, titles, strict=True)]
    return np.concatenate(panels, axis=1)


def save_gif(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    from PIL import Image

    if not frames:
        raise ValueError("至少需要 1 帧才能保存视频")

    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / fps)))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def crop_from_center(shape: tuple[int, int], center_y: int, center_x: int, crop_size: int) -> tuple[slice, slice]:
    half = crop_size // 2
    h, w = shape
    y0 = min(max(0, center_y - half), max(0, h - crop_size))
    x0 = min(max(0, center_x - half), max(0, w - crop_size))
    return slice(y0, y0 + crop_size), slice(x0, x0 + crop_size)


def crop_around_motion(clean: np.ndarray, frame_index: int, crop_size: int) -> tuple[slice, slice]:
    moving, departed = build_motion_masks(clean, frame_index=frame_index, threshold=80.0)
    mask = moving | departed
    if not np.any(mask):
        h, w = clean.shape[1:]
        y = h // 2
        x = w // 2
    else:
        ys, xs = np.where(mask)
        y = int(np.median(ys))
        x = int(np.median(xs))

    return crop_from_center(clean.shape[1:], center_y=y, center_x=x, crop_size=crop_size)


def build_method_frames(data: dict[str, np.ndarray], ghost_raw: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "noisy": data["noisy"],
        "2dnr": data["2dnr"],
        "dataset_3dnr": data["3dnr"],
        "ghost3dnr": ghost_raw,
        "clean": data["clean"],
    }


def make_raw_video_frames(
    methods: dict[str, np.ndarray],
    y_slice: slice,
    x_slice: slice,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    preview_stacks = {
        name: np.stack(
            [
                raw_to_luma_preview(frame[y_slice, x_slice], black_level=args.black_level, white_level=args.white_level)
                for frame in frames
            ],
            axis=0,
        )
        for name, frames in methods.items()
    }
    reference = np.concatenate([preview_stacks["clean"].reshape(-1), preview_stacks["ghost3dnr"].reshape(-1)])
    low = float(np.percentile(reference, 1.0))
    high = float(np.percentile(reference, 99.5))

    video_frames = []
    for frame_index in range(next(iter(methods.values())).shape[0]):
        global_frame = args.start + frame_index
        images = [
            normalize_for_preview(preview_stacks[name][frame_index], low=low, high=high)
            for name in ["noisy", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"]
        ]
        titles = [f"noisy f{global_frame:04d}", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"]
        video_frames.append(make_panel(images, titles))
    return video_frames


def make_srgb_video_frames(
    methods: dict[str, np.ndarray],
    y_slice: slice,
    x_slice: slice,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    rgb_stacks = {
        name: np.stack(
            [
                raw_to_srgb_half(frame[y_slice, x_slice], black_level=args.black_level, white_level=args.white_level)
                for frame in frames
            ],
            axis=0,
        )
        for name, frames in methods.items()
    }
    gains = estimate_gray_world_gains(rgb_stacks["clean"])
    reference = np.concatenate(
        [
            (rgb_stacks["clean"] * gains.reshape(1, 1, 1, 3)).reshape(-1),
            (rgb_stacks["ghost3dnr"] * gains.reshape(1, 1, 1, 3)).reshape(-1),
        ]
    )
    low = float(np.percentile(reference, 1.0))
    high = float(np.percentile(reference, 99.5))

    video_frames = []
    for frame_index in range(next(iter(methods.values())).shape[0]):
        global_frame = args.start + frame_index
        images = [
            rgb_to_uint8_preview(rgb_stacks[name][frame_index], gains=gains, low=low, high=high)
            for name in ["noisy", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"]
        ]
        titles = [f"noisy f{global_frame:04d}", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"]
        video_frames.append(make_panel(images, titles))
    return video_frames


def write_visual_videos(
    h5_path: Path,
    data: dict[str, np.ndarray],
    ghost_raw: np.ndarray,
    y_slice: slice,
    x_slice: slice,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    methods = build_method_frames(data, ghost_raw)
    raw_frames = make_raw_video_frames(methods, y_slice=y_slice, x_slice=x_slice, args=args)
    srgb_frames = make_srgb_video_frames(methods, y_slice=y_slice, x_slice=x_slice, args=args)

    prefix = f"{h5_path.parent.name}_{h5_path.stem}_frames{args.start:04d}_{args.start + len(ghost_raw) - 1:04d}"
    raw_path = out_dir / f"{prefix}_raw_video.gif"
    srgb_path = out_dir / f"{prefix}_srgb_video.gif"
    save_gif(raw_path, raw_frames, fps=args.video_fps)
    save_gif(srgb_path, srgb_frames, fps=args.video_fps)
    return raw_path, srgb_path


def print_metrics(data: dict[str, np.ndarray], ghost3dnr: np.ndarray, frame_index: int) -> None:
    clean = data["clean"].astype(np.float32)
    static = static_mask_from_clean(clean, threshold=40.0)
    moving, departed = build_motion_masks(clean, frame_index=frame_index, threshold=80.0)

    methods = {
        "noisy": data["noisy"].astype(np.float32),
        "2dnr": data["2dnr"].astype(np.float32),
        "dataset_3dnr": data["3dnr"].astype(np.float32),
        "ghost3dnr": ghost3dnr.astype(np.float32),
    }
    print(f"评估窗口内静止 mask 占比: {float(np.mean(static)):.4f}")
    print(f"评估帧 moving mask 占比: {float(np.mean(moving)):.4f}")
    print(f"评估帧 departed/残影候选 mask 占比: {float(np.mean(departed)):.4f}")
    print("\nmethod, static_mse, moving_mae, departed_mae")
    for name, frames in methods.items():
        print(
            f"{name}, "
            f"{mse_on_mask(frames[frame_index], clean[frame_index], static):.4f}, "
            f"{mean_abs_on_mask(frames[frame_index], clean[frame_index], moving):.4f}, "
            f"{mean_abs_on_mask(frames[frame_index], clean[frame_index], departed):.4f}"
        )


def run_demo(args: argparse.Namespace) -> None:
    h5_path = Path(args.h5_path)
    data = read_h5_window(h5_path, start=args.start, count=args.count, noisy_channel=args.noisy_channel)

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
    ghost01, stats = denoise_sequence_array(data["noisy"], config=config)
    ghost_raw = np.clip(np.rint(ghost01 * (args.white_level - args.black_level) + args.black_level), 0, 65535).astype(
        np.uint16
    )

    frame_index = min(args.eval_index, ghost_raw.shape[0] - 1)
    print_metrics(data, ghost_raw, frame_index=frame_index)
    print("\nghost3dnr stats:")
    for item in stats:
        print(
            f"frame={args.start + item.frame_index:04d}, "
            f"sigma01={item.sigma01:.6f}, "
            f"motion={item.mean_motion_score:.4f}, "
            f"history_weight={item.mean_history_weight:.4f}"
        )

    if args.crop_center_y is None or args.crop_center_x is None:
        y_slice, x_slice = crop_around_motion(data["clean"], frame_index=frame_index, crop_size=args.crop_size)
    else:
        y_slice, x_slice = crop_from_center(
            data["clean"].shape[1:],
            center_y=args.crop_center_y,
            center_x=args.crop_center_x,
            crop_size=args.crop_size,
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    luma_images = {
        "noisy": raw_to_luma_raw(data["noisy"][frame_index], white_level=args.white_level),
        "2dnr": raw_to_luma_raw(data["2dnr"][frame_index], white_level=args.white_level),
        "dataset_3dnr": raw_to_luma_raw(data["3dnr"][frame_index], white_level=args.white_level),
        "ghost3dnr": raw_to_luma_raw(ghost_raw[frame_index], white_level=args.white_level),
        "clean": raw_to_luma_raw(data["clean"][frame_index], white_level=args.white_level),
    }
    crop_values = [luma_images["clean"][y_slice, x_slice], luma_images["ghost3dnr"][y_slice, x_slice]]
    low = float(np.percentile(np.stack(crop_values), 1))
    high = float(np.percentile(np.stack(crop_values), 99.5))

    panel = make_panel(
        images=[
            normalize_for_preview(luma_images["noisy"][y_slice, x_slice], low=low, high=high),
            normalize_for_preview(luma_images["2dnr"][y_slice, x_slice], low=low, high=high),
            normalize_for_preview(luma_images["dataset_3dnr"][y_slice, x_slice], low=low, high=high),
            normalize_for_preview(luma_images["ghost3dnr"][y_slice, x_slice], low=low, high=high),
            normalize_for_preview(luma_images["clean"][y_slice, x_slice], low=low, high=high),
        ],
        titles=["noisy", "2dnr", "dataset_3dnr", "ghost3dnr", "clean"],
    )
    save_png(out_dir / f"{h5_path.parent.name}_{h5_path.stem}_frame{args.start + frame_index:04d}_panel.png", panel)

    residual = luma_images["ghost3dnr"][y_slice, x_slice] - luma_images["clean"][y_slice, x_slice]
    save_png(
        out_dir / f"{h5_path.parent.name}_{h5_path.stem}_frame{args.start + frame_index:04d}_ghost_residual.png",
        normalize_signed_residual(residual),
    )

    save_png(
        out_dir / f"{h5_path.parent.name}_{h5_path.stem}_frame{args.start + frame_index:04d}_ghost3dnr.png",
        normalize_for_preview(luma_images["ghost3dnr"]),
    )

    if not args.skip_video:
        raw_video_path, srgb_video_path = write_visual_videos(
            h5_path=h5_path,
            data=data,
            ghost_raw=ghost_raw,
            y_slice=y_slice,
            x_slice=x_slice,
            out_dir=out_dir,
            args=args,
        )
        print("\n视频输出:")
        print(f"RAW  video: {raw_video_path}")
        print(f"sRGB video: {srgb_video_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 H5 数据集上验证拖影型 RAW 3DNR")
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--eval-index", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-center-y", type=int, default=None)
    parser.add_argument("--crop-center-x", type=int, default=None)
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
    parser.add_argument("--min-motion-threshold", type=float, default=0.004)
    parser.add_argument("--motion-map-blur-radius", type=int, default=2)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--skip-video", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    run_demo(parser.parse_args(argv))


if __name__ == "__main__":
    main()

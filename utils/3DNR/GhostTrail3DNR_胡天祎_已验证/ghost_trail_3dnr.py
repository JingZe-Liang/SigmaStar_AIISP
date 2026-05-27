from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GhostTrail3DNRConfig:
    black_level: float = 0.0
    white_level: float = 4095.0
    sigma01: float | None = None
    static_history_weight: float = 0.88
    motion_history_weight: float = 0.94
    motion_current_floor: float = 0.85
    trail_strength: float = 1.35
    trail_decay: float = 0.72
    echo_delay: int = 2
    echo_taps: int = 4
    background_history_weight: float = 0.92
    motion_threshold_scale: float = 1.0
    motion_softness: float = 0.7
    min_motion_threshold: float = 0.008
    motion_map_blur_radius: int = 2


@dataclass(frozen=True)
class Frame3DNRStats:
    frame_index: int
    sigma01: float
    mean_motion_score: float
    mean_history_weight: float


class GhostTrail3DNRProcessor:
    def __init__(self, config: GhostTrail3DNRConfig | None = None) -> None:
        self.config = config or GhostTrail3DNRConfig()
        _validate_config(self.config)
        self.frame_index = 0
        self.history: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self.frame_buffer: list[np.ndarray] = []

    def process(self, raw_frame: np.ndarray) -> tuple[np.ndarray, Frame3DNRStats]:
        current = normalize_raw(
            raw_frame,
            black_level=self.config.black_level,
            white_level=self.config.white_level,
        )
        sigma01 = float(self.config.sigma01 if self.config.sigma01 is not None else estimate_sigma01(current))

        if self.history is None or self.background is None:
            self.history = current.copy()
            self.background = current.copy()
            self._push_frame(current)
            stats = Frame3DNRStats(
                frame_index=self.frame_index,
                sigma01=sigma01,
                mean_motion_score=0.0,
                mean_history_weight=0.0,
            )
            self.frame_index += 1
            return current.copy(), stats

        motion_score = build_motion_score(current, self.history, sigma01=sigma01, config=self.config)
        weight = _history_weight_map(motion_score, config=self.config)
        blend_weight = _expand_weight(weight, current)
        base = blend_weight * self.history + (1.0 - blend_weight) * current
        base = np.clip(base, 0.0, 1.0).astype(np.float32)

        ghost_residual = _build_echo_residual_from_history(
            past_frames=self.frame_buffer,
            current=current,
            background=self.background,
            sigma01=sigma01,
            config=self.config,
        )
        output = base + np.float32(self.config.trail_strength) * ghost_residual
        output = np.clip(output, 0.0, 1.0).astype(np.float32)

        bg_motion = build_motion_score(current, self.background, sigma01=sigma01, config=self.config)
        bg_keep = (
            np.float32(self.config.background_history_weight)
            + (1.0 - np.float32(self.config.background_history_weight)) * bg_motion
        )
        bg_blend = _expand_weight(np.clip(bg_keep, 0.0, 0.999).astype(np.float32), current)
        self.background = bg_blend * self.background + (1.0 - bg_blend) * current
        self.background = np.clip(self.background, 0.0, 1.0).astype(np.float32)
        self.history = base
        self._push_frame(current)

        stats = Frame3DNRStats(
            frame_index=self.frame_index,
            sigma01=sigma01,
            mean_motion_score=float(np.mean(motion_score)),
            mean_history_weight=float(np.mean(weight)),
        )
        self.frame_index += 1
        return output, stats

    def _push_frame(self, current: np.ndarray) -> None:
        self.frame_buffer.append(current.copy())
        max_len = self.config.echo_delay * self.config.echo_taps
        if len(self.frame_buffer) > max_len:
            del self.frame_buffer[: len(self.frame_buffer) - max_len]


def _validate_config(config: GhostTrail3DNRConfig) -> None:
    if config.white_level <= config.black_level:
        raise ValueError("white_level 必须大于 black_level")
    if not 0.0 <= config.static_history_weight < 1.0:
        raise ValueError("static_history_weight 必须在 [0, 1) 内")
    if not 0.0 <= config.motion_history_weight < 1.0:
        raise ValueError("motion_history_weight 必须在 [0, 1) 内")
    if not 0.0 <= config.motion_current_floor <= 1.0:
        raise ValueError("motion_current_floor 必须在 [0, 1] 内")
    if not 0.0 <= config.trail_strength <= 3.0:
        raise ValueError("trail_strength 必须在 [0, 3] 内")
    if not 0.0 <= config.trail_decay < 1.0:
        raise ValueError("trail_decay 必须在 [0, 1) 内")
    if config.echo_delay < 1:
        raise ValueError("echo_delay 必须大于等于 1")
    if config.echo_taps < 1:
        raise ValueError("echo_taps 必须大于等于 1")
    if not 0.0 <= config.background_history_weight < 1.0:
        raise ValueError("background_history_weight 必须在 [0, 1) 内")
    if config.motion_threshold_scale <= 0.0:
        raise ValueError("motion_threshold_scale 必须大于 0")
    if config.motion_softness <= 0.0:
        raise ValueError("motion_softness 必须大于 0")
    if config.min_motion_threshold <= 0.0:
        raise ValueError("min_motion_threshold 必须大于 0")
    if config.motion_map_blur_radius < 0:
        raise ValueError("motion_map_blur_radius 不能小于 0")
    if config.sigma01 is not None and config.sigma01 <= 0.0:
        raise ValueError("sigma01 必须大于 0")


def normalize_raw(raw: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    denom = float(white_level - black_level)
    if denom <= 0.0:
        raise ValueError("white_level 必须大于 black_level")
    raw01 = (raw.astype(np.float32) - float(black_level)) / denom
    return np.clip(raw01, 0.0, 1.0).astype(np.float32)


def restore_raw_dtype(raw01: np.ndarray, dtype: np.dtype, black_level: float, white_level: float) -> np.ndarray:
    raw = raw01.astype(np.float32) * float(white_level - black_level) + float(black_level)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(raw), info.min, info.max).astype(dtype)
    return raw.astype(dtype)


def _box_blur_axis(data: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return data.astype(np.float32, copy=False)

    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(data, pad_width=pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis, dtype=np.float64)

    zero_shape = list(cumsum.shape)
    zero_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), cumsum], axis=axis)

    window = 2 * radius + 1
    start_slice = [slice(None)] * cumsum.ndim
    end_slice = [slice(None)] * cumsum.ndim
    start_slice[axis] = slice(0, -window)
    end_slice[axis] = slice(window, None)
    blurred = (cumsum[tuple(end_slice)] - cumsum[tuple(start_slice)]) / float(window)
    return blurred.astype(np.float32)


def box_blur_2d(data: np.ndarray, radius: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"box_blur_2d 只支持二维数组，收到 shape={data.shape}")
    blurred = _box_blur_axis(data.astype(np.float32, copy=False), radius=radius, axis=0)
    return _box_blur_axis(blurred, radius=radius, axis=1)


def raw_luma_proxy(frame01: np.ndarray) -> np.ndarray:
    if frame01.ndim == 3:
        return np.mean(frame01.astype(np.float32), axis=2)
    if frame01.ndim != 2:
        raise ValueError(f"单帧必须是 [H,W] 或 [H,W,C]，收到 shape={frame01.shape}")

    h, w = frame01.shape
    pad_h = h % 2
    pad_w = w % 2
    padded = np.pad(frame01.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="edge")
    cell_mean = (padded[0::2, 0::2] + padded[0::2, 1::2] + padded[1::2, 0::2] + padded[1::2, 1::2]) * 0.25
    proxy = np.repeat(np.repeat(cell_mean, 2, axis=0), 2, axis=1)
    return proxy[:h, :w].astype(np.float32)


def estimate_sigma01(frame01: np.ndarray) -> float:
    proxy = raw_luma_proxy(frame01)
    smooth = box_blur_2d(proxy, radius=1)
    residual = proxy - smooth
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    sigma01 = 1.4826 * mad
    return float(np.clip(sigma01, 1e-4, 0.25))


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x01 = np.clip(x, 0.0, 1.0).astype(np.float32)
    return x01 * x01 * (3.0 - 2.0 * x01)


def build_motion_score(
    current01: np.ndarray,
    history01: np.ndarray,
    sigma01: float,
    config: GhostTrail3DNRConfig | None = None,
) -> np.ndarray:
    cfg = config or GhostTrail3DNRConfig()
    _validate_config(cfg)

    current_proxy = raw_luma_proxy(current01)
    history_proxy = raw_luma_proxy(history01)
    diff = np.abs(current_proxy - history_proxy).astype(np.float32)

    if cfg.motion_map_blur_radius > 0:
        diff = box_blur_2d(diff, radius=cfg.motion_map_blur_radius)

    threshold = max(cfg.min_motion_threshold, cfg.motion_threshold_scale * float(sigma01))
    width = max(cfg.min_motion_threshold, cfg.motion_softness * threshold)
    score = _smoothstep01((diff - threshold) / width)

    if cfg.motion_map_blur_radius > 0:
        score = box_blur_2d(score, radius=cfg.motion_map_blur_radius)

    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _as_frame_stack(frames: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    stack = np.asarray(frames)
    if stack.ndim not in (3, 4):
        raise ValueError(f"frames 必须是 [T,H,W] 或 [T,H,W,C]，收到 shape={stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("frames 至少需要 1 帧")
    return stack


def _history_weight_map(motion_score: np.ndarray, config: GhostTrail3DNRConfig) -> np.ndarray:
    static_w = np.float32(config.static_history_weight)
    motion_w = np.float32(config.motion_history_weight)
    weight = static_w * (1.0 - motion_score) + motion_w * motion_score
    current_floor = np.float32(config.motion_current_floor)
    max_motion_weight = 1.0 - current_floor * motion_score
    weight = np.minimum(weight, max_motion_weight)
    return np.clip(weight, 0.0, 0.999).astype(np.float32)


def _expand_weight(weight: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return weight[..., None] if frame.ndim == 3 else weight


def _activity_score(activity: np.ndarray, sigma01: float, config: GhostTrail3DNRConfig) -> np.ndarray:
    threshold = max(config.min_motion_threshold, config.motion_threshold_scale * float(sigma01))
    width = max(config.min_motion_threshold, config.motion_softness * threshold)
    return _smoothstep01((activity.astype(np.float32) - threshold) / width)


def _departed_trail_score(
    current: np.ndarray,
    trail: np.ndarray,
    background: np.ndarray,
    sigma01: float,
    config: GhostTrail3DNRConfig,
) -> np.ndarray:
    current_activity = np.abs(raw_luma_proxy(current) - raw_luma_proxy(background))
    trail_activity = np.abs(raw_luma_proxy(trail) - raw_luma_proxy(background))
    current_score = _activity_score(current_activity, sigma01=sigma01, config=config)
    trail_score = _activity_score(trail_activity, sigma01=sigma01, config=config)
    score = trail_score * (1.0 - current_score)

    if config.motion_map_blur_radius > 0:
        score = box_blur_2d(score, radius=config.motion_map_blur_radius)

    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _build_echo_residual(
    frames01: np.ndarray,
    current: np.ndarray,
    background: np.ndarray,
    frame_index: int,
    sigma01: float,
    config: GhostTrail3DNRConfig,
) -> np.ndarray:
    current_activity = np.abs(raw_luma_proxy(current) - raw_luma_proxy(background))
    current_score = _activity_score(current_activity, sigma01=sigma01, config=config)
    clear_current = (1.0 - current_score) * (1.0 - current_score)

    residual = np.zeros_like(current, dtype=np.float32)
    for tap_index in range(config.echo_taps):
        echo_index = frame_index - (tap_index + 1) * config.echo_delay
        if echo_index < 0:
            break

        echo = frames01[echo_index]
        echo_activity = np.abs(raw_luma_proxy(echo) - raw_luma_proxy(background))
        echo_score = _activity_score(echo_activity, sigma01=sigma01, config=config)
        echo_score = echo_score * clear_current * np.float32(config.trail_decay**tap_index)

        if config.motion_map_blur_radius > 0:
            echo_score = box_blur_2d(echo_score, radius=config.motion_map_blur_radius)

        residual += _expand_weight(echo_score, current) * (echo - background)

    return residual.astype(np.float32)


def _build_echo_residual_from_history(
    past_frames: Sequence[np.ndarray],
    current: np.ndarray,
    background: np.ndarray,
    sigma01: float,
    config: GhostTrail3DNRConfig,
) -> np.ndarray:
    current_activity = np.abs(raw_luma_proxy(current) - raw_luma_proxy(background))
    current_score = _activity_score(current_activity, sigma01=sigma01, config=config)
    clear_current = (1.0 - current_score) * (1.0 - current_score)

    residual = np.zeros_like(current, dtype=np.float32)
    for tap_index in range(config.echo_taps):
        delay = (tap_index + 1) * config.echo_delay
        if len(past_frames) < delay:
            break

        echo = past_frames[-delay]
        echo_activity = np.abs(raw_luma_proxy(echo) - raw_luma_proxy(background))
        echo_score = _activity_score(echo_activity, sigma01=sigma01, config=config)
        echo_score = echo_score * clear_current * np.float32(config.trail_decay**tap_index)

        if config.motion_map_blur_radius > 0:
            echo_score = box_blur_2d(echo_score, radius=config.motion_map_blur_radius)

        residual += _expand_weight(echo_score, current) * (echo - background)

    return residual.astype(np.float32)


def denoise_sequence_array(
    frames: Sequence[np.ndarray] | np.ndarray,
    config: GhostTrail3DNRConfig | None = None,
) -> tuple[np.ndarray, list[Frame3DNRStats]]:
    cfg = config or GhostTrail3DNRConfig()
    _validate_config(cfg)

    raw_stack = _as_frame_stack(frames)
    outputs01 = np.empty_like(raw_stack, dtype=np.float32)
    processor = GhostTrail3DNRProcessor(cfg)
    stats = []
    for frame_index in range(raw_stack.shape[0]):
        outputs01[frame_index], frame_stats = processor.process(raw_stack[frame_index])
        stats.append(frame_stats)

    return outputs01, stats


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    key: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", path.name.lower()):
        if part.isdigit():
            key.append((1, int(part)))
        elif part:
            key.append((0, part))
    return key


def find_sequence_paths(input_dir: str | Path, pattern: str = "*.tiff") -> list[Path]:
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"输入目录不存在: {folder}")
    paths = sorted((path for path in folder.glob(pattern) if path.is_file()), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"没有在 {folder} 中找到匹配 {pattern!r} 的文件")
    return paths


def read_tiff_stack(paths: Sequence[Path]) -> tuple[np.ndarray, np.dtype]:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("需要安装 tifffile 才能读写 TIFF") from exc

    frames = [np.asarray(tifffile.imread(path)) for path in paths]
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"所有帧尺寸必须一致，收到: {sorted(str(shape) for shape in shapes)}")
    return np.stack(frames, axis=0), frames[0].dtype


def write_tiff_stack(
    outputs01: np.ndarray,
    input_paths: Sequence[Path],
    output_dir: str | Path,
    dtype: np.dtype,
    config: GhostTrail3DNRConfig,
    suffix: str = "_ghost3dnr",
) -> list[Path]:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("需要安装 tifffile 才能读写 TIFF") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for frame01, input_path in zip(outputs01, input_paths, strict=True):
        output = restore_raw_dtype(
            frame01,
            dtype=dtype,
            black_level=config.black_level,
            white_level=config.white_level,
        )
        out_path = out_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
        tifffile.imwrite(out_path, output)
        written_paths.append(out_path)
    return written_paths


def run_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.tiff",
    config: GhostTrail3DNRConfig | None = None,
    suffix: str = "_ghost3dnr",
) -> tuple[list[Path], list[Frame3DNRStats]]:
    cfg = config or GhostTrail3DNRConfig()
    paths = find_sequence_paths(input_dir, pattern=pattern)
    raw_stack, dtype = read_tiff_stack(paths)
    outputs01, stats = denoise_sequence_array(raw_stack, config=cfg)
    written_paths = write_tiff_stack(outputs01, paths, output_dir, dtype=dtype, config=cfg, suffix=suffix)
    return written_paths, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAW 域拖影型递归 3DNR")
    parser.add_argument("--input-dir", required=True, help="输入 TIFF 序列目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--glob", default="*.tiff", help="输入文件匹配模式，例如 '*.tiff' 或 'frame*_noisy3.tiff'")
    parser.add_argument("--black-level", type=float, default=0.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--sigma01", type=float, default=None, help="归一化 [0,1] 噪声标准差；不填则逐帧估计")
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
    parser.add_argument("--suffix", default="_ghost3dnr")
    return parser


def _config_from_args(args: argparse.Namespace) -> GhostTrail3DNRConfig:
    return GhostTrail3DNRConfig(
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    written_paths, stats = run_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.glob,
        config=config,
        suffix=args.suffix,
    )

    print(f"共处理 {len(written_paths)} 帧，输出目录: {Path(args.output_dir)}")
    for item in stats:
        print(
            f"frame={item.frame_index:04d} "
            f"sigma01={item.sigma01:.6f} "
            f"motion={item.mean_motion_score:.4f} "
            f"history_weight={item.mean_history_weight:.4f}"
        )


if __name__ == "__main__":
    main()

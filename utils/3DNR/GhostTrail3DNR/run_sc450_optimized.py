from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import h5py
import numpy as np


@dataclass(frozen=True)
class BayerEcho3DNRConfig:
    black_level: float = 256.0
    white_level: float = 4095.0
    spatial_strength: float = 0.18
    spatial_edge_threshold: float = 0.018
    static_history_weight: float = 0.86
    motion_history_weight: float = 0.35
    motion_current_floor: float = 0.88
    background_history_weight: float = 0.94
    motion_threshold: float = 0.018
    motion_softness: float = 0.018
    motion_blur_radius: int = 1
    echo_delay: int = 2
    echo_taps: int = 4
    echo_decay: float = 0.68
    echo_strength: float = 1.0
    echo_current_suppression_power: float = 3.0
    matte_threshold: float = 0.026
    matte_softness: float = 0.018
    matte_dilate_radius: int = 2
    matte_feather_radius: int = 2
    trail_opacity: float = 0.78
    trail_max_alpha: float = 0.88
    trail_contrast: float = 1.18


class BayerEcho3DNR:
    def __init__(self, config: BayerEcho3DNRConfig) -> None:
        validate_config(config)
        self.config = config
        self.history: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self.buffer: list[np.ndarray] = []

    def process(self, raw: np.ndarray) -> np.ndarray:
        current = normalize_raw(raw, self.config.black_level, self.config.white_level)
        planes = pack_bayer_planes(current)
        planes = spatial_denoise_planes(planes, self.config)

        if self.history is None or self.background is None:
            self.history = planes.copy()
            self.background = planes.copy()
            self._push(planes)
            return unpack_bayer_planes(planes)

        motion = motion_score(planes, self.history, self.config)
        history_weight = history_weight_map(motion, self.config)
        base = history_weight[None, :, :] * self.history + (1.0 - history_weight[None, :, :]) * planes
        base = np.clip(base, 0.0, 1.0).astype(np.float32)

        out_planes = composite_foreground_echo(planes, base, self.background, self.buffer, self.config)

        bg_motion = motion_score(planes, self.background, self.config)
        bg_keep = self.config.background_history_weight + (1.0 - self.config.background_history_weight) * bg_motion
        self.background = bg_keep[None, :, :] * self.background + (1.0 - bg_keep[None, :, :]) * planes
        self.background = np.clip(self.background, 0.0, 1.0).astype(np.float32)
        self.history = base
        self._push(planes)
        return unpack_bayer_planes(out_planes)

    def _push(self, planes: np.ndarray) -> None:
        self.buffer.append(planes.copy())
        max_len = self.config.echo_delay * self.config.echo_taps
        if len(self.buffer) > max_len:
            del self.buffer[: len(self.buffer) - max_len]


class RawIspPreview:
    def __init__(self, exposure: float) -> None:
        self.exposure = exposure
        self.gains: np.ndarray | None = None

    def render(self, raw01: np.ndarray) -> np.ndarray:
        rgb = raw_to_half_rgb(raw01)
        if self.gains is None:
            self.gains = gray_world_gains(rgb)
        balanced = np.clip(rgb * self.gains.reshape(1, 1, 3) * self.exposure, 0.0, 1.0)
        srgb = np.clip(balanced, 0.0, 1.0) ** (1.0 / 2.2)
        return np.rint(srgb * 255.0).astype(np.uint8)


def validate_config(config: BayerEcho3DNRConfig) -> None:
    if config.white_level <= config.black_level:
        raise ValueError("white_level must be greater than black_level")
    if not 0.0 <= config.spatial_strength <= 1.0:
        raise ValueError("spatial_strength must be in [0, 1]")
    if not 0.0 <= config.static_history_weight < 1.0:
        raise ValueError("static_history_weight must be in [0, 1)")
    if not 0.0 <= config.motion_history_weight < 1.0:
        raise ValueError("motion_history_weight must be in [0, 1)")
    if not 0.0 <= config.motion_current_floor <= 1.0:
        raise ValueError("motion_current_floor must be in [0, 1]")
    if not 0.0 <= config.background_history_weight < 1.0:
        raise ValueError("background_history_weight must be in [0, 1)")
    if config.motion_threshold <= 0.0:
        raise ValueError("motion_threshold must be positive")
    if config.motion_softness <= 0.0:
        raise ValueError("motion_softness must be positive")
    if config.motion_blur_radius < 0:
        raise ValueError("motion_blur_radius must be non-negative")
    if config.echo_delay < 1:
        raise ValueError("echo_delay must be at least 1")
    if config.echo_taps < 1:
        raise ValueError("echo_taps must be at least 1")
    if not 0.0 <= config.echo_decay < 1.0:
        raise ValueError("echo_decay must be in [0, 1)")
    if not 0.0 <= config.echo_strength <= 2.0:
        raise ValueError("echo_strength must be in [0, 2]")
    if config.echo_current_suppression_power <= 0.0:
        raise ValueError("echo_current_suppression_power must be positive")
    if config.matte_threshold <= 0.0:
        raise ValueError("matte_threshold must be positive")
    if config.matte_softness <= 0.0:
        raise ValueError("matte_softness must be positive")
    if config.matte_dilate_radius < 0:
        raise ValueError("matte_dilate_radius must be non-negative")
    if config.matte_feather_radius < 0:
        raise ValueError("matte_feather_radius must be non-negative")
    if not 0.0 <= config.trail_opacity <= 1.0:
        raise ValueError("trail_opacity must be in [0, 1]")
    if not 0.0 <= config.trail_max_alpha <= 1.0:
        raise ValueError("trail_max_alpha must be in [0, 1]")
    if config.trail_contrast <= 0.0:
        raise ValueError("trail_contrast must be positive")


def normalize_raw(raw: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    denom = float(white_level - black_level)
    if denom <= 0.0:
        raise ValueError("white_level must be greater than black_level")
    return np.clip((raw.astype(np.float32) - float(black_level)) / denom, 0.0, 1.0).astype(np.float32)


def pack_bayer_planes(raw01: np.ndarray) -> np.ndarray:
    h_even = raw01.shape[0] - raw01.shape[0] % 2
    w_even = raw01.shape[1] - raw01.shape[1] % 2
    raw_even = raw01[:h_even, :w_even]
    return np.stack(
        [
            raw_even[0::2, 0::2],
            raw_even[0::2, 1::2],
            raw_even[1::2, 0::2],
            raw_even[1::2, 1::2],
        ],
        axis=0,
    ).astype(np.float32)


def unpack_bayer_planes(planes: np.ndarray) -> np.ndarray:
    _, half_h, half_w = planes.shape
    raw = np.empty((half_h * 2, half_w * 2), dtype=np.float32)
    raw[0::2, 0::2] = planes[0]
    raw[0::2, 1::2] = planes[1]
    raw[1::2, 0::2] = planes[2]
    raw[1::2, 1::2] = planes[3]
    return raw


def spatial_denoise_planes(planes: np.ndarray, config: BayerEcho3DNRConfig) -> np.ndarray:
    out = np.empty_like(planes, dtype=np.float32)
    for channel in range(planes.shape[0]):
        plane = planes[channel]
        smooth = box_blur_2d(plane, radius=1)
        edge = np.abs(plane - smooth)
        weight = np.exp(-edge / max(config.spatial_edge_threshold, 1e-6)).astype(np.float32)
        out[channel] = (1.0 - config.spatial_strength * weight) * plane + config.spatial_strength * weight * smooth
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def box_blur_2d(data: np.ndarray, radius: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"box_blur_2d expects a 2D array, got shape={data.shape}")
    if radius <= 0:
        return data.astype(np.float32, copy=False)
    blurred = box_blur_axis(data.astype(np.float32, copy=False), radius=radius, axis=0)
    return box_blur_axis(blurred, radius=radius, axis=1)


def box_blur_axis(data: np.ndarray, radius: int, axis: int) -> np.ndarray:
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


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x01 = np.clip(x, 0.0, 1.0).astype(np.float32)
    return x01 * x01 * (3.0 - 2.0 * x01)


def planes_luma(planes: np.ndarray) -> np.ndarray:
    return np.mean(planes, axis=0).astype(np.float32)


def motion_score(current: np.ndarray, reference: np.ndarray, config: BayerEcho3DNRConfig) -> np.ndarray:
    diff = np.abs(planes_luma(current) - planes_luma(reference))
    diff = box_blur_2d(diff, radius=config.motion_blur_radius)
    score = smoothstep01((diff - config.motion_threshold) / max(config.motion_softness, 1e-6))
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def history_weight_map(motion: np.ndarray, config: BayerEcho3DNRConfig) -> np.ndarray:
    weight = config.static_history_weight * (1.0 - motion) + config.motion_history_weight * motion
    max_weight = 1.0 - config.motion_current_floor * motion
    return np.clip(np.minimum(weight, max_weight), 0.0, 0.999).astype(np.float32)


def activity_score(planes: np.ndarray, background: np.ndarray, config: BayerEcho3DNRConfig) -> np.ndarray:
    diff = np.abs(planes_luma(planes) - planes_luma(background))
    diff = box_blur_2d(diff, radius=config.motion_blur_radius)
    score = smoothstep01((diff - config.motion_threshold) / max(config.motion_softness, 1e-6))
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def foreground_matte(planes: np.ndarray, background: np.ndarray, config: BayerEcho3DNRConfig) -> np.ndarray:
    diff = np.abs(planes_luma(planes) - planes_luma(background))
    diff = box_blur_2d(diff, radius=config.motion_blur_radius)
    matte = smoothstep01((diff - config.matte_threshold) / max(config.matte_softness, 1e-6))
    matte = soft_dilate_2d(matte, radius=config.matte_dilate_radius)
    matte = box_blur_2d(matte, radius=config.matte_feather_radius)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def soft_dilate_2d(data: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return data.astype(np.float32, copy=False)
    padded = np.pad(data.astype(np.float32, copy=False), ((radius, radius), (radius, radius)), mode="edge")
    candidates = []
    for y_offset in range(2 * radius + 1):
        for x_offset in range(2 * radius + 1):
            candidates.append(padded[y_offset : y_offset + data.shape[0], x_offset : x_offset + data.shape[1]])
    return np.maximum.reduce(candidates).astype(np.float32)


def composite_foreground_echo(
    current: np.ndarray,
    base: np.ndarray,
    background: np.ndarray,
    buffer: Sequence[np.ndarray],
    config: BayerEcho3DNRConfig,
) -> np.ndarray:
    current_matte = foreground_matte(current, background, config)
    clear_current = (1.0 - current_matte) ** config.echo_current_suppression_power
    trail_color = np.zeros_like(current, dtype=np.float32)
    trail_alpha = np.zeros_like(current_matte, dtype=np.float32)

    for tap in range(config.echo_taps):
        delay = (tap + 1) * config.echo_delay
        if len(buffer) < delay:
            break

        old = buffer[-delay]
        matte = foreground_matte(old, background, config)
        tap_alpha = matte * clear_current * np.float32(config.trail_opacity * (config.echo_decay**tap))
        tap_alpha = np.clip(tap_alpha, 0.0, config.trail_max_alpha)
        incoming_alpha = tap_alpha * (1.0 - trail_alpha)
        boosted_old = boost_foreground_contrast(old, background, config)
        trail_color += incoming_alpha[None, :, :] * boosted_old
        trail_alpha += incoming_alpha

    alpha = np.clip(config.echo_strength * trail_alpha, 0.0, config.trail_max_alpha)
    has_trail = alpha > 1e-6
    layer = np.where(has_trail[None, :, :], trail_color / np.maximum(trail_alpha, 1e-6)[None, :, :], base)
    return np.clip((1.0 - alpha[None, :, :]) * base + alpha[None, :, :] * layer, 0.0, 1.0).astype(np.float32)


def boost_foreground_contrast(old: np.ndarray, background: np.ndarray, config: BayerEcho3DNRConfig) -> np.ndarray:
    foreground_delta = old - background
    boosted = background + config.trail_contrast * foreground_delta
    return np.clip(boosted, 0.0, 1.0).astype(np.float32)


def raw_to_half_rgb(raw01: np.ndarray) -> np.ndarray:
    h_even = raw01.shape[0] - raw01.shape[0] % 2
    w_even = raw01.shape[1] - raw01.shape[1] % 2
    raw_even = raw01[:h_even, :w_even]
    blue = raw_even[0::2, 1::2]
    red = raw_even[1::2, 0::2]
    green = 0.5 * (raw_even[0::2, 0::2] + raw_even[1::2, 1::2])
    return np.stack([red, green, blue], axis=2).astype(np.float32)


def gray_world_gains(rgb: np.ndarray) -> np.ndarray:
    mean = np.mean(rgb, axis=(0, 1))
    target = float(np.mean(mean))
    return np.clip(target / np.maximum(mean, 1e-6), 0.25, 4.0).astype(np.float32)


def shard_paths(scene_dir: Path) -> list[Path]:
    return sorted(scene_dir.glob("shard_*.h5"), key=lambda path: int(path.stem.split("_")[-1]))


def current_noisy_frame(h5_file: h5py.File, index: int, noisy_channel: int) -> np.ndarray:
    return np.asarray(h5_file["noisy"][index, noisy_channel])


def iter_scene_frames(scene_dir: Path, noisy_channel: int, max_frames: int) -> Iterator[np.ndarray]:
    yielded = 0
    for shard_path in shard_paths(scene_dir):
        with h5py.File(shard_path, "r") as h5_file:
            noisy = h5_file["noisy"]
            for index in range(int(noisy.shape[0])):
                yield np.asarray(noisy[index, noisy_channel])
                yielded += 1
                if yielded >= max_frames:
                    return


def write_avi_rgb(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("frames must contain at least one RGB frame")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    row_size = ((width * 3 + 3) // 4) * 4
    frame_size = row_size * height
    frame_count = len(frames)
    idx_entries: list[tuple[bytes, int, int, int]] = []

    with path.open("wb") as file:
        file.write(b"RIFF")
        riff_size_pos = file.tell()
        file.write(struct.pack("<I", 0))
        file.write(b"AVI ")
        write_hdrl(file, width, height, fps, frame_count, frame_size)

        file.write(b"LIST")
        movi_size_pos = file.tell()
        file.write(struct.pack("<I", 0))
        file.write(b"movi")
        movi_data_start = file.tell()

        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"all frames must have shape {(height, width)}, got {frame.shape[:2]}")

            chunk_start = file.tell()
            file.write(b"00db")
            file.write(struct.pack("<I", frame_size))
            file.write(rgb_to_bgr_bottom_up(frame, row_size))
            if frame_size % 2:
                file.write(b"\x00")
            idx_entries.append((b"00db", 0x10, chunk_start - movi_data_start, frame_size))

        movi_end = file.tell()
        file.write(b"idx1")
        file.write(struct.pack("<I", len(idx_entries) * 16))
        for chunk_id, flags, offset, size in idx_entries:
            file.write(chunk_id)
            file.write(struct.pack("<III", flags, offset, size))

        end = file.tell()
        file.seek(riff_size_pos)
        file.write(struct.pack("<I", end - 8))
        file.seek(movi_size_pos)
        file.write(struct.pack("<I", movi_end - movi_size_pos - 4))


def write_hdrl(file: BinaryIO, width: int, height: int, fps: int, frame_count: int, frame_size: int) -> None:
    hdrl = bytearray()
    avih = struct.pack(
        "<IIIIIIIIIIIIII",
        int(1_000_000 / fps),
        frame_size * fps,
        0,
        0x10,
        frame_count,
        0,
        1,
        frame_size,
        width,
        height,
        0,
        0,
        0,
        0,
    )
    hdrl.extend(make_chunk(b"avih", avih))

    strl = bytearray()
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIiiii",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        frame_count,
        frame_size,
        0xFFFFFFFF,
        0,
        0,
        0,
        width,
        height,
    )
    strl.extend(make_chunk(b"strh", strh))
    strf = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        24,
        0,
        frame_size,
        0,
        0,
        0,
        0,
    )
    strl.extend(make_chunk(b"strf", strf))
    hdrl.extend(make_list(b"strl", bytes(strl)))
    file.write(make_list(b"hdrl", bytes(hdrl)))


def make_chunk(chunk_id: bytes, data: bytes) -> bytes:
    pad = b"\x00" if len(data) % 2 else b""
    return chunk_id + struct.pack("<I", len(data)) + data + pad


def make_list(list_id: bytes, data: bytes) -> bytes:
    pad = b"\x00" if len(data) % 2 else b""
    return b"LIST" + struct.pack("<I", len(data) + 4) + list_id + data + pad


def rgb_to_bgr_bottom_up(frame: np.ndarray, row_size: int) -> bytes:
    frame8 = np.asarray(frame, dtype=np.uint8)
    if frame8.ndim != 3 or frame8.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape [H, W, 3], got {frame8.shape}")

    bgr = np.ascontiguousarray(frame8[..., ::-1])
    rows = []
    pad = row_size - bgr.shape[1] * 3
    padding = b"\x00" * pad
    for row in bgr[::-1]:
        rows.append(row.tobytes())
        if pad:
            rows.append(padding)
    return b"".join(rows)


def hstack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"comparison frames must share shape, got {left.shape} and {right.shape}")
    return np.ascontiguousarray(np.concatenate([left, right], axis=1))


def parse_scene_dir(path: str | Path) -> Path:
    scene_dir = Path(path)
    if (scene_dir / "metadata.json").exists():
        return scene_dir

    children = [child for child in scene_dir.iterdir() if child.is_dir() and (child / "metadata.json").exists()]
    if len(children) == 1:
        return children[0]
    raise FileNotFoundError(f"cannot find a unique scene with metadata.json under {scene_dir}")


def read_metadata(scene_dir: Path) -> dict[str, object]:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized Bayer-domain echo 3DNR for the sc450 H5 scenes.")
    parser.add_argument("--scene-dir", required=True, help="SC450 RAW 序列目录")
    parser.add_argument("--output-dir", default="sc450_outputs")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--noisy-channel", type=int, default=1)
    parser.add_argument("--exposure", type=float, default=18.0)
    parser.add_argument("--raw-name", default="original_noisy.avi")
    parser.add_argument("--denoised-name", default="optimized_3dnr_foreground_echo.avi")
    parser.add_argument("--comparison-name", default="comparison_noisy_vs_foreground_echo.avi")
    parser.add_argument("--black-level", type=float, default=256.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--spatial-strength", type=float, default=0.18)
    parser.add_argument("--static-history-weight", type=float, default=0.86)
    parser.add_argument("--motion-history-weight", type=float, default=0.35)
    parser.add_argument("--trail-strength", type=float, default=1.0)
    parser.add_argument("--trail-opacity", type=float, default=0.78)
    parser.add_argument("--trail-max-alpha", type=float, default=0.88)
    parser.add_argument("--trail-contrast", type=float, default=1.18)
    parser.add_argument("--matte-threshold", type=float, default=0.026)
    parser.add_argument("--matte-dilate-radius", type=int, default=2)
    return parser


def config_from_args(args: argparse.Namespace) -> BayerEcho3DNRConfig:
    return BayerEcho3DNRConfig(
        black_level=args.black_level,
        white_level=args.white_level,
        spatial_strength=args.spatial_strength,
        static_history_weight=args.static_history_weight,
        motion_history_weight=args.motion_history_weight,
        echo_strength=args.trail_strength,
        trail_opacity=args.trail_opacity,
        trail_max_alpha=args.trail_max_alpha,
        trail_contrast=args.trail_contrast,
        matte_threshold=args.matte_threshold,
        matte_dilate_radius=args.matte_dilate_radius,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    scene_dir = parse_scene_dir(args.scene_dir)
    metadata = read_metadata(scene_dir)
    output_dir = Path(args.output_dir) / scene_dir.name
    config = config_from_args(args)
    processor = BayerEcho3DNR(config)
    raw_isp = RawIspPreview(exposure=args.exposure)
    den_isp = RawIspPreview(exposure=args.exposure)
    raw_video_frames: list[np.ndarray] = []
    denoised_video_frames: list[np.ndarray] = []
    comparison_frames: list[np.ndarray] = []

    frame_iter = iter_scene_frames(scene_dir, noisy_channel=args.noisy_channel, max_frames=args.max_frames)
    print(
        f"scene={scene_dir.name}, target_frames={args.max_frames}, frame_shape={metadata['frame_shape']}, "
        f"output={output_dir}",
        flush=True,
    )

    for index, raw in enumerate(frame_iter):
        raw01 = normalize_raw(raw, config.black_level, config.white_level)
        denoised01 = processor.process(raw)
        raw_frame = raw_isp.render(raw01)
        denoised_frame = den_isp.render(denoised01)
        raw_video_frames.append(raw_frame)
        denoised_video_frames.append(denoised_frame)
        comparison_frames.append(hstack_frames(raw_frame, denoised_frame))
        if (index + 1) % 15 == 0:
            print(f"processed {index + 1} frames", flush=True)

    if not raw_video_frames:
        raise FileNotFoundError(f"no frames were read from {scene_dir}")

    raw_video_path = output_dir / args.raw_name
    denoised_video_path = output_dir / args.denoised_name
    comparison_path = output_dir / args.comparison_name
    write_avi_rgb(raw_video_path, raw_video_frames, fps=args.fps)
    write_avi_rgb(denoised_video_path, denoised_video_frames, fps=args.fps)
    write_avi_rgb(comparison_path, comparison_frames, fps=args.fps)
    print(f"raw noisy video: {raw_video_path}", flush=True)
    print(f"optimized 3dnr video: {denoised_video_path}", flush=True)
    print(f"comparison video: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()

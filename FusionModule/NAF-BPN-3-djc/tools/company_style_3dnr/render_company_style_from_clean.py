"""Generate a clean-based, deliberately imperfect company-style 3DNR video.

The pipeline is intentionally a low-cost ISP simulation:
clean RAW -> reduced Poisson/read noise -> strong Bayer-plane 2DNR ->
coarse frame-difference gating -> two-frame temporal blend in static areas
and mostly-current-noisy output in motion areas.

Only rendered videos, PNG previews, metrics, and JSON metadata are written.
No synthetic RAW sequence or H5 file is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


def shard_paths(root: Path) -> list[Path]:
    paths = list(root.glob("shard_*.h5"))
    return sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def split_bayer(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # GBRG: G1, B, R, G2. Every operation remains within one CFA position.
    return raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]


def merge_bayer(
    planes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.empty(shape, dtype=np.float32)
    out[0::2, 0::2], out[0::2, 1::2] = planes[0], planes[1]
    out[1::2, 0::2], out[1::2, 1::2] = planes[2], planes[3]
    return out


class NoiseSynthesizer:
    def __init__(self, black_level: float, white_level: float, k: float,
                 read_sigma: float, scale: float, seed: int) -> None:
        self.black_level = black_level
        self.white_level = white_level
        self.k = k
        self.read_sigma = read_sigma
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, clean: np.ndarray) -> np.ndarray:
        signal = np.clip(clean.astype(np.float32) - self.black_level, 0.0, None)
        shot = self.k * self.rng.poisson(signal / self.k).astype(np.float32)
        read = self.rng.normal(0.0, self.read_sigma, clean.shape).astype(np.float32)
        noise = (shot - signal) + read
        return np.clip(clean.astype(np.float32) + self.scale * noise,
                       0.0, self.white_level)


class StrongBayer2DNR:
    def __init__(self, diameter: int, sigma_space: float, sigma_range: float) -> None:
        self.diameter = diameter
        self.sigma_space = sigma_space
        self.sigma_range = sigma_range

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        filtered = tuple(
            cv2.bilateralFilter(
                plane.astype(np.float32, copy=False),
                self.diameter,
                self.sigma_range,
                self.sigma_space,
            )
            for plane in split_bayer(raw)
        )
        return merge_bayer(filtered, raw.shape)


class CompanyStyle3DNR:
    def __init__(self, threshold: float, block: int, dilation: int,
                 static_2dnr_weight: float, motion_noise_gain: float) -> None:
        self.threshold = threshold
        self.block = block
        self.dilation = dilation
        self.static_2dnr_weight = static_2dnr_weight
        self.motion_noise_gain = motion_noise_gain
        self.previous_proxy: np.ndarray | None = None
        self.previous_noisy: np.ndarray | None = None

    @staticmethod
    def proxy(raw: np.ndarray) -> np.ndarray:
        planes = split_bayer(raw)
        # The cross-plane mean is used only as a luma-like motion score.
        return cv2.blur(np.mean(np.stack(planes), axis=0), (5, 5))

    def __call__(self, noisy: np.ndarray, two_dnr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proxy = self.proxy(two_dnr)
        if self.previous_proxy is None:
            mask_half = np.zeros_like(proxy, dtype=np.uint8)
        else:
            score = np.abs(proxy - self.previous_proxy)
            block_w = max(1, proxy.shape[1] // self.block)
            block_h = max(1, proxy.shape[0] // self.block)
            block_score = cv2.resize(score, (block_w, block_h), interpolation=cv2.INTER_AREA)
            blocks = (block_score >= self.threshold).astype(np.uint8)
            if self.dilation > 1:
                blocks = cv2.dilate(blocks, np.ones((self.dilation, self.dilation), np.uint8))
            mask_half = cv2.resize(blocks, (proxy.shape[1], proxy.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.previous_noisy is None:
            temporal = noisy
        else:
            temporal = 0.5 * noisy + 0.5 * self.previous_noisy
        # Static output stays close to 2DNR; temporal noisy texture is only a small gain.
        static = self.static_2dnr_weight * two_dnr + (1.0 - self.static_2dnr_weight) * temporal
        # Motion deliberately amplifies the current noisy residual to reproduce company-style artifacts.
        motion = two_dnr + self.motion_noise_gain * (noisy - two_dnr)
        mask_full = cv2.resize(mask_half, (noisy.shape[1], noisy.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        output = np.where(mask_full, motion, static).astype(np.float32)
        self.previous_proxy = proxy
        self.previous_noisy = noisy.copy()
        return np.clip(output, 0.0, 4095.0), mask_full


class FixedISP:
    def __init__(self, black_level: float, white_level: float, exposure_compensation: float = 1.0) -> None:
        self.black_level = black_level
        self.white_level = white_level
        self.exposure_compensation = exposure_compensation
        self.gains: tuple[float, float] | None = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        x = np.clip((raw.astype(np.float32) - self.black_level) /
                    (self.white_level - self.black_level), 0.0, 1.0)
        planes = list(split_bayer(x))
        if self.gains is None:
            green = float((planes[0].mean() + planes[3].mean()) / 2.0)
            self.gains = (
                green / (float(planes[2].mean()) + 1e-6),
                green / (float(planes[1].mean()) + 1e-6),
            )
        planes[2] *= self.gains[0]
        planes[1] *= self.gains[1]
        # Exposure compensation is applied in the linear domain, before demosaic/gamma.
        balanced = np.clip(merge_bayer(tuple(planes), raw.shape) * self.exposure_compensation, 0.0, 1.0)
        bgr16 = cv2.cvtColor(np.clip(balanced * 65535.0, 0, 65535).astype(np.uint16), cv2.COLOR_BayerGB2BGR)
        return np.clip(np.power(bgr16.astype(np.float32) / 65535.0, 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)


def annotate(frame: np.ndarray, text: str, frame_index: int) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (min(650, out.shape[1]), 76), (0, 0, 0), -1)
    cv2.putText(out, text, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, f"FRAME {frame_index:04d}", (24, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class VideoWriterPipe:
    def __init__(self, path: Path, width: int, height: int, fps: int, codec: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if codec == "h264":
            args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        elif codec == "ffv1":
            args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "ffv1",
                    "-level", "3", "-coder", "1", "-context", "1", "-g", "1", "-pix_fmt", "bgr24", str(path)]
        else:
            raise ValueError(codec)
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("video encoder stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"{self.process.args[0]} failed ({code}): {error[-1000:]}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", type=Path, default=Path("data/H5/scene_10"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/company_style_preview"))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--black-level", type=float, default=16.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--shot-k", type=float, default=4.0)
    parser.add_argument("--read-sigma", type=float, default=120.0)
    parser.add_argument("--noise-scale", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--bilateral-diameter", type=int, default=9)
    parser.add_argument("--bilateral-sigma-space", type=float, default=7.0)
    parser.add_argument("--bilateral-sigma-range", type=float, default=110.0)
    parser.add_argument("--motion-threshold", type=float, default=45.0)
    parser.add_argument("--motion-block", type=int, default=8)
    parser.add_argument("--motion-dilation", type=int, default=3)
    parser.add_argument("--static-2dnr-weight", type=float, default=0.80,
                        help="2DNR weight in static 3DNR regions; remaining weight adds slight temporal texture")
    parser.add_argument("--motion-noise-gain", type=float, default=1.35,
                        help="gain applied to noisy-2DNR residual in motion regions")
    parser.add_argument("--max-frames", type=int, default=None, help="optional smoke-test limit")
    parser.add_argument("--exposure-compensation", type=float, default=1.0,
                        help="linear ISP exposure multiplier, e.g. 5.0")
    parser.add_argument("--output-stem", type=str, default="clean_based_company_style_3dnr")
    parser.add_argument("--no-lossless", action="store_true", help="write only the H.264 MP4")
    args = parser.parse_args()
    if not 0.0 <= args.noise_scale <= 1.0:
        raise ValueError("noise-scale must be in [0, 1]")
    if not 0.0 <= args.static_2dnr_weight <= 1.0:
        raise ValueError("static-2dnr-weight must be in [0, 1]")
    if args.motion_noise_gain < 0.0:
        raise ValueError("motion-noise-gain must be non-negative")
    if args.exposure_compensation <= 0.0:
        raise ValueError("exposure-compensation must be positive")

    shards = shard_paths(args.h5_dir)
    if not shards:
        raise FileNotFoundError(f"no shard_*.h5 under {args.h5_dir}")
    with h5py.File(shards[0], "r") as h5:
        if "clean" not in h5:
            raise KeyError("clean")
        frames_per_shard, height, width = h5["clean"].shape
        if (height, width) != (1080, 1920):
            raise ValueError(f"expected 1080x1920 clean frames, got {(height, width)}")
    total_frames = len(shards) * frames_per_shard
    expected_frames = min(total_frames, args.max_frames) if args.max_frames is not None else total_frames
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "clean_based_company_style_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_base = out_dir / args.output_stem
    panel_w, panel_h = width // 2, height // 2
    writers = [VideoWriterPipe(output_base.with_suffix(".mp4"), width, height, args.fps, "h264")]
    if not args.no_lossless:
        writers.append(VideoWriterPipe(output_base.with_name(output_base.name + "_lossless").with_suffix(".mkv"), width, height, args.fps, "ffv1"))
    noise = NoiseSynthesizer(args.black_level, args.white_level, args.shot_k, args.read_sigma, args.noise_scale, args.seed)
    two_dnr = StrongBayer2DNR(args.bilateral_diameter, args.bilateral_sigma_space, args.bilateral_sigma_range)
    three_dnr = CompanyStyle3DNR(args.motion_threshold, args.motion_block, args.motion_dilation, args.static_2dnr_weight, args.motion_noise_gain)
    isp = FixedISP(args.black_level, args.white_level, args.exposure_compensation)
    fixed_indices = set(range(0, expected_frames, 20))
    high_motion_saved = 0
    coverage: list[float] = []
    rows: list[dict[str, Any]] = []
    h5_hashes = {str(path): sha256(path) for path in shards}
    previous_outputs: dict[str, np.ndarray] = {}
    frame_index = 0
    try:
        for shard in shards:
            with h5py.File(shard, "r") as h5:
                for local_index in range(frames_per_shard):
                    if frame_index >= expected_frames:
                        break
                    clean = h5["clean"][local_index].astype(np.float32)
                    noisy = noise(clean)
                    denoised = two_dnr(noisy)
                    simulated, mask = three_dnr(noisy, denoised)
                    rendered = {name: isp(raw) for name, raw in (("noisy", noisy), ("2dnr", denoised), ("3dnr", simulated), ("clean", clean))}
                    panels = {name: annotate(cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_AREA), title, frame_index)
                              for (name, image), title in zip(rendered.items(), ("NOISY", "2DNR", "SIMULATED 3DNR", "CLEAN"))}
                    frame = np.vstack((np.hstack((panels["noisy"], panels["2dnr"])), np.hstack((panels["3dnr"], panels["clean"]))))
                    for writer in writers:
                        writer.write(frame)

                    motion_fraction = float(mask.mean())
                    coverage.append(motion_fraction)
                    motion_noise = float(np.std((simulated - clean)[mask])) if mask.any() else 0.0
                    static_noise = float(np.std((simulated - clean)[~mask])) if (~mask).any() else 0.0
                    row: dict[str, Any] = {"frame": frame_index, "motion_fraction": motion_fraction,
                                           "noisy_residual_std": float(np.std(noisy - clean)),
                                           "2dnr_residual_std": float(np.std(denoised - clean)),
                                           "3dnr_static_residual_std": static_noise,
                                           "3dnr_motion_residual_std": motion_noise}
                    for name, image in (("2dnr", denoised), ("3dnr", simulated)):
                        if name in previous_outputs:
                            row[f"{name}_frame_mad"] = float(np.mean(np.abs(image - previous_outputs[name])))
                        previous_outputs[name] = image
                    rows.append(row)

                    if frame_index in fixed_indices or (motion_fraction >= 0.10 and high_motion_saved < 8):
                        suffix = "fixed" if frame_index in fixed_indices else "high_motion"
                        cv2.imwrite(str(preview_dir / f"frame_{frame_index:04d}_{suffix}.png"), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if suffix == "high_motion":
                            high_motion_saved += 1
                    frame_index += 1
            if frame_index >= expected_frames:
                break
    finally:
        for writer in writers:
            writer.close()

    if frame_index != expected_frames:
        raise RuntimeError(f"frame count mismatch: rendered {frame_index}, expected {expected_frames}")
    after_hashes = {str(path): sha256(path) for path in shards}
    if h5_hashes != after_hashes:
        raise RuntimeError("H5 SHA-256 changed during rendering")
    summary = {
        "frames": frame_index,
        "mean_motion_fraction": float(np.mean(coverage)),
        "mean_noisy_residual_std": float(np.mean([r["noisy_residual_std"] for r in rows])),
        "mean_2dnr_residual_std": float(np.mean([r["2dnr_residual_std"] for r in rows])),
        "mean_3dnr_static_residual_std": float(np.mean([r["3dnr_static_residual_std"] for r in rows])),
        "mean_3dnr_motion_residual_std": float(np.mean([r["3dnr_motion_residual_std"] for r in rows if r["motion_fraction"] > 0])) if any(r["motion_fraction"] > 0 for r in rows) else 0.0,
    }
    config = {
        "input": str(args.h5_dir), "shards": [str(p) for p in shards], "frames": frame_index,
        "resolution": [width, height], "fps": args.fps, "bayer": "GBRG",
        "isp": {"exposure_compensation_linear": args.exposure_compensation, "gamma": 2.2},
        "noise": {"operator": "Poisson shot + Gaussian read noise", "black_level": args.black_level, "white_level": args.white_level, "shot_k": args.shot_k, "read_sigma": args.read_sigma, "noise_scale": args.noise_scale, "seed": args.seed},
        "2dnr": {"operator": "per-CFA-plane float32 bilateral", "diameter": args.bilateral_diameter, "sigma_space": args.bilateral_sigma_space, "sigma_range": args.bilateral_sigma_range},
        "3dnr": {"operator": "coarse 2DNR frame-difference gate + low-texture static blend + amplified motion residual", "motion_threshold": args.motion_threshold, "motion_block": args.motion_block, "motion_dilation": args.motion_dilation, "static": f"{args.static_2dnr_weight:.2f} current 2dnr + {1.0 - args.static_2dnr_weight:.2f} causal 2-frame noisy mean", "motion": f"current 2dnr + {args.motion_noise_gain:.2f} * (current noisy - current 2dnr)"},
        "outputs": {"playback": str(output_base.with_suffix(".mp4")), "lossless": None if args.no_lossless else str(output_base.with_name(output_base.name + "_lossless").with_suffix(".mkv")), "raw_intermediates_saved": False},
        "h5_sha256_before_after": h5_hashes,
        "summary": summary,
    }
    output_base.with_suffix(".json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "clean_based_company_style_metrics.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

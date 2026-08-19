from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from data import CODE_MAX, FRAME_COUNT, HEIGHT, WIDTH, discover_sequences, read_candidate, read_source


ROOT = Path(__file__).resolve().parent
SOURCE_BLACK = 252
NR_BLACK = 300
FPS = 25
GAIN_PATTERN = re.compile(r"R=(\d+),G=(\d+),B=(\d+)")


class VideoWriter:
    def __init__(self, output: Path):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise FileNotFoundError("未找到 ffmpeg")
        self.output = output
        self.process = subprocess.Popen(
            [executable, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(output)],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
            raise ValueError("视频帧必须是 1920x1080 BGR uint8")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        detail = self.process.stderr.read().decode("utf-8", errors="replace")
        if self.process.wait() != 0:
            raise RuntimeError(f"视频编码失败: {detail}")


def parse_args():
    parser = argparse.ArgumentParser(description="为 N2N 四相位重组结果生成 AI 与四宫格视频")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", choices=("128x", "645x", "all"), default="all")
    parser.add_argument("--exposure-multiplier", type=float, default=1.0, help="在自动标定曝光后的额外显示曝光补偿")
    return parser.parse_args()


def gains_from_path(path: Path) -> tuple[int, int, int]:
    for text in (path.name, path.parent.name):
        match = GAIN_PATTERN.search(text)
        if match:
            return tuple(int(value) for value in match.groups())
    raise ValueError(f"无法从路径解析 R/G/B 增益: {path}")


def read_raw(path: Path) -> np.ndarray:
    return np.asarray(np.memmap(path, dtype="<u2", mode="r", shape=(HEIGHT, WIDTH))).copy()


def to_bgr(frame: np.ndarray, gains: tuple[int, int, int], black_level: int, exposure: float) -> np.ndarray:
    normalized = np.clip((frame.astype(np.float32) - black_level) / max(CODE_MAX - black_level, 1), 0.0, 1.0)
    red_gain, green_gain, blue_gain = gains
    normalized[0::2, 0::2] *= blue_gain / green_gain
    normalized[1::2, 1::2] *= red_gain / green_gain
    mosaic = np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)
    bgr = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR).astype(np.float32) / 65535.0
    return np.rint(np.power(np.clip(bgr * exposure, 0.0, 1.0), 1.0 / 2.2) * 255.0).astype(np.uint8)


def label(frame: np.ndarray, name: str, sequence: str, index: int) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (500, 65), (0, 0, 0), -1)
    cv2.putText(output, name, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"{sequence}  frame {index:03d}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def estimate_exposure(dnr2_path: Path, reference_png: Path, gains: tuple[int, int, int]) -> float:
    raw = read_raw(dnr2_path)
    reference = cv2.imread(str(reference_png), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"缺少 ISP 标定 PNG: {reference_png}")
    preview = to_bgr(raw, gains, NR_BLACK, 1.0)
    raw_luma = np.percentile((0.0722 * preview[::8, ::8, 0] + 0.7152 * preview[::8, ::8, 1] + 0.2126 * preview[::8, ::8, 2]) / 255.0, 90)
    ref_luma = np.percentile((0.0722 * reference[::8, ::8, 0] + 0.7152 * reference[::8, ::8, 1] + 0.2126 * reference[::8, ::8, 2]) / 255.0, 90)
    return float(np.clip((ref_luma ** 2.2) / max(raw_luma, 1e-6), 0.25, 64.0))


def verify_video(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise FileNotFoundError("未找到 ffprobe")
    result = subprocess.run([executable, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    if (stream["codec_name"], int(stream["width"]), int(stream["height"]), int(stream["nb_read_frames"])) != ("h264", WIDTH, HEIGHT, FRAME_COUNT):
        raise ValueError(f"视频校验失败: {path}")
    return {"file": path.name, "codec": stream["codec_name"], "width": int(stream["width"]), "height": int(stream["height"]), "fps": float(Fraction(stream["avg_frame_rate"])), "frames": int(stream["nb_read_frames"])}


def render_sequence(sequence, result_root: Path, exposure_multiplier: float) -> dict:
    raw_paths = tuple(result_root.glob("out_*.raw"))
    if len(raw_paths) != FRAME_COUNT:
        raise ValueError(f"{sequence.name} 输出 RAW 不完整: {len(raw_paths)}/200")
    gains = gains_from_path(sequence.source_path)
    exposure = estimate_exposure(sequence.dnr2_paths[100], sequence.dnr2_paths[100].with_suffix(".png"), gains) * exposure_multiplier
    ai_path = result_root / f"{sequence.name}_ai_fused.mp4"
    comparison_path = result_root / f"{sequence.name}_comparison.mp4"
    ai_writer = VideoWriter(ai_path)
    comparison_writer = VideoWriter(comparison_path)
    try:
        for index in range(FRAME_COUNT):
            noisy = read_source(sequence, index)
            dnr2 = read_candidate(sequence.dnr2_paths[index])
            dnr3 = read_candidate(sequence.dnr3_paths[index])
            ai = read_raw(result_root / f"out_{index:04d}.raw")
            panels = [
                label(to_bgr(noisy, gains, SOURCE_BLACK, exposure), "Noisy", sequence.name, index),
                label(to_bgr(dnr2, gains, NR_BLACK, exposure), "2DNR", sequence.name, index),
                label(to_bgr(dnr3, gains, NR_BLACK, exposure), "3DNR", sequence.name, index),
                label(to_bgr(ai, gains, NR_BLACK, exposure), "AI Fused", sequence.name, index),
            ]
            ai_writer.write(panels[3])
            small = [cv2.resize(panel, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA) for panel in panels]
            comparison_writer.write(np.concatenate([np.concatenate(small[:2], axis=1), np.concatenate(small[2:], axis=1)], axis=0))
    finally:
        ai_writer.close()
        comparison_writer.close()
    return {"name": sequence.name, "display_exposure": exposure, "videos": [verify_video(ai_path), verify_video(comparison_path)]}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sequences = discover_sequences(Path(config["data_root"]), tuple(config["sequence_names"]))
    if args.sequence != "all":
        sequences = tuple(item for item in sequences if item.name == args.sequence)
    if args.exposure_multiplier <= 0:
        raise ValueError("exposure-multiplier 必须大于 0")
    results = [render_sequence(sequence, args.output_root / sequence.name, args.exposure_multiplier) for sequence in sequences]
    (args.output_root / "video_manifest.json").write_text(json.dumps({"fps": FPS, "exposure_multiplier": args.exposure_multiplier, "sequences": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"视频已生成: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()

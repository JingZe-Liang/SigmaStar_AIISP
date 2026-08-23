from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


STAGE2_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STAGE2_ROOT.parents[2]
PHASE2_ROOT = WORKSPACE_ROOT / "Phase2"
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from dataset_io import discover_dataset  # noqa: E402


PANEL_WIDTH = 960
PANEL_HEIGHT = 540
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    width = max(170, 20 + 18 * len(label))
    cv2.rectangle(panel, (0, 0), (width, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        panel,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def make_motion_panel(base: np.ndarray, probability: np.ndarray) -> np.ndarray:
    panel = np.clip(base.astype(np.float32) * 0.28, 0, 255).astype(np.uint8)
    strength = probability.astype(np.float32) / 255.0
    overlay = np.zeros_like(panel)
    overlay[..., 2] = probability
    overlay[..., 1] = np.clip(probability.astype(np.float32) * 0.18, 0, 255).astype(np.uint8)
    weight = strength[..., None]
    return np.clip(panel * (1.0 - weight) + overlay * weight, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Stage2 predicted-motion/2DNR/3DNR/fusion four-view MP4"
    )
    parser.add_argument("--dataset-root", type=Path, default=PHASE2_ROOT / "DATASET")
    parser.add_argument(
        "--inference-root", type=Path, default=STAGE2_ROOT / "outputs" / "inference_final_all"
    )
    parser.add_argument(
        "--output-root", type=Path, default=STAGE2_ROOT / "outputs" / "four_view"
    )
    parser.add_argument("--sequences", nargs="+", default=["128x", "645x"])
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--crf", type=int, default=6)
    parser.add_argument("--preset", default="slow")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    catalog = discover_dataset(args.dataset_root)
    requested = set(args.sequences)
    sequences = [s for s in catalog.fusion_sequences if s.sequence_id in requested]
    if {s.sequence_id for s in sequences} != requested:
        raise ValueError(
            f"Unknown sequences: {sorted(requested - {s.sequence_id for s in sequences})}"
        )

    for sequence in sequences:
        sid = sequence.sequence_id
        inference_dir = args.inference_root / sid
        fusion_video = inference_dir / "fusion_isp_compatible.mp4"
        motion_path = inference_dir / "predicted_motion_u8.raw"
        expected_motion_bytes = sequence.frame_count * PANEL_HEIGHT * PANEL_WIDTH
        if not fusion_video.is_file():
            raise FileNotFoundError(f"Missing compatible fusion video: {fusion_video}")
        if not motion_path.is_file() or motion_path.stat().st_size != expected_motion_bytes:
            raise FileNotFoundError(f"Missing or invalid predicted motion stream: {motion_path}")

        motion = np.memmap(
            motion_path,
            dtype=np.uint8,
            mode="r",
            shape=(sequence.frame_count, PANEL_HEIGHT, PANEL_WIDTH),
        )
        fusion_capture = cv2.VideoCapture(str(fusion_video), cv2.CAP_FFMPEG)
        if not fusion_capture.isOpened():
            raise RuntimeError(f"Could not open fusion video: {fusion_video}")

        output_dir = args.output_root / sid
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "stage2_motion_2dnr_3dnr_fusion.mp4"
        partial = output_dir / "stage2_motion_2dnr_3dnr_fusion.partial.mp4"
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-v", "error",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
                "-framerate", str(args.fps),
                "-i", "pipe:0",
                "-an",
                "-vf", "format=yuv420p",
                "-c:v", "libx264",
                "-preset", args.preset,
                "-crf", str(args.crf),
                "-profile:v", "high",
                "-level:v", "4.1",
                "-tag:v", "avc1",
                "-color_range", "tv",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-movflags", "+faststart",
                "-y", str(partial),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if encoder.stdin is None:
            raise RuntimeError("Could not open FFmpeg input")

        try:
            for index in range(sequence.frame_count):
                d2 = cv2.imread(str(sequence.denoised_pngs[index]), cv2.IMREAD_COLOR)
                d3 = cv2.imread(str(sequence.fused_pngs[index]), cv2.IMREAD_COLOR)
                if d2 is None or d3 is None:
                    raise FileNotFoundError(f"Missing ISP PNG at frame {index} for {sid}")
                ok, fusion = fusion_capture.read()
                if not ok or fusion is None:
                    raise RuntimeError(f"Could not decode fusion frame {index} for {sid}")
                d2 = cv2.resize(d2, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
                d3 = cv2.resize(d3, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
                fusion = cv2.resize(fusion, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
                motion_panel = make_motion_panel(d2, motion[index])
                canvas = np.vstack(
                    [
                        np.hstack(
                            [label_panel(motion_panel, "PRED MOTION"), label_panel(d2, "2DNR")]
                        ),
                        np.hstack([label_panel(d3, "3DNR"), label_panel(fusion, "FUSION")]),
                    ]
                )
                encoder.stdin.write(np.ascontiguousarray(canvas).tobytes())
                if index == 100:
                    cv2.imwrite(str(output_dir / "frame_0100.png"), canvas)
                if (index + 1) % 25 == 0 or index + 1 == sequence.frame_count:
                    print(f"{sid}: {index + 1}/{sequence.frame_count}", flush=True)
        except Exception as error:
            try:
                encoder.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            stderr = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
            encoder.kill()
            encoder.wait()
            fusion_capture.release()
            raise RuntimeError(f"Four-view encoder stopped: {stderr}") from error

        fusion_capture.release()
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg encode failed ({return_code}): {stderr}")
        partial.replace(output)

        verify = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        if verify.returncode:
            raise RuntimeError(f"FFmpeg verification failed: {verify.stderr}")
        capture = cv2.VideoCapture(str(output), cv2.CAP_MSMF)
        if not capture.isOpened():
            raise RuntimeError(f"Windows Media Foundation cannot open {output}")
        decoded_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3):
                raise RuntimeError("Unexpected decoded four-view frame")
            decoded_frames += 1
        capture.release()
        if decoded_frames != sequence.frame_count:
            raise RuntimeError(
                f"Expected {sequence.frame_count} frames, decoded {decoded_frames}"
            )

        summary = {
            "sequence_id": sid,
            "output": str(output.resolve()),
            "layout": {
                "top_left": "Stage2 predicted motion overlay",
                "top_right": "2DNR",
                "bottom_left": "3DNR",
                "bottom_right": "Stage2 fusion",
            },
            "external_md_used": False,
            "codec": "H.264/AVC",
            "codec_tag": "avc1",
            "pixel_format": "yuv420p",
            "crf": args.crf,
            "fps": args.fps,
            "frame_count": sequence.frame_count,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "windows_media_foundation_decoded_frames": decoded_frames,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

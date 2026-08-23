from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from dataset_io import RawStreamReader, discover_dataset


PANEL_WIDTH = 960
PANEL_HEIGHT = 540
OUTPUT_WIDTH = PANEL_WIDTH * 2
OUTPUT_HEIGHT = PANEL_HEIGHT * 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (170, 38), (0, 0, 0), thickness=-1)
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


def make_md_panel(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    panel = np.clip(base.astype(np.float32) * 0.28, 0, 255).astype(np.uint8)
    panel[mask > 0] = (0, 0, 255)
    return panel


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create synchronized MD/2DNR/3DNR/fusion four-view MP4")
    parser.add_argument("--dataset-root", type=Path, default=root / "DATASET")
    parser.add_argument("--inference-root", type=Path, default=root / "DERIVED" / "inference_final_all")
    parser.add_argument("--output-root", type=Path, default=root / "DERIVED" / "four_view")
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
        raise ValueError(f"Unknown sequences: {sorted(requested - {s.sequence_id for s in sequences})}")

    for sequence in sequences:
        sid = sequence.sequence_id
        inference_dir = args.inference_root / sid
        md_path = args.inference_root.parent / "md_mog2" / sid / "md_mog2.raw"
        fusion_video = inference_dir / "fusion_isp_compatible.mp4"
        if not fusion_video.is_file():
            raise FileNotFoundError(f"Missing compatible fusion video: {fusion_video}")
        if not md_path.is_file():
            raise FileNotFoundError(f"Missing MD stream: {md_path}")

        d2_paths = sequence.denoised_pngs
        d3_paths = sequence.fused_pngs
        md = np.memmap(
            md_path,
            dtype=np.uint8,
            mode="r",
            shape=(sequence.frame_count, PANEL_HEIGHT, PANEL_WIDTH),
        )
        fusion_capture = cv2.VideoCapture(str(fusion_video), cv2.CAP_FFMPEG)
        if not fusion_capture.isOpened():
            raise RuntimeError(f"Could not open fusion video: {fusion_video}")

        output_dir = args.output_root / sid
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "md_2dnr_3dnr_fusion.mp4"
        partial = output_dir / "md_2dnr_3dnr_fusion.partial.mp4"
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
                "-framerate",
                str(args.fps),
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                args.preset,
                "-crf",
                str(args.crf),
                "-profile:v",
                "high",
                "-level:v",
                "4.1",
                "-tag:v",
                "avc1",
                "-color_range",
                "tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-movflags",
                "+faststart",
                "-y",
                str(partial),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if encoder.stdin is None:
            raise RuntimeError("Could not open FFmpeg input")

        try:
            for index in range(sequence.frame_count):
                d2 = cv2.imread(str(d2_paths[index]), cv2.IMREAD_COLOR)
                d3 = cv2.imread(str(d3_paths[index]), cv2.IMREAD_COLOR)
                if d2 is None or d3 is None:
                    raise FileNotFoundError(f"Missing ISP PNG at frame {index} for {sid}")
                ok, fusion = fusion_capture.read()
                if not ok or fusion is None:
                    raise RuntimeError(f"Could not decode fusion frame {index} for {sid}")
                d2 = cv2.resize(d2, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
                d3 = cv2.resize(d3, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
                fusion = cv2.resize(fusion, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
                md_panel = make_md_panel(d2, md[index])
                canvas = np.vstack(
                    [
                        np.hstack([label_panel(md_panel, "MD"), label_panel(d2, "2DNR")]),
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
            raise RuntimeError(f"Four-view verification failed: {verify.stderr}")
        capture = cv2.VideoCapture(str(output), cv2.CAP_MSMF)
        if not capture.isOpened():
            raise RuntimeError(f"Windows Media Foundation cannot open {output}")
        decoded_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3):
                raise RuntimeError(f"Unexpected four-view shape: {frame.shape}")
            decoded_frames += 1
        capture.release()
        if decoded_frames != sequence.frame_count:
            raise RuntimeError(f"Expected {sequence.frame_count} frames, decoded {decoded_frames}")

        summary = {
            "sequence_id": sid,
            "output": str(output.resolve()),
            "layout": {
                "top_left": "MD overlay",
                "top_right": "2DNR",
                "bottom_left": "3DNR",
                "bottom_right": "fusion",
            },
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

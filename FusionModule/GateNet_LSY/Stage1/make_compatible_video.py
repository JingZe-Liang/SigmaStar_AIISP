from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_with_opencv(path: Path, backend: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path), backend)
    opened = capture.isOpened()
    result: dict[str, object] = {"opened": opened}
    if opened:
        result.update(
            {
                "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            }
        )
        decoded_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != (1080, 1920, 3):
                raise RuntimeError(f"Invalid decoded frame shape: {None if frame is None else frame.shape}")
            decoded_frames += 1
        result["decoded_frames"] = decoded_frames
    capture.release()
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create broadly playable H.264 MP4 copies")
    parser.add_argument("--root", type=Path, default=root / "DERIVED" / "inference_final_all")
    parser.add_argument("--sequences", nargs="+", default=["128x", "645x"])
    parser.add_argument("--crf", type=int, default=6)
    parser.add_argument("--preset", default="slow")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for sequence_id in args.sequences:
        directory = args.root / sequence_id
        master_candidates = [
            directory / "fusion_isp_master_12bit.mp4",
            directory / "fusion_isp.mp4",
        ]
        master = next((path for path in master_candidates if path.is_file()), None)
        if master is None:
            raise FileNotFoundError(f"No ISP master found for {sequence_id}")
        output = directory / "fusion_isp_compatible.mp4"
        partial = directory / "fusion_isp_compatible.partial.mp4"
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(master),
            "-an",
            "-sws_flags",
            "lanczos+accurate_rnd+full_chroma_int",
            "-vf",
            "scale=in_range=pc:out_range=tv,format=yuv420p",
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
        ]
        print(f"{sequence_id}: creating compatible H.264 MP4", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"FFmpeg failed ({completed.returncode}): {completed.stderr}")
        partial.replace(output)

        ffmpeg_verify = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        if ffmpeg_verify.returncode:
            raise RuntimeError(f"FFmpeg verification failed: {ffmpeg_verify.stderr}")
        opencv_ffmpeg = verify_with_opencv(output, cv2.CAP_FFMPEG)
        opencv_msmf = verify_with_opencv(output, cv2.CAP_MSMF)
        if not opencv_ffmpeg.get("opened") or opencv_ffmpeg.get("decoded_frames") != 200:
            raise RuntimeError(f"OpenCV FFmpeg verification failed: {opencv_ffmpeg}")

        summary = {
            "sequence_id": sequence_id,
            "source_master": str(master.resolve()),
            "output": str(output.resolve()),
            "codec": "H.264/AVC",
            "codec_tag": "avc1",
            "profile": "High",
            "level": "4.1",
            "pixel_format": "yuv420p",
            "bit_depth": 8,
            "crf": args.crf,
            "preset": args.preset,
            "faststart": True,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "ffmpeg_full_decode": True,
            "opencv_ffmpeg": opencv_ffmpeg,
            "opencv_windows_msmf": opencv_msmf,
        }
        (directory / "compatible_video_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

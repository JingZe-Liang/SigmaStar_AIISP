from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from dataset_io import RawStreamReader, discover_dataset


# OpenCV names Bayer conversion codes by the resulting channel order. These
# mappings are verified against the dataset's rendered PNG reference frames.
CFA_TO_BGR = {
    "RGGB": cv2.COLOR_BayerBG2BGR,
    "BGGR": cv2.COLOR_BayerRG2BGR,
    "GBRG": cv2.COLOR_BayerGR2BGR,
    "GRBG": cv2.COLOR_BayerGB2BGR,
}


def cfa_slices(cfa_pattern: str) -> dict[str, tuple[slice, slice]]:
    labels: list[str] = []
    green_index = 0
    for color in cfa_pattern:
        if color == "G":
            green_index += 1
            labels.append(f"G{green_index}")
        else:
            labels.append(color)
    return {
        label: (slice(index // 2, None, 2), slice(index % 2, None, 2))
        for index, label in enumerate(labels)
    }


class FixedSequenceISP:
    def __init__(
        self,
        *,
        cfa_pattern: str,
        black_level: float,
        white_level: float,
        r_gain: float,
        b_gain: float,
        exposure: float,
        gamma: float = 2.2,
    ) -> None:
        self.cfa_pattern = cfa_pattern
        self.black_level = black_level
        self.white_level = white_level
        self.r_gain = r_gain
        self.b_gain = b_gain
        self.exposure = exposure
        self.gamma = gamma
        self.slices = cfa_slices(cfa_pattern)

    def linear_bgr(self, raw: np.ndarray) -> np.ndarray:
        mosaic = np.clip(
            (raw.astype(np.float32) - self.black_level)
            / (self.white_level - self.black_level),
            0.0,
            1.0,
        )
        mosaic[self.slices["R"]] *= self.r_gain
        mosaic[self.slices["B"]] *= self.b_gain
        mosaic_u16 = np.clip(np.rint(mosaic * 65535.0), 0, 65535).astype(np.uint16)
        return cv2.cvtColor(mosaic_u16, CFA_TO_BGR[self.cfa_pattern]).astype(np.float32) / 65535.0

    def process_12bit(self, raw: np.ndarray) -> np.ndarray:
        linear = np.clip(self.linear_bgr(raw) * self.exposure, 0.0, 1.0)
        display = np.power(linear, 1.0 / self.gamma)
        return np.clip(np.rint(display * 4095.0), 0, 4095).astype("<u2")


def calibrate_exposure(
    sequence,
    isp: FixedSequenceISP,
    frame_indices: list[int],
    *,
    spatial_step: int = 12,
) -> tuple[float, float]:
    reader = RawStreamReader(sequence.denoised)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_index in frame_indices:
        linear = isp.linear_bgr(reader.read_frame(frame_index))[::spatial_step, ::spatial_step]
        reference = cv2.imread(
            str(sequence.denoised_pngs[frame_index]), cv2.IMREAD_COLOR
        )
        if reference is None:
            raise FileNotFoundError(sequence.denoised_pngs[frame_index])
        samples.append((linear, reference[::spatial_step, ::spatial_step].astype(np.float32) / 255.0))

    best_error = float("inf")
    best_exposure = 1.0
    for exposure in np.geomspace(0.25, 16.0, 97):
        errors = [
            np.mean(
                np.abs(
                    np.power(np.clip(linear * exposure, 0.0, 1.0), 1.0 / isp.gamma)
                    - reference
                )
            )
            for linear, reference in samples
        ]
        error = float(np.mean(errors))
        if error < best_error:
            best_error = error
            best_exposure = float(exposure)
    return best_exposure, best_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def planar_gbr12_bytes(frame_bgr: np.ndarray) -> bytes:
    """Serialize a BGR image as FFmpeg's planar little-endian gbrp12le."""
    return b"".join(
        np.ascontiguousarray(frame_bgr[..., channel], dtype="<u2").tobytes()
        for channel in (1, 0, 2)
    )


def decoded_raw_hash(ffmpeg: str, video_path: Path) -> tuple[str, int]:
    process = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gbrp12le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg decoder output")
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := process.stdout.read(1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg decode failed ({return_code}): {stderr}")
    return digest.hexdigest(), byte_count


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render inferred Bayer RAW streams through ISP to MP4")
    parser.add_argument("--dataset-root", type=Path, default=root / "DATASET")
    parser.add_argument("--input", type=Path, default=root / "DERIVED" / "inference_final_all")
    parser.add_argument("--sequences", nargs="+", default=["128x", "645x"])
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--black-level", type=float, default=300.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--preset", default="fast")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = discover_dataset(args.dataset_root)
    requested = set(args.sequences)
    sequences = [s for s in catalog.fusion_sequences if s.sequence_id in requested]
    if {s.sequence_id for s in sequences} != requested:
        raise ValueError(f"Unknown sequence IDs: {sorted(requested - {s.sequence_id for s in sequences})}")

    for sequence in sequences:
        sequence_dir = args.input / sequence.sequence_id
        raw_path = sequence_dir / "fusion.raw"
        expected_bytes = sequence.frame_count * sequence.source.height * sequence.source.width * 2
        if not raw_path.is_file() or raw_path.stat().st_size != expected_bytes:
            raise FileNotFoundError(f"Missing or invalid inferred RAW stream: {raw_path}")
        if len(sequence.denoised_pngs) != sequence.frame_count:
            raise RuntimeError(f"Reference PNG sequence is incomplete: {sequence.sequence_id}")

        metadata = sequence.source.metadata
        isp = FixedSequenceISP(
            cfa_pattern=sequence.source.cfa_pattern,
            black_level=args.black_level,
            white_level=args.white_level,
            r_gain=float(metadata["r_gain"]) / float(metadata["g_gain"]),
            b_gain=float(metadata["b_gain"]) / float(metadata["g_gain"]),
            exposure=1.0,
            gamma=args.gamma,
        )
        calibration_frames = [23, sequence.frame_count // 2, sequence.frame_count - 4]
        isp.exposure, calibration_mae = calibrate_exposure(
            sequence, isp, calibration_frames
        )

        output_path = sequence_dir / "fusion_isp_master_12bit.mp4"
        partial_path = sequence_dir / "fusion_isp_master_12bit.partial.mp4"
        preview_path = sequence_dir / "preview_frame_0100.png"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "gbrp12le",
                "-video_size",
                f"{sequence.source.width}x{sequence.source.height}",
                "-framerate",
                str(args.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx265",
                "-preset",
                args.preset,
                "-x265-params",
                "lossless=1:log-level=error",
                "-pix_fmt",
                "gbrp12le",
                "-tag:v",
                "hvc1",
                "-movflags",
                "+faststart",
                "-y",
                str(partial_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if encoder.stdin is None:
            raise RuntimeError("Could not open FFmpeg encoder input")

        raw_stream = np.memmap(
            raw_path,
            dtype="<u2",
            mode="r",
            shape=(sequence.frame_count, sequence.source.height, sequence.source.width),
        )
        started = time.perf_counter()
        source_digest = hashlib.sha256()
        source_byte_count = 0
        print(
            f"{sequence.sequence_id}: exposure={isp.exposure:.4f}, "
            f"calibration_mae={calibration_mae:.4f}",
            flush=True,
        )
        try:
            for frame_index in range(sequence.frame_count):
                frame = isp.process_12bit(raw_stream[frame_index])
                frame_bytes = planar_gbr12_bytes(frame)
                encoder.stdin.write(frame_bytes)
                source_digest.update(frame_bytes)
                source_byte_count += len(frame_bytes)
                if frame_index == 100:
                    preview = np.clip(
                        np.rint(frame.astype(np.float32) * (255.0 / 4095.0)), 0, 255
                    ).astype(np.uint8)
                    if not cv2.imwrite(str(preview_path), preview):
                        raise RuntimeError(f"Could not write preview: {preview_path}")
                if (frame_index + 1) % 25 == 0 or frame_index + 1 == sequence.frame_count:
                    print(f"  {frame_index + 1}/{sequence.frame_count}", flush=True)
        except Exception:
            encoder.stdin.close()
            encoder.kill()
            encoder.wait()
            raise
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg encode failed ({return_code}): {stderr}")
        partial_path.replace(output_path)

        decoded_hash, decoded_byte_count = decoded_raw_hash(ffmpeg, output_path)
        encoded_source_hash = source_digest.hexdigest()
        lossless_verified = (
            decoded_hash == encoded_source_hash and decoded_byte_count == source_byte_count
        )
        if not lossless_verified:
            raise RuntimeError(
                "Lossless verification failed: decoded 12-bit frames differ from encoder input"
            )

        summary = {
            "sequence_id": sequence.sequence_id,
            "input_raw": str(raw_path.resolve()),
            "output_video": output_path.name,
            "codec": "HEVC/libx265 lossless",
            "codec_tag": "hvc1",
            "pixel_format": "gbrp12le",
            "chroma_subsampling": "none (RGB 4:4:4)",
            "preset": args.preset,
            "fps": args.fps,
            "duration_seconds": sequence.frame_count / args.fps,
            "frame_count": sequence.frame_count,
            "width": sequence.source.width,
            "height": sequence.source.height,
            "cfa_pattern": sequence.source.cfa_pattern,
            "black_level": args.black_level,
            "white_level": args.white_level,
            "r_gain": isp.r_gain,
            "b_gain": isp.b_gain,
            "exposure": isp.exposure,
            "gamma": isp.gamma,
            "calibration_frames": calibration_frames,
            "calibration_mae_0_to_1": calibration_mae,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "encoded_frame_bytes": source_byte_count,
            "encoded_frame_sha256": encoded_source_hash,
            "decoded_frame_bytes": decoded_byte_count,
            "decoded_frame_sha256": decoded_hash,
            "lossless_verified": lossless_verified,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (sequence_dir / "isp_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

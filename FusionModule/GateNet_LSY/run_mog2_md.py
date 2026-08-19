from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from dataset_io import RawStreamReader, discover_dataset


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _green_plane(raw12: np.ndarray, cfa_pattern: str) -> np.ndarray:
    if cfa_pattern != "RGGB":
        raise ValueError(f"MOG2 output generation currently expects RGGB, got {cfa_pattern}")
    green_1 = raw12[0::2, 1::2].astype(np.uint32)
    green_2 = raw12[1::2, 0::2].astype(np.uint32)
    return ((green_1 + green_2) // 2).astype(np.uint16)


def _validate_outputs(
    stream_path: Path,
    mask_dir: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> None:
    expected_size = frame_count * width * height
    actual_size = stream_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"MD stream size mismatch: expected {expected_size}, got {actual_size}"
        )
    pngs = sorted(mask_dir.glob("out_*.png"))
    if len(pngs) != frame_count:
        raise RuntimeError(f"Expected {frame_count} PNG masks, found {len(pngs)}")
    for expected_index, path in enumerate(pngs):
        if path.stem != f"out_{expected_index:04d}":
            raise RuntimeError(f"Non-contiguous MD mask sequence at {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (height, width) or image.dtype != np.uint8:
            raise RuntimeError(f"Invalid MD PNG: {path}")
        values = np.unique(image)
        if not np.all(np.isin(values, (0, 255))):
            raise RuntimeError(f"Non-binary MD PNG: {path}, values={values.tolist()}")


def run_sequence(
    sequence,
    output_root: Path,
    *,
    history: int,
    var_threshold: float,
    warmup_frames: int,
    median_kernel: int,
    overwrite: bool,
) -> dict:
    output_dir = output_root / sequence.sequence_id
    temporary_dir = output_root / f".{sequence.sequence_id}.tmp"
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    mask_dir = temporary_dir / "masks"
    mask_dir.mkdir(parents=True)

    reader = RawStreamReader(sequence.denoised)
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False,
    )
    width = sequence.denoised.width // 2
    height = sequence.denoised.height // 2
    stream_path = temporary_dir / "md_mog2.raw"
    statistics: list[dict] = []

    with stream_path.open("wb") as stream:
        for frame_index in range(sequence.frame_count):
            denoised = reader.read_frame(frame_index)
            green = _green_plane(denoised, sequence.denoised.cfa_pattern)
            mask = subtractor.apply(green, learningRate=-1)
            if median_kernel > 1:
                mask = cv2.medianBlur(mask, median_kernel)
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            stream.write(mask.tobytes(order="C"))
            png_path = mask_dir / f"out_{frame_index:04d}.png"
            if not cv2.imwrite(str(png_path), mask):
                raise RuntimeError(f"Failed to write {png_path}")
            statistics.append(
                {
                    "frame_index": frame_index,
                    "training_valid": frame_index >= warmup_frames,
                    "foreground_fraction": float(np.count_nonzero(mask) / mask.size),
                }
            )

    stats_path = temporary_dir / "foreground_fraction.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_index", "training_valid", "foreground_fraction"),
        )
        writer.writeheader()
        writer.writerows(statistics)

    valid_fractions = [
        row["foreground_fraction"] for row in statistics if row["training_valid"]
    ]
    metadata = {
        "algorithm": "OpenCV BackgroundSubtractorMOG2",
        "opencv_version": cv2.__version__,
        "sequence_id": sequence.sequence_id,
        "input_role": "2dnr",
        "input_path": str(sequence.denoised.path),
        "input_encoding": sequence.denoised.encoding,
        "input_cfa_pattern": sequence.denoised.cfa_pattern,
        "input_plane": "integer mean of RGGB G1 and G2",
        "output_encoding": "uint8 binary mask, background=0, foreground=255",
        "output_shape": [sequence.frame_count, height, width],
        "history": history,
        "var_threshold": var_threshold,
        "detect_shadows": False,
        "learning_rate": "OpenCV automatic (-1)",
        "median_kernel": median_kernel,
        "warmup_frames": warmup_frames,
        "training_frame_range": [warmup_frames, sequence.frame_count - 1],
        "valid_foreground_fraction": {
            "mean": float(np.mean(valid_fractions)),
            "median": float(np.median(valid_fractions)),
            "p95": float(np.percentile(valid_fractions, 95)),
            "max": float(np.max(valid_fractions)),
        },
    }
    metadata_path = temporary_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _validate_outputs(
        stream_path,
        mask_dir,
        frame_count=sequence.frame_count,
        width=width,
        height=height,
    )
    metadata["stream_sha256"] = _sha256(stream_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_dir, output_dir)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MOG2 MD masks from paired 2DNR")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("DERIVED/md_mog2"))
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--var-threshold", type=float, default=64.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--median-kernel", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.history <= 0:
        parser.error("--history must be positive")
    if args.var_threshold <= 0:
        parser.error("--var-threshold must be positive")
    if args.median_kernel <= 0 or args.median_kernel % 2 == 0:
        parser.error("--median-kernel must be a positive odd integer")

    catalog = discover_dataset(args.dataset_root)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sequence in catalog.fusion_sequences:
        if args.warmup_frames >= sequence.frame_count:
            parser.error(
                f"--warmup-frames must be below {sequence.frame_count} for {sequence.sequence_id}"
            )
        summaries.append(
            run_sequence(
                sequence,
                args.output,
                history=args.history,
                var_threshold=args.var_threshold,
                warmup_frames=args.warmup_frames,
                median_kernel=args.median_kernel,
                overwrite=args.overwrite,
            )
        )
    print(
        json.dumps(
            [
                {
                    "sequence_id": summary["sequence_id"],
                    **summary["valid_foreground_fraction"],
                    "stream_sha256": summary["stream_sha256"],
                }
                for summary in summaries
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

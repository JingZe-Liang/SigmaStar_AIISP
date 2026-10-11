#!/usr/bin/env python3
"""Diagnose the 1000x A4 input mapping and registration on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.stream_io import open_stream, pack_frame


CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
PACKED_HEIGHT = 540
PACKED_WIDTH = 960
WHITE_LEVEL = 4095.0


def _stats(values: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "p01": float(np.percentile(flattened, 1)),
        "p50": float(np.percentile(flattened, 50)),
        "p99": float(np.percentile(flattened, 99)),
    }


def _channel_stats(values: np.ndarray) -> dict[str, dict[str, float]]:
    return {name: _stats(values[index]) for index, name in enumerate(CHANNEL_NAMES)}


def _normalise_dn(values: np.ndarray, black: np.ndarray) -> np.ndarray:
    denominator = WHITE_LEVEL - black[:, None, None]
    return np.clip((values - black[:, None, None]) / denominator, 0.0, 1.0)


def _align(values: np.ndarray, shift_x: int, shift_y: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = values.shape[-2:]
    y0, y1 = max(0, -shift_y), min(height, height - shift_y)
    x0, x1 = max(0, -shift_x), min(width, width - shift_x)
    aligned = np.zeros_like(values)
    overlap = np.zeros((height, width), dtype=bool)
    aligned[..., y0:y1, x0:x1] = values[..., y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
    overlap[y0:y1, x0:x1] = True
    return aligned, overlap


def _geometric_overlap(width: int, height: int, shift_x: float, shift_y: float) -> float:
    return max(0.0, width - abs(shift_x)) * max(0.0, height - abs(shift_y)) / (width * height)


def diagnose(supervision_dir: Path, max_frames: int, sample_stride: int) -> dict[str, object]:
    manifest_path = supervision_dir / "stream_manifests" / "04_shdarkroom_1000x.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)

    report = json.loads((supervision_dir / "references" / "a1_reference_report.json").read_text(encoding="utf-8"))
    reference = np.load(supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy", mmap_mode="r").astype(np.float32, copy=False)
    valid_region = np.load(supervision_dir / "references" / "04_shdarkroom_10x_valid_region.npy", mmap_mode="r").astype(bool, copy=False)
    motion = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")[:max_frames, 0, ::2, ::2]

    black = np.asarray([report["black_levels"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    fit = manifest["exposure_fit"]
    slopes = np.asarray([fit["slopes"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    intercepts = np.asarray([fit["intercepts"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    registration = manifest["registration"]
    shift_x = int(registration["integer_shift_x"])
    shift_y = int(registration["integer_shift_y"])
    frames = min(max_frames, int(manifest["frames_used"]), int(reader.frames), int(motion.shape[0]))
    sample_frame_stride = max(1, frames // 32)

    raw_samples: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    mapped_samples: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    normalized_samples: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    residual_values: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    reference_values: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    raw_sum = np.zeros(4, dtype=np.float64)
    mapped_sum = np.zeros(4, dtype=np.float64)
    normalized_sum = np.zeros(4, dtype=np.float64)
    sample_count = 0
    static_count = np.zeros(4, dtype=np.int64)
    overlap_count = 0

    for frame_index in range(frames):
        packed = pack_frame(reader.read_frame(frame_index), adapter).astype(np.float32)
        mapped = slopes[:, None, None] * packed + intercepts[:, None, None]
        normalized = _normalise_dn(mapped, black)
        aligned_mapped, overlap = _align(mapped, shift_x, shift_y)
        aligned_normalized, _ = _align(normalized, shift_x, shift_y)
        aligned_motion, _ = _align(motion[frame_index], shift_x, shift_y)
        static = overlap & (aligned_motion < 0.5) & valid_region
        static &= np.all(aligned_normalized < ((3800.0 - black) / (WHITE_LEVEL - black))[:, None, None], axis=0)
        target = np.broadcast_to(reference, aligned_mapped.shape)
        residual = aligned_mapped - target

        if frame_index % sample_frame_stride == 0:
            sampled = (slice(None), slice(None, None, sample_stride), slice(None, None, sample_stride))
            raw_sample = packed[sampled]
            mapped_sample = mapped[sampled]
            normalized_sample = normalized[sampled]
            static_sample = static[::sample_stride, ::sample_stride]
            for index in range(4):
                raw_samples[index].append(raw_sample[index].reshape(-1))
                mapped_samples[index].append(mapped_sample[index].reshape(-1))
                normalized_samples[index].append(normalized_sample[index].reshape(-1))
                residual_values[index].append(residual[index, ::sample_stride, ::sample_stride][static_sample])
                reference_values[index].append(target[index, ::sample_stride, ::sample_stride][static_sample])
        raw_sum += packed.reshape(4, -1).sum(axis=1, dtype=np.float64)
        mapped_sum += mapped.reshape(4, -1).sum(axis=1, dtype=np.float64)
        normalized_sum += normalized.reshape(4, -1).sum(axis=1, dtype=np.float64)
        sample_count += packed.shape[1] * packed.shape[2]
        static_count += np.full(4, np.count_nonzero(static), dtype=np.int64)
        overlap_count += int(np.count_nonzero(overlap))

    def joined(collection: list[list[np.ndarray]]) -> np.ndarray:
        return np.asarray([np.concatenate(values) for values in collection])

    raw_values = joined(raw_samples)
    mapped_values = joined(mapped_samples)
    normalized_values = joined(normalized_samples)
    residual_arrays = joined(residual_values)
    reference_arrays = joined(reference_values)
    overlap_ratio = overlap_count / (frames * PACKED_HEIGHT * PACKED_WIDTH)
    return {
        "gain": "1000x",
        "frames": frames,
        "sample_stride": sample_stride,
        "sample_frame_stride": sample_frame_stride,
        "black_levels_dn": dict(zip(CHANNEL_NAMES, black.tolist())),
        "exposure_fit": {"slopes": dict(zip(CHANNEL_NAMES, slopes.tolist())), "intercepts": dict(zip(CHANNEL_NAMES, intercepts.tolist()))},
        "registration": {
            "float_shift_sensor_px": {
                "x": float(registration["shift_x"]),
                "y": float(registration["shift_y"]),
            },
            "float_shift_packed_px": {
                "x": float(registration["shift_x"]) / 2.0,
                "y": float(registration["shift_y"]) / 2.0,
            },
            "integer_shift_x": shift_x,
            "integer_shift_y": shift_y,
            "ncc": float(registration["ncc"]),
            "pixel_overlap_ratio": float(overlap_ratio),
            "geometric_float_overlap_ratio": _geometric_overlap(
                PACKED_WIDTH,
                PACKED_HEIGHT,
                float(registration["shift_x"]) / 2.0,
                float(registration["shift_y"]) / 2.0,
            ),
        },
        "raw_packed_dn": _channel_stats(raw_values),
        "exposure_mapped_dn": _channel_stats(mapped_values),
        "normalized": _channel_stats(normalized_values),
        "mean_dn_all_pixels": {
            "raw_packed": dict(zip(CHANNEL_NAMES, (raw_sum / sample_count).tolist())),
            "exposure_mapped": dict(zip(CHANNEL_NAMES, (mapped_sum / sample_count).tolist())),
            "normalized": dict(zip(CHANNEL_NAMES, (normalized_sum / sample_count).tolist())),
        },
        "reference_dn": _channel_stats(reference),
        "static_region": {
            "pixel_count": dict(zip(CHANNEL_NAMES, static_count.tolist())),
            "residual_dn": _channel_stats(residual_arrays),
            "reference_dn": _channel_stats(reference_arrays),
            "static_fraction_of_frame": float(np.mean(static_count / (frames * PACKED_HEIGHT * PACKED_WIDTH))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.max_frames < 1 or args.sample_stride < 1:
        raise ValueError("max-frames and sample-stride must be positive")
    result = diagnose(args.supervision_dir, args.max_frames, args.sample_stride)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

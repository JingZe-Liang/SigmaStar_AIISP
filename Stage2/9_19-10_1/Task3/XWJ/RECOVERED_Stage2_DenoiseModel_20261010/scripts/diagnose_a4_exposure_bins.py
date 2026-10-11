#!/usr/bin/env python3
"""Compare exposure-fit residuals by reference intensity for A4 gains."""

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
BIN_EDGES = np.asarray((0.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0))


def _normalise_dn(values: np.ndarray, black: np.ndarray) -> np.ndarray:
    return np.clip(
        (values - black[:, None, None]) / (WHITE_LEVEL - black[:, None, None]),
        0.0,
        1.0,
    )


def _align(values: np.ndarray, shift_x: int, shift_y: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = values.shape[-2:]
    y0, y1 = max(0, -shift_y), min(height, height - shift_y)
    x0, x1 = max(0, -shift_x), min(width, width - shift_x)
    aligned = np.zeros_like(values)
    overlap = np.zeros((height, width), dtype=bool)
    aligned[..., y0:y1, x0:x1] = values[..., y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
    overlap[y0:y1, x0:x1] = True
    return aligned, overlap


def _empty_accumulators() -> dict[str, np.ndarray]:
    shape = (len(CHANNEL_NAMES), len(BIN_EDGES) - 1)
    return {
        "count": np.zeros(shape, dtype=np.float64),
        "reference_sum": np.zeros(shape, dtype=np.float64),
        "mapped_sum": np.zeros(shape, dtype=np.float64),
        "residual_sum": np.zeros(shape, dtype=np.float64),
        "residual_square_sum": np.zeros(shape, dtype=np.float64),
        "reference_square_sum": np.zeros(shape, dtype=np.float64),
        "reference_mapped_sum": np.zeros(shape, dtype=np.float64),
    }


def _update_channel_bins(
    accumulators: dict[str, np.ndarray],
    channel: int,
    reference: np.ndarray,
    mapped: np.ndarray,
    selected: np.ndarray,
) -> None:
    reference_values = reference[selected].astype(np.float64, copy=False)
    mapped_values = mapped[selected].astype(np.float64, copy=False)
    residual_values = mapped_values - reference_values
    labels = np.digitize(reference_values, BIN_EDGES, right=False) - 1
    valid_labels = (labels >= 0) & (labels < len(BIN_EDGES) - 1)
    labels = labels[valid_labels]
    reference_values = reference_values[valid_labels]
    mapped_values = mapped_values[valid_labels]
    residual_values = residual_values[valid_labels]
    weights = lambda values: np.bincount(labels, weights=values, minlength=len(BIN_EDGES) - 1)
    accumulators["count"][channel] += weights(np.ones_like(reference_values))
    accumulators["reference_sum"][channel] += weights(reference_values)
    accumulators["mapped_sum"][channel] += weights(mapped_values)
    accumulators["residual_sum"][channel] += weights(residual_values)
    accumulators["residual_square_sum"][channel] += weights(residual_values * residual_values)
    accumulators["reference_square_sum"][channel] += weights(reference_values * reference_values)
    accumulators["reference_mapped_sum"][channel] += weights(reference_values * mapped_values)


def _finalize_bins(accumulators: dict[str, np.ndarray]) -> dict[str, list[dict[str, float | int | None]]]:
    result: dict[str, list[dict[str, float | int | None]]] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        rows: list[dict[str, float | int | None]] = []
        for bin_index in range(len(BIN_EDGES) - 1):
            count = accumulators["count"][channel, bin_index]
            if count == 0:
                rows.append({"lower_dn": int(BIN_EDGES[bin_index]), "upper_dn": int(BIN_EDGES[bin_index + 1]), "count": 0})
                continue
            reference_sum = accumulators["reference_sum"][channel, bin_index]
            mapped_sum = accumulators["mapped_sum"][channel, bin_index]
            residual_sum = accumulators["residual_sum"][channel, bin_index]
            residual_square_sum = accumulators["residual_square_sum"][channel, bin_index]
            reference_square_sum = accumulators["reference_square_sum"][channel, bin_index]
            reference_mapped_sum = accumulators["reference_mapped_sum"][channel, bin_index]
            denominator = count * reference_square_sum - reference_sum * reference_sum
            slope = (count * reference_mapped_sum - reference_sum * mapped_sum) / denominator if denominator > 0 else None
            intercept = (mapped_sum - slope * reference_sum) / count if slope is not None else None
            residual_mean = residual_sum / count
            residual_variance = max(0.0, residual_square_sum / count - residual_mean * residual_mean)
            rows.append({
                "lower_dn": int(BIN_EDGES[bin_index]),
                "upper_dn": int(BIN_EDGES[bin_index + 1]),
                "count": int(count),
                "reference_mean_dn": float(reference_sum / count),
                "mapped_mean_dn": float(mapped_sum / count),
                "residual_mean_dn": float(residual_mean),
                "residual_std_dn": float(np.sqrt(residual_variance)),
                "fit_slope": None if slope is None else float(slope),
                "fit_intercept_dn": None if intercept is None else float(intercept),
            })
        result[name] = rows
    return result


def diagnose_gain(supervision_dir: Path, gain: str, max_frames: int) -> dict[str, object]:
    manifest = json.loads((supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json").read_text(encoding="utf-8"))
    report = json.loads((supervision_dir / "references" / "a1_reference_report.json").read_text(encoding="utf-8"))
    reference = np.load(supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy", mmap_mode="r").astype(np.float32, copy=False)
    valid_region = np.load(supervision_dir / "references" / "04_shdarkroom_10x_valid_region.npy", mmap_mode="r").astype(bool, copy=False)
    motion = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")[:max_frames, 0, ::2, ::2]
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)

    black = np.asarray([report["black_levels"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    fit = manifest["exposure_fit"]
    slopes = np.asarray([fit["slopes"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    intercepts = np.asarray([fit["intercepts"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    registration = manifest["registration"]
    shift_x = int(registration["integer_shift_x"])
    shift_y = int(registration["integer_shift_y"])
    frames = min(max_frames, int(manifest["frames_used"]), int(reader.frames), int(motion.shape[0]))
    accumulators = _empty_accumulators()
    static_pixels = 0
    overlap_pixels = 0

    for frame_index in range(frames):
        packed = pack_frame(reader.read_frame(frame_index), adapter).astype(np.float32)
        mapped = slopes[:, None, None] * packed + intercepts[:, None, None]
        normalized = _normalise_dn(mapped, black)
        aligned_mapped, overlap = _align(mapped, shift_x, shift_y)
        aligned_normalized, _ = _align(normalized, shift_x, shift_y)
        aligned_motion, _ = _align(motion[frame_index], shift_x, shift_y)
        static = overlap & (aligned_motion < 0.5) & valid_region
        static &= np.all(aligned_normalized < ((3800.0 - black) / (WHITE_LEVEL - black))[:, None, None], axis=0)
        for channel in range(len(CHANNEL_NAMES)):
            _update_channel_bins(accumulators, channel, reference[channel], aligned_mapped[channel], static)
        static_pixels += int(np.count_nonzero(static))
        overlap_pixels += int(np.count_nonzero(overlap))

    return {
        "gain": gain,
        "frames": frames,
        "registration": {
            "integer_shift_x": shift_x,
            "integer_shift_y": shift_y,
            "float_shift_sensor_px": {"x": float(registration["shift_x"]), "y": float(registration["shift_y"])},
            "float_shift_packed_px": {"x": float(registration["shift_x"]) / 2.0, "y": float(registration["shift_y"]) / 2.0},
            "ncc": float(registration["ncc"]),
            "overlap_ratio": float(overlap_pixels / (frames * PACKED_HEIGHT * PACKED_WIDTH)),
        },
        "static_fraction": float(static_pixels / (frames * PACKED_HEIGHT * PACKED_WIDTH)),
        "bins": _finalize_bins(accumulators),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--gains", default="1000x,645x")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/exposure_fit_diagnosis.json"))
    args = parser.parse_args()
    if args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    result = {"bin_edges_dn": BIN_EDGES.tolist(), "gains": {}}
    for gain in (value.strip() for value in args.gains.split(",")):
        if gain:
            result["gains"][gain] = diagnose_gain(args.supervision_dir, gain, args.max_frames)
    payload = json.dumps(result, indent=2) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

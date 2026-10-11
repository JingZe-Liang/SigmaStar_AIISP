#!/usr/bin/env python3
"""Diagnose the direct 1000x-to-645x exposure relationship on CPU."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.exposure import fit_exposure_pair
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, open_stream, pack_frame


PACKED_HEIGHT = 540
PACKED_WIDTH = 960
WHITE_LEVEL = 4095.0
BIN_EDGES = np.asarray((0.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0))


def _load_manifest(supervision_dir: Path, gain: str) -> dict:
    return json.loads((supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json").read_text(encoding="utf-8"))


def _temporal_mean(manifest: dict, max_frames: int) -> np.ndarray:
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames = min(max_frames, int(manifest["frames_used"]), int(reader.frames))
    total = np.zeros((4, PACKED_HEIGHT, PACKED_WIDTH), dtype=np.float64)
    for frame_index in range(frames):
        total += pack_frame(reader.read_frame(frame_index), adapter).astype(np.float64)
    return (total / frames).astype(np.float32)


def _overlap_pair(
    baseline: np.ndarray,
    moving: np.ndarray,
    shift_x: int,
    shift_y: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = baseline.shape[-2:]
    y0, y1 = max(0, -shift_y), min(height, height - shift_y)
    x0, x1 = max(0, -shift_x), min(width, width - shift_x)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"shift ({shift_x}, {shift_y}) leaves no overlap")
    return baseline[..., y0:y1, x0:x1], moving[..., y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x], (y0, y1, x0, x1)


def _static_region(
    baseline_manifest: dict,
    moving_manifest: dict,
    crop: tuple[int, int, int, int],
    frames: int,
) -> np.ndarray:
    y0, y1, x0, x1 = crop
    baseline_motion = np.load(Path(baseline_manifest["motion"]["mask_path"]), mmap_mode="r")[:frames, 0, ::2, ::2]
    moving_motion = np.load(Path(moving_manifest["motion"]["mask_path"]), mmap_mode="r")[:frames, 0, ::2, ::2]
    baseline_static = (1.0 - baseline_motion.mean(axis=0)) >= 0.9
    moving_static = (1.0 - moving_motion.mean(axis=0)) >= 0.9
    shift_x = int(moving_manifest["registration"]["integer_shift_x"])
    shift_y = int(moving_manifest["registration"]["integer_shift_y"])
    moving_static_crop = moving_static[y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
    return baseline_static[y0:y1, x0:x1] & moving_static_crop


def _channel_bins(reference: np.ndarray, mapped: np.ndarray, static: np.ndarray) -> dict[str, list[dict[str, float | int]]]:
    result: dict[str, list[dict[str, float | int]]] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        reference_values = reference[channel][static].astype(np.float64, copy=False)
        mapped_values = mapped[channel][static].astype(np.float64, copy=False)
        residual_values = mapped_values - reference_values
        labels = np.digitize(reference_values, BIN_EDGES, right=False) - 1
        rows: list[dict[str, float | int]] = []
        for index in range(len(BIN_EDGES) - 1):
            selected = labels == index
            count = int(np.count_nonzero(selected))
            if count == 0:
                rows.append({"lower_dn": int(BIN_EDGES[index]), "upper_dn": int(BIN_EDGES[index + 1]), "count": 0})
                continue
            residual = residual_values[selected]
            rows.append({
                "lower_dn": int(BIN_EDGES[index]),
                "upper_dn": int(BIN_EDGES[index + 1]),
                "count": count,
                "reference_mean_dn": float(reference_values[selected].mean()),
                "mapped_mean_dn": float(mapped_values[selected].mean()),
                "residual_mean_dn": float(residual.mean()),
                "residual_std_dn": float(residual.std()),
            })
        result[name] = rows
    return result


def diagnose(supervision_dir: Path, max_frames: int) -> dict[str, object]:
    baseline_manifest = _load_manifest(supervision_dir, "645x")
    moving_manifest = _load_manifest(supervision_dir, "1000x")
    baseline_mean = _temporal_mean(baseline_manifest, max_frames)
    moving_mean = _temporal_mean(moving_manifest, max_frames)
    frames = min(max_frames, int(baseline_manifest["frames_used"]), int(moving_manifest["frames_used"]))
    shift_x = int(moving_manifest["registration"]["integer_shift_x"])
    shift_y = int(moving_manifest["registration"]["integer_shift_y"])
    baseline_crop, moving_crop, crop = _overlap_pair(baseline_mean, moving_mean, shift_x, shift_y)
    static = _static_region(baseline_manifest, moving_manifest, crop, frames)
    baseline_black = baseline_manifest["black_levels"]
    moving_black = moving_manifest["black_levels"]
    fit = fit_exposure_pair(
        moving_crop,
        moving_black,
        baseline_crop,
        baseline_black,
        static,
        reference_label="645x",
    )
    slopes = np.asarray([fit.slopes[name] for name in CHANNEL_NAMES], dtype=np.float32)
    intercepts = np.asarray([fit.intercepts[name] for name in CHANNEL_NAMES], dtype=np.float32)
    mapped = slopes[:, None, None] * moving_crop + intercepts[:, None, None]
    return {
        "moving_gain": "1000x",
        "reference_gain": "645x",
        "frames": frames,
        "registration": {
            "integer_shift_x": shift_x,
            "integer_shift_y": shift_y,
            "float_shift_sensor_px": {
                "x": float(moving_manifest["registration"]["shift_x"]),
                "y": float(moving_manifest["registration"]["shift_y"]),
            },
            "ncc": float(moving_manifest["registration"]["ncc"]),
            "overlap_ratio": float((crop[1] - crop[0]) * (crop[3] - crop[2]) / (PACKED_HEIGHT * PACKED_WIDTH)),
        },
        "static_fraction_of_overlap": float(static.mean()),
        "fit_1000x_to_645x": asdict(fit),
        "bins_by_645x_reference_dn": _channel_bins(baseline_crop, mapped, static),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/1000x_vs_645x_exposure_diagnosis.json"))
    args = parser.parse_args()
    if args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    result = diagnose(args.supervision_dir, args.max_frames)
    payload = json.dumps(result, indent=2) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit temporal fidelity in motion regions for an existing A4 prediction."""

import argparse
import json
from pathlib import Path

import numpy as np

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json
from raw_fusion.finetune.a4 import _read_aligned_video


_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
_PACKED_HEIGHT = 540
_PACKED_WIDTH = 960


def _aligned_motion_mask(supervision_dir: Path, gain: str, frames: int) -> np.ndarray:
    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    source = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")[:frames, 0, ::2, ::2]
    shift_x = int(manifest["registration"]["integer_shift_x"])
    shift_y = int(manifest["registration"]["integer_shift_y"])
    aligned = np.ones((frames, _PACKED_HEIGHT, _PACKED_WIDTH), dtype=np.float32)
    y0, y1 = max(0, -shift_y), min(_PACKED_HEIGHT, _PACKED_HEIGHT - shift_y)
    x0, x1 = max(0, -shift_x), min(_PACKED_WIDTH, _PACKED_WIDTH - shift_x)
    aligned[:, y0:y1, x0:x1] = source[:, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
    return aligned


def _channel_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_abs_dn": float(np.mean(np.abs(values))),
        "p95_abs_dn": float(np.percentile(np.abs(values), 95)),
    }


def audit_motion_temporal_fidelity(
    *,
    supervision_dir: Path,
    prediction_path: Path,
    gain: str,
    output_path: Path,
) -> dict[str, object]:
    prediction = np.load(prediction_path, mmap_mode="r").astype(np.float32, copy=False)
    if prediction.ndim != 4 or prediction.shape[1:] != (4, _PACKED_HEIGHT, _PACKED_WIDTH):
        raise ValueError(f"unexpected prediction shape: {prediction.shape}")

    input_video, overlap = _read_aligned_video(supervision_dir, gain, prediction.shape[0])
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    black = _channel_vector(report["black_levels"]).astype(np.float32)
    scale = (_WHITE_LEVEL - black)[None, :, None, None]
    prediction_dn = prediction * scale + black[None, :, None, None]
    input_dn = input_video * scale + black[None, :, None, None]

    motion = _aligned_motion_mask(supervision_dir, gain, prediction.shape[0])
    valid = overlap[:, 0].astype(bool)
    moving = valid & (motion >= 0.5)
    transition_mask = moving[1:] | moving[:-1]
    static_transition_mask = valid[1:] & valid[:-1] & ~transition_mask

    prediction_diff = np.abs(np.diff(prediction_dn, axis=0))
    input_diff = np.abs(np.diff(input_dn, axis=0))
    motion_difference: list[dict[str, object]] = []
    static_difference: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for channel, name in enumerate(_CHANNEL_NAMES):
        moving_prediction = prediction_diff[:, channel][transition_mask]
        moving_input = input_diff[:, channel][transition_mask]
        static_prediction = prediction_diff[:, channel][static_transition_mask]
        static_input = input_diff[:, channel][static_transition_mask]
        motion_difference.append({
            "channel": name,
            "prediction": _channel_stats(moving_prediction),
            "input": _channel_stats(moving_input),
            "prediction_to_input_mean_ratio": float(np.mean(moving_prediction) / max(np.mean(moving_input), 1e-6)),
        })
        static_difference.append({
            "channel": name,
            "prediction": _channel_stats(static_prediction),
            "input": _channel_stats(static_input),
            "prediction_to_input_mean_ratio": float(np.mean(static_prediction) / max(np.mean(static_input), 1e-6)),
        })

    for frame in range(1, prediction.shape[0]):
        mask = transition_mask[frame - 1]
        if not np.any(mask):
            continue
        prediction_values = prediction_diff[frame - 1, :, mask]
        input_values = input_diff[frame - 1, :, mask]
        prediction_mean = float(np.mean(prediction_values))
        input_mean = float(np.mean(input_values))
        transition_rows.append({
            "from_frame": frame - 1,
            "to_frame": frame,
            "prediction_mean_abs_dn": prediction_mean,
            "input_mean_abs_dn": input_mean,
            "prediction_to_input_ratio": prediction_mean / max(input_mean, 1e-6),
            "moving_pixels": int(np.count_nonzero(mask)),
        })

    transition_rows.sort(key=lambda row: float(row["prediction_to_input_ratio"]))
    result: dict[str, object] = {
        "protocol": "A3 motion temporal fidelity audit",
        "gain": gain,
        "prediction_path": str(prediction_path.resolve()),
        "frames": int(prediction.shape[0]),
        "motion_fraction": float(np.mean(moving)),
        "motion_difference": motion_difference,
        "static_difference": static_difference,
        "lowest_motion_transition_ratios": transition_rows[:20],
        "highest_motion_transition_ratios": list(reversed(transition_rows[-10:])),
        "interpretation": {
            "prediction_to_input_ratio_is_a_diagnostic": True,
            "motion_region_has_no_clean_reference_metric": True,
            "lower_motion_ratio_can_indicate_temporal_content_suppression": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--prediction", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/prediction_645x.npy"))
    parser.add_argument("--gain", default="645x")
    parser.add_argument("--output", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/motion_audit_645x.json"))
    args = parser.parse_args()
    print(json.dumps(audit_motion_temporal_fidelity(
        supervision_dir=args.supervision_dir,
        prediction_path=args.prediction,
        gain=args.gain,
        output_path=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()

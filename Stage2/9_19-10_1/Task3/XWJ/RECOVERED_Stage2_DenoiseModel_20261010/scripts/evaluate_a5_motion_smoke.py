#!/usr/bin/env python3
"""Compare A5 motion smoke against A3 on disjoint motion and native checks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import _channel_vector, _load_json
from raw_fusion.finetune.a4 import (
    _read_aligned_video,
    evaluate_outputs,
    infer_video,
    load_checkpoint_model,
)
from raw_fusion.finetune.a5_native import evaluate_native_outputs, read_native_video, _load_native_reference
from audit_motion_temporal_fidelity import _aligned_motion_mask


def _motion_summary(prediction: np.ndarray, input_video: np.ndarray, motion: np.ndarray) -> dict[str, float]:
    pred_diff = np.abs(np.diff(prediction, axis=0))
    input_diff = np.abs(np.diff(input_video, axis=0))
    transition = (motion[1:] >= 0.5) | (motion[:-1] >= 0.5)
    moving_pred = pred_diff.transpose(0, 2, 3, 1)[transition]
    moving_input = input_diff.transpose(0, 2, 3, 1)[transition]
    static = ~transition
    static_pred = pred_diff.transpose(0, 2, 3, 1)[static]
    return {
        "motion_mean_abs_normalized": float(np.mean(moving_pred)),
        "input_motion_mean_abs_normalized": float(np.mean(moving_input)),
        "motion_to_input_ratio": float(np.mean(moving_pred) / max(np.mean(moving_input), 1e-8)),
        "static_mean_abs_normalized": float(np.mean(static_pred)),
    }


def evaluate(
    *,
    supervision_dir: Path,
    checkpoint: Path,
    baseline_645_prediction: Path,
    baseline_native_prediction: Path,
    native_probe_dir: Path,
    output_dir: Path,
    device: torch.device,
    tile_batch: int,
    use_ema: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_model(checkpoint, device, ema=use_ema)

    start, frames = 160, 40
    video_645, overlap_645 = _read_aligned_video(supervision_dir, "645x", frames, start_frame=start)
    candidate_645 = infer_video(model, video_645, device, tile_batch=tile_batch)
    np.save(output_dir / "prediction_645x_holdout_0160_0199.npy", candidate_645.astype(np.float16))
    baseline_645_all = np.load(baseline_645_prediction, mmap_mode="r")
    baseline_645 = baseline_645_all[start:start + frames].astype(np.float32)
    motion_645 = _aligned_motion_mask(supervision_dir, "645x", start + frames)[start:start + frames]
    ref = np.load(supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy")
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    black = _channel_vector(report["black_levels"])
    cross_gain = {
        "baseline": evaluate_outputs(baseline_645, video_645, ref, overlap_645, motion_645, black),
        "candidate": evaluate_outputs(candidate_645, video_645, ref, overlap_645, motion_645, black),
        "baseline_motion": _motion_summary(baseline_645, video_645, motion_645),
        "candidate_motion": _motion_summary(candidate_645, video_645, motion_645),
    }
    del video_645, overlap_645, baseline_645, candidate_645

    native_video, native_motion, native_black = read_native_video(supervision_dir, "1000x", 40)
    candidate_native = infer_video(model, native_video, device, tile_batch=tile_batch)
    np.save(output_dir / "prediction_1000x_native_frames_0000_0039.npy", candidate_native.astype(np.float16))
    baseline_native = np.load(baseline_native_prediction, mmap_mode="r")[:40].astype(np.float32)
    native_ref, native_valid = _load_native_reference(native_probe_dir)
    native = {
        "baseline": evaluate_native_outputs(
            baseline_native, native_video, native_ref, native_valid, native_motion, native_black
        ),
        "candidate": evaluate_native_outputs(
            candidate_native, native_video, native_ref, native_valid, native_motion, native_black
        ),
    }
    result: dict[str, object] = {
        "protocol": "A5 motion smoke regression evaluation",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_weights": "ema" if use_ema else "model",
        "645x_holdout_frame_range": [160, 200],
        "1000x_native_frame_range": [0, 40],
        "cross_gain_645x": cross_gain,
        "native_1000x": native,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finetune_mis20s1/training_a5_motion_smoke_flow1/checkpoint_step_000020.pth"))
    parser.add_argument("--baseline-645-prediction", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/prediction_645x.npy"))
    parser.add_argument("--baseline-native-prediction", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/native_1000x/native_prediction_1000x.npy"))
    parser.add_argument("--native-probe-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/a5_motion_smoke_flow1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()
    print(json.dumps(evaluate(
        supervision_dir=args.supervision_dir,
        checkpoint=args.checkpoint,
        baseline_645_prediction=args.baseline_645_prediction,
        baseline_native_prediction=args.baseline_native_prediction,
        native_probe_dir=args.native_probe_dir,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        tile_batch=args.tile_batch,
        use_ema=args.weights == "ema",
    ), indent=2))


if __name__ == "__main__":
    main()

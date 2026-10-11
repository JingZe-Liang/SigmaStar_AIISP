#!/usr/bin/env python3
"""Evaluate an A6 motion expert on disjoint real 645x and native 1000x data."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.a4 import (  # noqa: E402
    _read_aligned_video,
    evaluate_outputs,
    infer_video,
)
from raw_fusion.finetune.a5_native import (  # noqa: E402
    _load_native_reference,
    evaluate_native_outputs,
    read_native_video,
)
from raw_fusion.finetune.a6_motion_expert import (  # noqa: E402
    load_motion_expert_checkpoint,
)
from audit_motion_temporal_fidelity import _aligned_motion_mask  # noqa: E402
from evaluate_a5_motion_smoke import _motion_summary  # noqa: E402


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
    weights: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_motion_expert_checkpoint(checkpoint, device, weights=weights)

    start, frames = 160, 40
    video_645, overlap_645 = _read_aligned_video(
        supervision_dir, "645x", frames, start_frame=start
    )
    candidate_645 = infer_video(model, video_645, device, tile_batch=tile_batch)
    np.save(
        output_dir / "prediction_645x_holdout_0160_0199.npy",
        candidate_645.astype(np.float16),
    )
    baseline_645_all = np.load(baseline_645_prediction, mmap_mode="r")
    baseline_645 = baseline_645_all[start:start + frames].astype(np.float32)
    motion_645 = _aligned_motion_mask(
        supervision_dir, "645x", start + frames
    )[start:start + frames]
    reference = np.load(
        supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy"
    )
    report = _load_json(
        supervision_dir / "references" / "a1_reference_report.json"
    )
    black = _channel_vector(report["black_levels"])
    cross_gain = {
        "baseline": evaluate_outputs(
            baseline_645, video_645, reference, overlap_645, motion_645, black
        ),
        "candidate": evaluate_outputs(
            candidate_645, video_645, reference, overlap_645, motion_645, black
        ),
        "baseline_motion": _motion_summary(
            baseline_645, video_645, motion_645
        ),
        "candidate_motion": _motion_summary(
            candidate_645, video_645, motion_645
        ),
    }
    del video_645, overlap_645, baseline_645, candidate_645

    native_video, native_motion, native_black = read_native_video(
        supervision_dir, "1000x", 40
    )
    candidate_native = infer_video(
        model, native_video, device, tile_batch=tile_batch
    )
    np.save(
        output_dir / "prediction_1000x_native_frames_0000_0039.npy",
        candidate_native.astype(np.float16),
    )
    baseline_native = np.load(
        baseline_native_prediction, mmap_mode="r"
    )[:40].astype(np.float32)
    native_reference, native_valid = _load_native_reference(native_probe_dir)
    native = {
        "baseline": evaluate_native_outputs(
            baseline_native,
            native_video,
            native_reference,
            native_valid,
            native_motion,
            native_black,
        ),
        "candidate": evaluate_native_outputs(
            candidate_native,
            native_video,
            native_reference,
            native_valid,
            native_motion,
            native_black,
        ),
    }
    result: dict[str, object] = {
        "protocol": "A6 frozen motion-expert real-video regression evaluation",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_weights": weights,
        "645x_holdout_frame_range": [160, 200],
        "1000x_native_frame_range": [0, 40],
        "cross_gain_645x": cross_gain,
        "native_1000x": native,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-645-prediction",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/prediction_645x.npy",
    )
    parser.add_argument(
        "--baseline-native-prediction",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/native_1000x/native_prediction_1000x.npy",
    )
    parser.add_argument(
        "--native-probe-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/a6_motion_expert",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--weights", choices=("model", "ema"), default="ema")
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
        weights=args.weights,
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()

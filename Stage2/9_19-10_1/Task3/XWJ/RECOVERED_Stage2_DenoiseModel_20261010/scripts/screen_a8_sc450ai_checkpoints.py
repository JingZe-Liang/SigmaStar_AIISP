#!/usr/bin/env python3
"""Screen earlier sc450ai checkpoints on fixed no-reference clips."""

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_a8_sc450ai import (  # noqa: E402
    _candidate_stress_metrics,
    _load_profile,
    _load_specs,
    _sha256,
    _single_clip_spec,
    _slice_baseline_tail,
)
from raw_fusion.finetune.a3 import _load_json  # noqa: E402
from raw_fusion.finetune.a4 import load_checkpoint_model  # noqa: E402
from raw_fusion.finetune.cross_sensor import (  # noqa: E402
    evaluate_stress_prediction,
    infer_stress_clips,
)


def _metric_delta(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, float]:
    keys = (
        "mean_output_bias_dn",
        "mean_abs_output_change_dn",
        "temporal_mad_ratio",
        "motion_temporal_ratio",
        "static_temporal_ratio",
        "spatial_high_frequency_ratio",
        "normalized_clip_fraction",
    )
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a8_sensor_adaptation/sc450ai/training_20261013",
    )
    parser.add_argument(
        "--cross-sensor-config",
        type=Path,
        default=REPO_ROOT / "configs/a3_cross_sensor_extension_20261012.json",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "artifacts/a3_cross_sensor_extension_20261012",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=Path("/tmp/a8_sc450ai_checkpoint_screen_20261013"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a8_sensor_adaptation/sc450ai/checkpoint_screen_20261013.json",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 200, 300, 400])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.is_file():
        raise FileExistsError(f"completed screen already exists: {args.output}")

    specs = _load_specs(args.cross_sensor_config)
    train_tail_spec = _single_clip_spec(specs["sc450ai_804x_lightchanging"], 142)
    holdout_spec = specs["sc450ai_1435x_motion_black_object"]
    train_dir = (
        args.baseline_root
        / "sequences/sc450ai/sc450ai_804x_lightchanging"
    )
    holdout_dir = (
        args.baseline_root
        / "sequences/sc450ai/sc450ai_1435x_motion_black_object"
    )
    train_profile = _load_profile(train_dir / "profile")
    holdout_profile = _load_profile(holdout_dir / "profile")
    baseline_train_path = _slice_baseline_tail(
        train_dir / "inference/prediction_stress.npy",
        args.inference_root / "baseline_train_tail.npy",
    )
    baseline_train = evaluate_stress_prediction(
        train_tail_spec,
        train_profile,
        baseline_train_path,
    )
    baseline_holdout = _load_json(holdout_dir / "metrics.json")
    baseline_holdout = {
        key: value
        for key, value in baseline_holdout.items()
        if key not in ("panel_manifest", "panel_count")
    }

    device = torch.device(args.device)
    checkpoints: dict[str, object] = {}
    for step in args.steps:
        checkpoint = args.training_dir / f"checkpoint_step_{step:06d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_dir = args.inference_root / f"step_{step:06d}"
        model = load_checkpoint_model(checkpoint, device, ema=True)
        try:
            train_prediction = checkpoint_dir / "train_tail.npy"
            infer_stress_clips(
                model,
                train_tail_spec,
                train_profile,
                device,
                train_prediction,
                tile_batch=1,
                resume_key=f"{_sha256(checkpoint)}:train-tail",
            )
            holdout_prediction = checkpoint_dir / "holdout.npy"
            infer_stress_clips(
                model,
                holdout_spec,
                holdout_profile,
                device,
                holdout_prediction,
                tile_batch=1,
                resume_key=f"{_sha256(checkpoint)}:holdout",
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_metrics = _candidate_stress_metrics(
            evaluate_stress_prediction(
                train_tail_spec, train_profile, train_prediction
            )
        )
        holdout_metrics = _candidate_stress_metrics(
            evaluate_stress_prediction(
                holdout_spec, holdout_profile, holdout_prediction
            )
        )
        checkpoints[str(step)] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "train_tail": train_metrics,
            "train_tail_delta": _metric_delta(baseline_train, train_metrics),
            "holdout": holdout_metrics,
            "holdout_delta": _metric_delta(baseline_holdout, holdout_metrics),
            "screen_pass": bool(
                train_metrics["legal_raw_fraction"] == 1.0
                and train_metrics["normalized_clip_fraction"]
                <= baseline_train["normalized_clip_fraction"] + 0.005
                and holdout_metrics["legal_raw_fraction"] == 1.0
                and holdout_metrics["normalized_clip_fraction"]
                <= baseline_holdout["normalized_clip_fraction"] + 0.005
                and holdout_metrics["spatial_high_frequency_ratio"]
                >= baseline_holdout["spatial_high_frequency_ratio"] + 0.02
                and holdout_metrics["static_temporal_ratio"]
                <= baseline_holdout["static_temporal_ratio"] + 0.05
            ),
        }
        print(
            json.dumps(
                {
                    "step": step,
                    "screen_pass": checkpoints[str(step)]["screen_pass"],
                    "train_tail_delta": checkpoints[str(step)]["train_tail_delta"],
                    "holdout_delta": checkpoints[str(step)]["holdout_delta"],
                }
            ),
            flush=True,
        )

    result = {
        "protocol": "A8 sc450ai earlier-checkpoint no-reference screen",
        "baseline_train_tail": baseline_train,
        "baseline_holdout": baseline_holdout,
        "checkpoints": checkpoints,
        "passing_steps": [
            int(step)
            for step, report in checkpoints.items()
            if report["screen_pass"]
        ],
        "note": (
            "A passing checkpoint still requires the full MIS20S1 guard and "
            "fixed-ISP video review."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

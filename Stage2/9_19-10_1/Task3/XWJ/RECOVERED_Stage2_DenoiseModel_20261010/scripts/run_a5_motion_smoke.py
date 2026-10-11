#!/usr/bin/env python3
"""Run the bounded A5 motion-supervision smoke experiment."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a5_motion import A5MotionConfig, train_motion_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision")
    parser.add_argument("--noise-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/noise_model")
    parser.add_argument("--a3-checkpoint", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth")
    parser.add_argument("--dynamic-train-manifest", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/a5_dynamic_train_flow1/manifest.json")
    parser.add_argument("--dynamic-holdout-manifest", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/a5_dynamic_holdout_flow1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/training_a5_motion_smoke_flow1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dynamic-weight", type=float, default=0.1)
    parser.add_argument("--static-anchor-weight", type=float, default=1.0)
    parser.add_argument("--trainable-scope", choices=("all", "mutual_attention"), default="all")
    parser.add_argument(
        "--static-target-mode",
        choices=("a1", "a3_teacher", "same_window_a3_teacher"),
        default="a1",
    )
    parser.add_argument("--motion-boundary-radius", type=int, default=1)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    config = A5MotionConfig(
        supervision_dir=str(args.supervision_dir),
        noise_dir=str(args.noise_dir),
        a3_checkpoint=str(args.a3_checkpoint),
        dynamic_train_manifest=str(args.dynamic_train_manifest),
        dynamic_holdout_manifest=str(args.dynamic_holdout_manifest),
        output_dir=str(args.output_dir),
        steps=args.steps,
        learning_rate=args.learning_rate,
        dynamic_weight=args.dynamic_weight,
        static_anchor_weight=args.static_anchor_weight,
        device=args.device,
        trainable_scope=args.trainable_scope,
        static_target_mode=args.static_target_mode,
        motion_boundary_radius=args.motion_boundary_radius,
    )
    print(json.dumps(train_motion_smoke(config), indent=2), flush=True)


if __name__ == "__main__":
    main()

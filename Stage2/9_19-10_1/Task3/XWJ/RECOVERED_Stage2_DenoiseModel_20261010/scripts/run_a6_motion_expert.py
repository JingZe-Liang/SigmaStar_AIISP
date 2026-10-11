#!/usr/bin/env python3
"""Train the frozen-A3 motion expert on exact low-gain dynamic targets."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a6_motion_expert import (  # noqa: E402
    A6MotionExpertConfig,
    train_motion_expert,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/noise_model",
    )
    parser.add_argument(
        "--a3-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/training_a6_motion_expert",
    )
    parser.add_argument(
        "--old-teacher-prediction",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/old_teacher_645x.npy",
    )
    parser.add_argument("--initial-expert-checkpoint", type=Path)
    parser.add_argument("--teacher-distillation-fraction", type=float, default=0.0)
    parser.add_argument("--procedural-motion-fraction", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--expert-channels", type=int, default=32)
    parser.add_argument("--max-residual", type=float, default=0.25)
    parser.add_argument("--motion-boundary-radius", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = A6MotionExpertConfig(
        supervision_dir=str(args.supervision_dir),
        noise_dir=str(args.noise_dir),
        a3_checkpoint=str(args.a3_checkpoint),
        output_dir=str(args.output_dir),
        old_teacher_prediction=str(args.old_teacher_prediction),
        initial_expert_checkpoint=(
            str(args.initial_expert_checkpoint)
            if args.initial_expert_checkpoint is not None
            else None
        ),
        teacher_distillation_fraction=args.teacher_distillation_fraction,
        procedural_motion_fraction=args.procedural_motion_fraction,
        steps=args.steps,
        learning_rate=args.learning_rate,
        expert_channels=args.expert_channels,
        max_residual=args.max_residual,
        motion_boundary_radius=args.motion_boundary_radius,
        eval_samples=args.eval_samples,
        device=args.device,
    )
    print(json.dumps(train_motion_expert(config), indent=2), flush=True)


if __name__ == "__main__":
    main()

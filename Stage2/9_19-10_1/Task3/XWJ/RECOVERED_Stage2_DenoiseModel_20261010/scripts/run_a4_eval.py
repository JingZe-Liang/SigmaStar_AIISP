#!/usr/bin/env python3
"""Run the reproducible A4 evaluation protocol."""

import argparse
from pathlib import Path

import torch

from raw_fusion.finetune.a4 import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/checkpoints/RViDeformer-L_CRVD.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation"))
    parser.add_argument("--gains", default="1000x,645x")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for A4 inference")
    device = torch.device(args.device)
    print(run_evaluation(supervision_dir=args.supervision_dir, checkpoint=args.checkpoint, teacher_checkpoint=args.teacher_checkpoint, output_dir=args.output_dir, gains=[v.strip() for v in args.gains.split(",") if v.strip()], max_frames=args.max_frames, device=device, tile_batch=args.tile_batch))


if __name__ == "__main__":
    main()

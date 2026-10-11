#!/usr/bin/env python3
"""Run the independent 1000x native-reference evaluation."""

import argparse
from pathlib import Path

import torch

from raw_fusion.finetune.a5_native import run_native_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--probe-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/native_1000x"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native 1000x inference")
    result = run_native_evaluation(
        supervision_dir=args.supervision_dir,
        probe_dir=args.probe_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        device=torch.device(args.device),
        tile_batch=args.tile_batch,
    )
    print(result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import train_joint as base
from .config import load_config, project_root
from .dataset_v3 import BayerAwarePatchDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/company.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    args = parser.parse_args()
    config = load_config(args.config)

    # Reuse the validated joint trainer while replacing only its dataset class.
    base.ThresholdNormalizedPatchDataset = BayerAwarePatchDataset
    output_dir = args.output_dir or project_root(config) / "outputs" / "checkpoints" / "joint_v3"
    checkpoint = base.train_joint(
        config,
        output_dir.resolve(),
        torch.device(args.device),
        epochs_override=args.epochs,
        steps_override=args.steps_per_epoch,
    )
    print(f"Saved Bayer-aware checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

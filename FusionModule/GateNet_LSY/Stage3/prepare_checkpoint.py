from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Add inference statistics to a Stage3 checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Stage2 checkpoint from the matching fold")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    reference_config = reference["config"]
    for key in ("train_statistics", "val_statistics", "black_source", "black_dnr", "warmup_frames"):
        if key in reference_config:
            config[key] = reference_config[key]
    checkpoint["config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

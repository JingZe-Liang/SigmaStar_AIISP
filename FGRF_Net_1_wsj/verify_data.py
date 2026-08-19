#!/usr/bin/env python3
"""Validate RAW streams, bidirectional flow counts, and one dataset sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()
    total = 0
    for config_path in args.config:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        dataset = build_dataset(config, training=True)
        sample = dataset[0]
        total += len(dataset)
        print(
            f"OK {Path(config_path).name}: frames={dataset.frame_count} "
            f"training_samples={len(dataset)} packed={dataset.packed_hw} "
            f"forward_pairs={dataset.forward_flow.pair_count} "
            f"backward_pairs={dataset.backward_flow.pair_count} "
            f"sample_static_ratio={sample['static_mask'].mean().item():.4f}"
        )
    print(f"OK combined training samples={total}")


if __name__ == "__main__":
    main()

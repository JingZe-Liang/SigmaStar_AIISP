#!/usr/bin/env python3
"""Validate v2 RAW streams, requested RAFT paths, and one supervision sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import build_dataset


_FLOW_645 = (
    Path("/HardDisk/jingzeliang/projects/SigmaStar_project/raft_sequence_inference/outputs/denoised_raft_things_645x/flow_npy"),
    Path("/HardDisk/jingzeliang/projects/SigmaStar_project/raft_sequence_inference/outputs/denoised_raft_things_645x_backward/flow_npy"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()
    for config_path in args.config:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if Path(config_path).name == "config_train_645.json":
            configured = (Path(config["flow"]["forward_dir"]), Path(config["flow"]["backward_dir"]))
            if configured != _FLOW_645:
                raise ValueError(f"645x must use requested RAFT flows: {configured}")
        dataset = build_dataset(config, samples_per_epoch=1, training=False, seed=int(config.get("seed", 1234)))
        sample = dataset[0]
        print(
            f"OK {Path(config_path).name}: frames={dataset.frame_count} packed={dataset.packed_hw} "
            f"flow={dataset.forward.pair_count}/{dataset.backward.pair_count} "
            f"static={sample['static_mask'].mean().item():.4f} motion={sample['motion_mask'].mean().item():.4f}",
            flush=True,
        )
        print(f"  forward={dataset.forward.directory}\n  backward={dataset.backward.directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

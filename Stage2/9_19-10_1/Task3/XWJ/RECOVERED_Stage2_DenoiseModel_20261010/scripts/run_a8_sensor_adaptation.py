#!/usr/bin/env python3
"""Run one conservative A8 static-reference sensor adaptation."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a8_adaptation import (  # noqa: E402
    load_adaptation_config,
    train_static_sensor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a8_sc635hai_adaptation_20261013.json",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_adaptation_config(args.config).with_runtime_overrides(
        steps=args.steps,
        output_dir=args.output_dir,
        device=args.device,
    )
    result = train_static_sensor(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

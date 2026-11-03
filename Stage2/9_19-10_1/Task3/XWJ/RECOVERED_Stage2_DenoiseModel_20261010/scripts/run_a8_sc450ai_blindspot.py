#!/usr/bin/env python3
"""Run conservative sc450ai blind-spot adaptation from immutable A3."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a8_blindspot import (  # noqa: E402
    load_blindspot_config,
    train_blindspot_sensor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a8_sc450ai_blindspot_20261013.json",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_blindspot_config(args.config).with_runtime_overrides(
        steps=args.steps,
        output_dir=args.output_dir,
        device=args.device,
    )
    result = train_blindspot_sensor(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit and validate an A9 target-sensor temporal noise profile."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a9_generalization import (  # noqa: E402
    build_sensor_noise_profile,
    load_noise_profile_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a9_sc450ai_noise_profile_20261016.json",
    )
    args = parser.parse_args()
    result = build_sensor_noise_profile(load_noise_profile_config(args.config))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

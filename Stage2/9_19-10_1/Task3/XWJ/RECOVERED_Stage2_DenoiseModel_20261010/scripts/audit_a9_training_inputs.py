#!/usr/bin/env python3
"""Audit A9 hashes, source sampling, tensor contracts, and source schedule."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a9_generalization import (  # noqa: E402
    audit_generalization_inputs,
    load_generalization_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a9_multi_sensor_generalization_20261016.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a9_multi_sensor_generalization/preflight_20261016.json",
    )
    args = parser.parse_args()
    result = audit_generalization_inputs(load_generalization_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

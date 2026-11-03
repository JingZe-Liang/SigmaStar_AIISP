#!/usr/bin/env python3
"""Audit existing sc450ai data for A10 supervision and degradation roles."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a10_sc450_audit import (  # noqa: E402
    audit_sc450ai_supervision,
    load_a10_audit_config,
    save_a10_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/a10_sc450ai_supervision_audit_20261016.json",
    )
    args = parser.parse_args()
    config = load_a10_audit_config(args.config)
    result = audit_sc450ai_supervision(config)
    save_a10_audit(config, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

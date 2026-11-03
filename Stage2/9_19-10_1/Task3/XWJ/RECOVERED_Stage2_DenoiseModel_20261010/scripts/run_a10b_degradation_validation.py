#!/usr/bin/env python3
"""Run deterministic A10b sc450ai degradation-only validation."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a10b_degradation import (  # noqa: E402
    load_a10b_config,
    save_a10b_result,
    validate_a10b_degradation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/a10b_sc450ai_degradation_validation_20261016.json",
    )
    args = parser.parse_args()
    config = load_a10b_config(args.config)
    result, visual = validate_a10b_degradation(config)
    result_path, visual_path = save_a10b_result(config, result, visual)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_generator": result["selected_generator"],
                "result": str(result_path.resolve()),
                "visual_samples": str(visual_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

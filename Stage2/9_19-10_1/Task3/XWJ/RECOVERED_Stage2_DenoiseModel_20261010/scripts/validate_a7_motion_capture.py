"""Validate an A7 MIS20S1 real-motion capture pilot without modifying source data."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a7_capture import (  # noqa: E402
    validate_capture_dataset,
    write_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="base directory for clip paths; defaults to the manifest directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "finetune_mis20s1"
        / "a7_motion_pilot"
        / "capture_validation.json",
    )
    args = parser.parse_args()

    report = validate_capture_dataset(args.manifest, args.dataset_root)
    write_validation_report(report, args.out)
    summary = {
        "overall_pass": report["overall_pass"],
        "clips_passed": sum(clip["passed"] for clip in report["clips"]),
        "clips_total": len(report["clips"]),
        "inventory_passed": report["inventory"]["passed"],
        "report": str(args.out.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

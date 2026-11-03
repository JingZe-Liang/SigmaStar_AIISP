"""Apply the frozen A14 per-source acceptance rules to all seven evaluated states."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a14_static_summary import summarize_evaluations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_evaluations(args.baseline, args.candidates, args.output_dir)
    print(json.dumps({"status": result["status"], "selected": result["selected"],
                      "summary": str(args.output_dir / "summary.json"),
                      "rows": str(args.output_dir / "rows.csv")}, indent=2))


if __name__ == "__main__":
    main()

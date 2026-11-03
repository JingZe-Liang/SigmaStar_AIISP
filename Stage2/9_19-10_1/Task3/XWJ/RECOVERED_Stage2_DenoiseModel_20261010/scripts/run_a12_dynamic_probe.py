#!/usr/bin/env python3
"""Prepare or evaluate immutable A12 constructed-motion diagnostics."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a12_dynamic_probe import evaluate_probes, prepare_probes, rescore_cached_probes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/a12_dynamic_probe_20261029.json")
    parser.add_argument("--phase", choices=("prepare", "evaluate", "rescore"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-name", default="a3_baseline")
    parser.add_argument("--source-evaluation-name", default="a3_baseline")
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir is not None else REPO_ROOT / config["output_dir"]
    if args.phase == "prepare":
        result = prepare_probes(config, REPO_ROOT, output_dir / "prepared")
    else:
        if not args.evaluation_name or Path(args.evaluation_name).name != args.evaluation_name:
            raise ValueError("evaluation-name must be one directory name")
        if args.phase == "evaluate":
            baseline = json.loads(args.baseline_config.read_text(encoding="utf-8")) if args.baseline_config is not None else None
            result = evaluate_probes(output_dir / "prepared", output_dir / args.evaluation_name, REPO_ROOT, args.device, baseline)
        else:
            if not args.source_evaluation_name or Path(args.source_evaluation_name).name != args.source_evaluation_name:
                raise ValueError("source-evaluation-name must be one directory name")
            result = rescore_cached_probes(output_dir / "prepared", output_dir / args.source_evaluation_name,
                                          output_dir / args.evaluation_name)
    print(json.dumps({"phase": args.phase, "cases": len(result["cases"]), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()

"""Evaluate one exact A12 checkpoint state on fixed probes and native windows."""

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a12_paired_eval import evaluate_paired_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--weights", choices=("model", "ema"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--phase", choices=("probes", "native", "all"), default="all")
    args = parser.parse_args()
    checkpoint, output_dir = args.checkpoint.resolve(), args.output_dir.resolve()
    os.chdir(REPO_ROOT)
    result = evaluate_paired_checkpoint(checkpoint, args.weights, args.label, output_dir, args.device,
                                        args.phase, args.checkpoint_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

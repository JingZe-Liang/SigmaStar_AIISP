"""Prepare frozen A14 static ROIs or evaluate an exact checkpoint state."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a14_static_eval import evaluate_static_checkpoint, prepare_static_cases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze reference-selected static ROI inputs and masks on CPU")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Score one exact model or EMA state in FP32")
    evaluate.add_argument("--prepared-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-sha256")
    evaluate.add_argument("--weights", choices=("model", "ema"), required=True)
    evaluate.add_argument("--label", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", required=True)
    evaluate.add_argument("--expected-gpu-uuid")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_static_cases(args.config, args.output_dir)
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "sources": len(result["sources"]), "cases": len(result["cases"])}))
    else:
        result = evaluate_static_checkpoint(args.prepared_dir, args.checkpoint, args.weights, args.label,
                                            args.output_dir, args.device, args.checkpoint_sha256, args.expected_gpu_uuid)
        print(json.dumps({"evaluation": str(args.output_dir / "evaluation.json"), "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()

"""Freeze all A14 training samples on CPU or explicitly train the static model."""

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a14_static_train import load_training_config, prepare_static_training, train_static_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Training-only runtime output directory; frozen config and samples stay unchanged")
    args = parser.parse_args()
    if args.output_dir is not None and not args.train:
        parser.error("--output-dir is supported only with --train")
    config_path = args.config.resolve()
    output_override = args.output_dir.resolve() if args.output_dir is not None else None
    os.chdir(REPO_ROOT)
    config = load_training_config(config_path)
    result = train_static_model(config, output_dir=output_override) if args.train else prepare_static_training(config)
    print(json.dumps({"status": result["status"], "output_dir": result.get("runtime_output_dir", config["output_dir"]),
                      "prepared_dir": config["prepared_dir"], "training_requested": args.train}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

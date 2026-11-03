"""Prepare the A12 sampling pilot; --train is required to execute optimization."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a12_train import A12TrainConfig, prepare_a12, train_a12  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/a12_sampling_pilot_20261029.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Explicitly run the configured CUDA training pilot")
    mode.add_argument("--prepare", action="store_true", help="Run CPU preparation only (the default)")
    mode.add_argument("--dry-run", action="store_true", help="Run CPU preparation only")
    parser.add_argument("--device", help="CUDA device for a later --train run; preparation remains on CPU")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, help="Total A12 updates, including any restored updates")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_override = args.output_dir.resolve() if args.output_dir is not None else None
    resume_override = args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None
    os.chdir(REPO_ROOT)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("dataset_root", "supervision_dir", "noise_dir", "output_dir", "checkpoint"):
        payload[key] = str(Path(payload[key]).resolve())
    if payload.get("resume_checkpoint") is not None:
        payload["resume_checkpoint"] = str(Path(payload["resume_checkpoint"]).resolve())
    config = A12TrainConfig(**payload)
    changes = {}
    if args.device is not None:
        changes["device"] = args.device
    if output_override is not None:
        changes["output_dir"] = str(output_override)
    if resume_override is not None:
        changes["resume_checkpoint"] = str(resume_override)
    if args.steps is not None:
        changes["steps"] = args.steps
    config = replace(config, **changes)
    result = train_a12(config) if args.train else prepare_a12(config)
    print(json.dumps({"status": result["status"], "output_dir": config.output_dir,
                      "training_requested": args.train}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare or explicitly train the shared-prefix A12 sample-refresh ablation."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a12_paired_train import A12PairedTrainConfig, prepare_paired, train_paired  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--train", action="store_true")
    parser.add_argument("--phase", choices=("shared", "fixed", "refresh"))
    parser.add_argument("--purpose", choices=("experiment", "resume_verification"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--parent-checkpoint-sha256")
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--expected-audit-sha256")
    parser.add_argument("--sample-overrides-path", type=Path)
    parser.add_argument("--expected-sample-overrides-sha256")
    args = parser.parse_args()
    config_path = args.config.resolve()
    overrides = {key: value for key, value in vars(args).items()
                 if key not in {"config", "prepare", "train"} and value is not None}
    for key in ("output_dir", "resume_checkpoint", "audit_path", "sample_overrides_path"):
        if key in overrides:
            overrides[key] = str(overrides[key].resolve())
    os.chdir(REPO_ROOT)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("dataset_root", "supervision_dir", "noise_dir", "output_dir", "checkpoint", "resume_checkpoint", "audit_path",
                "sample_overrides_path"):
        if payload.get(key) is not None:
            payload[key] = str(Path(payload[key]).resolve())
    config = replace(A12PairedTrainConfig(**payload), **overrides)
    result = train_paired(config) if args.train else prepare_paired(config)
    print(json.dumps({"status": result["status"], "phase": config.phase,
                      "output_dir": config.output_dir, "training_requested": args.train}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

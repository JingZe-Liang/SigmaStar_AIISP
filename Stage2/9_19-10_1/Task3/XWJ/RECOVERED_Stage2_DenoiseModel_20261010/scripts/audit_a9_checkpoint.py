#!/usr/bin/env python3
"""Audit A9 checkpoint schema and prove frozen A3 tensors stayed bit-exact."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a8_adaptation import sha256_file  # noqa: E402


def _state(payload: dict[str, object], key: str) -> dict[str, torch.Tensor]:
    state = payload[key]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {key} state must be a mapping")
    return state


def _compare(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    allowed: set[str],
) -> dict[str, object]:
    if baseline.keys() != candidate.keys():
        raise ValueError("candidate and A3 state keys differ")
    changed: list[str] = []
    unauthorized: list[str] = []
    maximum = 0.0
    per_tensor: dict[str, float] = {}
    for name in baseline:
        first = baseline[name].detach().cpu()
        second = candidate[name].detach().cpu()
        if torch.equal(first, second):
            continue
        changed.append(name)
        if name not in allowed:
            unauthorized.append(name)
        difference = float(torch.max(torch.abs(first.float() - second.float())))
        per_tensor[name] = difference
        maximum = max(maximum, difference)
    return {
        "changed_tensors": changed,
        "changed_tensor_count": len(changed),
        "unauthorized_tensors": unauthorized,
        "unauthorized_tensor_count": len(unauthorized),
        "maximum_absolute_change": maximum,
        "per_tensor_maximum_absolute_change": per_tensor,
    }


def _log_summary(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("A9 training log is empty")
    sources: dict[str, list[dict[str, object]]] = {}
    for record in records:
        for source in record["sources"]:
            sources.setdefault(str(source), []).append(record)
    return {
        "steps": len(records),
        "finite": all(
            np.isfinite(float(record[key]))
            for record in records
            for key in ("loss", "charbonnier", "gradient", "highpass", "gradient_norm")
        ),
        "maximum_gradient_norm": max(float(record["gradient_norm"]) for record in records),
        "per_source": {
            source: {
                "samples": len(items),
                "mean_loss": float(np.mean([float(item["loss"]) for item in items])),
                "median_loss": float(np.median([float(item["loss"]) for item in items])),
                "maximum_loss": max(float(item["loss"]) for item in items),
            }
            for source, items in sources.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a3-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth",
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_payload = torch.load(args.a3_checkpoint, map_location="cpu", weights_only=True)
    candidate_payload = torch.load(
        args.candidate_checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(baseline_payload, dict) or not isinstance(candidate_payload, dict):
        raise TypeError("A3 and A9 checkpoints must be mappings")
    if candidate_payload.get("schema") != "a9-unified-multi-sensor-generalization-v1":
        raise ValueError("unexpected A9 checkpoint schema")
    trainable = candidate_payload["trainable"]
    if not isinstance(trainable, dict):
        raise TypeError("A9 trainable report must be a mapping")
    allowed = {str(name) for name in trainable["names"]}
    baseline = _state(baseline_payload, "ema")
    live = _compare(baseline, _state(candidate_payload, "model"), allowed)
    ema = _compare(baseline, _state(candidate_payload, "ema"), allowed)
    passed = (
        candidate_payload.get("step") == 100
        and live["unauthorized_tensor_count"] == 0
        and ema["unauthorized_tensor_count"] == 0
    )
    result = {
        "status": "passed" if passed else "failed",
        "a3_checkpoint": str(args.a3_checkpoint.resolve()),
        "a3_checkpoint_sha256": sha256_file(args.a3_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint),
        "schema": candidate_payload["schema"],
        "step": candidate_payload["step"],
        "allowed_trainable_tensors": sorted(allowed),
        "live_weights": live,
        "ema_weights": ema,
        "training_log": _log_summary(args.training_log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

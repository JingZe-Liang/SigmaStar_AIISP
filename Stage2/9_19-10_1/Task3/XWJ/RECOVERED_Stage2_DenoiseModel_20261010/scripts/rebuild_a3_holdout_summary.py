#!/usr/bin/env python3
"""Rebuild a canonical A3 holdout summary from recovered A5 evaluations."""

import argparse
import json
from pathlib import Path


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def rebuild(evaluation_dir: Path) -> dict[str, object]:
    cross_gain: dict[str, list[Path]] = {}
    motion: dict[str, list[Path]] = {}
    native: dict[str, list[Path]] = {}
    for path in sorted(evaluation_dir.glob("*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cross = payload.get("cross_gain_645x", {})
        native_block = payload.get("native_1000x", {})
        if "baseline" in cross:
            cross_gain.setdefault(_canonical(cross["baseline"]), []).append(path)
        if "baseline_motion" in cross:
            motion.setdefault(_canonical(cross["baseline_motion"]), []).append(path)
        if "baseline" in native_block:
            native.setdefault(_canonical(native_block["baseline"]), []).append(path)

    if len(cross_gain) != 1 or len(motion) != 1 or len(native) != 1:
        raise RuntimeError(
            "recovered evaluations do not agree on one A3 baseline: "
            f"cross={len(cross_gain)}, motion={len(motion)}, native={len(native)}"
        )

    cross_key, cross_sources = next(iter(cross_gain.items()))
    motion_key, motion_sources = next(iter(motion.items()))
    native_key, native_sources = next(iter(native.items()))
    common_sources = sorted(
        set(cross_sources) & set(motion_sources) & set(native_sources)
    )
    return {
        "protocol": "recovered A3 disjoint holdout baseline consensus",
        "checkpoint": "artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth",
        "checkpoint_weights": "ema",
        "645x_holdout_frame_range": [160, 200],
        "1000x_native_frame_range": [0, 40],
        "consensus_source_count": len(common_sources),
        "consensus_sources": [str(path) for path in common_sources],
        "cross_gain_645x": {
            "metrics": json.loads(cross_key),
            "motion": json.loads(motion_key),
        },
        "native_1000x": json.loads(native_key),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("artifacts/finetune_mis20s1/evaluation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/finetune_mis20s1/evaluation/a3_baseline_holdout_recovered.json"
        ),
    )
    args = parser.parse_args()
    result = rebuild(args.evaluation_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

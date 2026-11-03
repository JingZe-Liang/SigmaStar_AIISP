#!/usr/bin/env python3
"""Build the deterministic A9 unified-generalization decision summary."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_static_summary(payload: dict[str, Any]) -> dict[str, float]:
    branches = [
        payload["even_input_vs_odd_reference"],
        payload["odd_input_vs_even_reference"],
    ]
    return {
        "mean_psnr_db": fmean(branch["a3"]["psnr_db"] for branch in branches),
        "mean_high_frequency_ratio": fmean(
            branch["a3_high_frequency_ratio"] for branch in branches
        ),
        "mean_temporal_mad_dn": fmean(
            branch["a3_temporal_mad_dn"] for branch in branches
        ),
        "maximum_output_bias_abs_dn": max(
            abs(branch["a3_output_bias_dn"]) for branch in branches
        ),
        "minimum_legal_raw_fraction": min(
            branch["legal_raw_fraction"] for branch in branches
        ),
        "maximum_normalized_clip_fraction": max(
            branch["normalized_clip_fraction"] for branch in branches
        ),
    }


def _candidate_static_summary(payload: dict[str, Any]) -> dict[str, float]:
    branches = [
        payload["even_input_vs_odd_reference"],
        payload["odd_input_vs_even_reference"],
    ]
    return {
        "mean_psnr_db": fmean(
            branch["candidate"]["psnr_db"] for branch in branches
        ),
        "mean_high_frequency_ratio": fmean(
            branch["candidate_high_frequency_ratio"] for branch in branches
        ),
        "mean_temporal_mad_dn": fmean(
            branch["candidate_temporal_mad_dn"] for branch in branches
        ),
        "maximum_output_bias_abs_dn": max(
            abs(branch["candidate_output_bias_dn"]) for branch in branches
        ),
        "minimum_legal_raw_fraction": min(
            branch["legal_raw_fraction"] for branch in branches
        ),
        "maximum_normalized_clip_fraction": max(
            branch["normalized_clip_fraction"] for branch in branches
        ),
    }


def _with_delta(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, Any]:
    return {
        "baseline_a3": baseline,
        "candidate_a9": candidate,
        "delta_a9_minus_a3": {
            key: candidate[key] - baseline[key]
            for key in baseline
        },
    }


def _dynamic_summary(payload: dict[str, Any]) -> dict[str, Any]:
    baseline = payload["baseline_a3"]
    candidate = payload["candidate"]
    baseline_metrics = {
        "temporal_mad_dn": baseline["a3_temporal_mad_dn"],
        "motion_temporal_ratio": baseline["motion_temporal_ratio"],
        "static_temporal_ratio": baseline["static_temporal_ratio"],
        "spatial_high_frequency_ratio": baseline["spatial_high_frequency_ratio"],
        "mean_output_bias_dn": baseline["mean_output_bias_dn"],
        "legal_raw_fraction": baseline["legal_raw_fraction"],
        "normalized_clip_fraction": baseline["normalized_clip_fraction"],
    }
    candidate_metrics = {
        "temporal_mad_dn": candidate["candidate_temporal_mad_dn"],
        "motion_temporal_ratio": candidate["motion_temporal_ratio"],
        "static_temporal_ratio": candidate["static_temporal_ratio"],
        "spatial_high_frequency_ratio": candidate["spatial_high_frequency_ratio"],
        "mean_output_bias_dn": candidate["mean_output_bias_dn"],
        "legal_raw_fraction": candidate["legal_raw_fraction"],
        "normalized_clip_fraction": candidate["normalized_clip_fraction"],
    }
    return _with_delta(baseline_metrics, candidate_metrics)


def _video_entry(payload: dict[str, Any]) -> dict[str, Any]:
    video = Path(payload["video"])
    actual_sha256 = _sha256(video)
    if actual_sha256 != payload["video_sha256"]:
        raise ValueError(f"video hash mismatch: {video}")
    return {
        "path": str(video.resolve()),
        "sha256": actual_sha256,
        "frames": payload["frames"],
        "fps": payload["fps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts/a9_multi_sensor_generalization",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016",
    )
    parser.add_argument(
        "--a3-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/summary.json",
    )
    args = parser.parse_args()

    checkpoint_audit = _load_json(
        args.root / "training_smoke_4090_retry2_20261016/checkpoint_audit.json"
    )
    training_summary = _load_json(
        args.root / "training_smoke_4090_retry2_20261016/summary.json"
    )
    noise_summary = _load_json(
        args.root / "sc450ai_noise_profile_20261016/summary.json"
    )
    preflight = _load_json(args.root / "preflight_20261016.json")
    sc465 = _load_json(args.evaluation_dir / "sc465sl/evaluation.json")
    sc635 = _load_json(args.evaluation_dir / "sc635hai/evaluation.json")
    sc450 = _load_json(args.evaluation_dir / "sc450ai/evaluation.json")
    visual_review = _load_json(args.evaluation_dir / "visual_review.json")
    mis_video = _load_json(
        args.evaluation_dir / "mis20s1/fixed_isp_video/video_manifest.json"
    )

    candidate_sha256 = checkpoint_audit["candidate_checkpoint_sha256"]
    for sensor, evaluation in (
        ("sc465sl", sc465),
        ("sc635hai", sc635),
        ("sc450ai", sc450),
    ):
        if evaluation["candidate_checkpoint_sha256"] != candidate_sha256:
            raise ValueError(f"candidate checkpoint mismatch in {sensor}")

    a3_sha256 = _sha256(args.a3_checkpoint)
    if a3_sha256 != checkpoint_audit["a3_checkpoint_sha256"]:
        raise ValueError("A3 checkpoint hash mismatch")
    candidate_path = Path(training_summary["checkpoint"])
    if _sha256(candidate_path) != candidate_sha256:
        raise ValueError("A9 checkpoint hash mismatch")

    sc465_static = _with_delta(
        _baseline_static_summary(sc465["baseline_a3"]),
        _candidate_static_summary(sc465["candidate"]),
    )
    sc635_static = _with_delta(
        _baseline_static_summary(sc635["static_train_domain"]["baseline_a3"]),
        _candidate_static_summary(sc635["static_train_domain"]["candidate"]),
    )
    sensor_gates = {
        "sc465sl": bool(sc465["automatic_gate"]["passed"]),
        "sc635hai": bool(sc635["automatic_gate"]["passed"]),
        "sc450ai": bool(sc450["automatic_gate"]["passed"]),
    }
    unified_pass = (
        checkpoint_audit["status"] == "passed"
        and noise_summary["status"] == "passed"
        and preflight["status"] == "passed"
        and all(sensor_gates.values())
        and bool(visual_review["overall_visual_pass"])
    )
    if unified_pass:
        decision = "retain_as_optional_unified_checkpoint"
        reason = "All quantitative and fixed-ISP visual gates passed."
    else:
        decision = "reject_as_unified_generalization_checkpoint"
        reason = (
            "sc450ai failed held-out high-frequency recovery and fixed-ISP review "
            "confirmed that missing black-subject detail was not recovered."
        )

    replay_guard = sc635["mis20s1_replay_guard"]
    payload = {
        "schema": "a9-unified-generalization-final-v1",
        "date": "2026-10-16",
        "decision": {
            "passed": unified_pass,
            "action": decision,
            "reason": reason,
            "replace_a3_checkpoint": False,
            "continue_same_protocol_beyond_step_100": False,
            "candidate_deployment_eligible": False,
        },
        "checkpoints": {
            "a3_baseline": {
                "path": str(args.a3_checkpoint.resolve()),
                "sha256": a3_sha256,
                "status": "unchanged baseline",
            },
            "a9_step_100": {
                "path": str(candidate_path.resolve()),
                "sha256": candidate_sha256,
                "weights_evaluated": "ema",
                "status": "diagnostic artifact only",
            },
        },
        "training": {
            "steps": training_summary["steps"],
            "source_counts": training_summary["source_counts"],
            "trainable": training_summary["trainable"],
            "checkpoint_audit_status": checkpoint_audit["status"],
            "unauthorized_live_tensor_changes": checkpoint_audit["live_weights"][
                "unauthorized_tensor_count"
            ],
            "unauthorized_ema_tensor_changes": checkpoint_audit["ema_weights"][
                "unauthorized_tensor_count"
            ],
            "maximum_gradient_norm": checkpoint_audit["training_log"][
                "maximum_gradient_norm"
            ],
        },
        "input_validation": {
            "preflight_status": preflight["status"],
            "sc450ai_noise_profile_status": noise_summary["status"],
            "sc450ai_validation_observations": noise_summary[
                "validation_observations"
            ],
            "sc450ai_median_relative_sigma_error": noise_summary["authenticity"][
                "aggregate_median_relative_sigma_error"
            ],
            "sc450ai_p90_relative_sigma_error": noise_summary["authenticity"][
                "aggregate_p90_relative_sigma_error"
            ],
        },
        "sensor_results": {
            "sc465sl": {
                "automatic_gate_passed": sensor_gates["sc465sl"],
                "static_holdout": sc465_static,
            },
            "sc635hai": {
                "automatic_gate_passed": sensor_gates["sc635hai"],
                "static_source_tail": sc635_static,
                "dynamic_holdout": _dynamic_summary(sc635["dynamic_holdout"]),
            },
            "sc450ai": {
                "automatic_gate_passed": sensor_gates["sc450ai"],
                "failed_checks": [
                    name
                    for name, passed in sc450["automatic_gate"]["checks"].items()
                    if not passed
                ],
                "dynamic_holdout": _dynamic_summary(sc450["dynamic_holdout"]),
            },
            "mis20s1_replay": {
                "psnr_delta_db": replay_guard["delta"]["psnr_db"],
                "high_frequency_ratio_delta": replay_guard["delta"][
                    "high_frequency_ratio"
                ],
                "temporal_mad_delta_dn": replay_guard["delta"][
                    "temporal_mad_dn"
                ],
            },
        },
        "visual_review": visual_review,
        "fixed_isp_videos": {
            "mis20s1": _video_entry(mis_video),
            "sc465sl": _video_entry(sc465["fixed_isp_video"]),
            "sc635hai": _video_entry(sc635["fixed_isp_video"]),
            "sc450ai": _video_entry(sc450["fixed_isp_video"]),
        },
        "remaining_work": [
            "The final objective of robust cross-sensor generalization is not achieved because sc450ai remains unsupported.",
            "A new supervision/degradation design is required before any longer A9 training run; extending the same 100-step protocol is not approved by this result.",
        ],
    }
    _save_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

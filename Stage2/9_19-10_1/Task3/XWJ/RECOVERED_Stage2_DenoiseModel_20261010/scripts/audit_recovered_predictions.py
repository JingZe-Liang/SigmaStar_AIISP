#!/usr/bin/env python3
"""Identify anonymous recovered prediction arrays by their evaluation metrics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_motion_temporal_fidelity import _aligned_motion_mask  # noqa: E402
from evaluate_a5_motion_smoke import _motion_summary  # noqa: E402
from raw_fusion.finetune.a3 import _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.a4 import _read_aligned_video, evaluate_outputs  # noqa: E402
from raw_fusion.finetune.a5_native import (  # noqa: E402
    _load_native_reference,
    evaluate_native_outputs,
    read_native_video,
)


EXPECTED = {
    "645x_psnr_db": 29.51644030829074,
    "645x_motion_to_input_ratio": 0.48879945278167725,
    "native_psnr_db": 52.36373272976205,
    "native_rmse_dn": 9.864322662353516,
}


def _distance(value: float, expected: float) -> float:
    return abs(value - expected)


def audit(
    *,
    supervision_dir: Path,
    native_probe_dir: Path,
    candidates_645x: list[Path],
    candidates_native: list[Path],
) -> dict[str, object]:
    video_645, overlap_645 = _read_aligned_video(
        supervision_dir, "645x", 40, start_frame=160
    )
    motion_645 = _aligned_motion_mask(supervision_dir, "645x", 200)[160:200]
    reference = np.load(
        supervision_dir / "references/04_shdarkroom_10x_clean_reference.npy"
    )
    reference_report = _load_json(
        supervision_dir / "references/a1_reference_report.json"
    )
    black = _channel_vector(reference_report["black_levels"])
    audited_645x: list[dict[str, object]] = []
    for path in candidates_645x:
        recovered = np.load(path, mmap_mode="r")
        if recovered.shape[0] == 200:
            recovered = recovered[160:200]
        prediction = recovered.astype(np.float32)
        metrics = evaluate_outputs(
            prediction, video_645, reference, overlap_645, motion_645, black
        )
        motion = _motion_summary(prediction, video_645, motion_645)
        psnr = float(metrics["finetuned"]["psnr_db"])
        ratio = float(motion["motion_to_input_ratio"])
        audited_645x.append(
            {
                "path": str(path.resolve()),
                "shape": list(prediction.shape),
                "metrics": metrics,
                "motion": motion,
                "baseline_distance": {
                    "psnr_db": _distance(psnr, EXPECTED["645x_psnr_db"]),
                    "motion_to_input_ratio": _distance(
                        ratio, EXPECTED["645x_motion_to_input_ratio"]
                    ),
                },
                "exact_baseline_fingerprint": (
                    psnr == EXPECTED["645x_psnr_db"]
                    and ratio == EXPECTED["645x_motion_to_input_ratio"]
                ),
                "matches_baseline_tolerance": (
                    _distance(psnr, EXPECTED["645x_psnr_db"]) <= 1e-3
                    and _distance(
                        ratio, EXPECTED["645x_motion_to_input_ratio"]
                    )
                    <= 1e-4
                ),
            }
        )
        del prediction

    del video_645, overlap_645, motion_645, reference

    native_video, native_motion, native_black = read_native_video(
        supervision_dir, "1000x", 40
    )
    native_reference, native_valid = _load_native_reference(native_probe_dir)
    audited_native: list[dict[str, object]] = []
    for path in candidates_native:
        recovered = np.load(path, mmap_mode="r")
        if recovered.shape[0] == 200:
            recovered = recovered[:40]
        prediction = recovered.astype(np.float32)
        metrics = evaluate_native_outputs(
            prediction,
            native_video,
            native_reference,
            native_valid,
            native_motion,
            native_black,
        )
        psnr = float(metrics["finetuned"]["psnr_db"])
        rmse = float(metrics["finetuned"]["rmse_dn"])
        audited_native.append(
            {
                "path": str(path.resolve()),
                "shape": list(prediction.shape),
                "metrics": metrics,
                "baseline_distance": {
                    "psnr_db": _distance(psnr, EXPECTED["native_psnr_db"]),
                    "rmse_dn": _distance(rmse, EXPECTED["native_rmse_dn"]),
                },
                "exact_baseline_fingerprint": (
                    psnr == EXPECTED["native_psnr_db"]
                    and rmse == EXPECTED["native_rmse_dn"]
                ),
                "matches_baseline_tolerance": (
                    _distance(psnr, EXPECTED["native_psnr_db"]) <= 1e-3
                    and _distance(rmse, EXPECTED["native_rmse_dn"]) <= 1e-3
                ),
            }
        )
        del prediction

    return {
        "protocol": "recovered prediction fingerprint audit",
        "expected_a3_baseline": EXPECTED,
        "candidates_645x": audited_645x,
        "candidates_native_1000x": audited_native,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--native-probe-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe",
    )
    parser.add_argument(
        "--candidate-645x",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--candidate-native",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/recovery_candidate_audit.json",
    )
    args = parser.parse_args()
    result = audit(
        supervision_dir=args.supervision_dir,
        native_probe_dir=args.native_probe_dir,
        candidates_645x=args.candidate_645x,
        candidates_native=args.candidate_native,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

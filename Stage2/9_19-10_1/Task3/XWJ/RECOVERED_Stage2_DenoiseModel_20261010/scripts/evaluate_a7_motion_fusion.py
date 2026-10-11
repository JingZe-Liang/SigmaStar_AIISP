#!/usr/bin/env python3
"""Sweep deterministic A3/official-teacher fusion on the 645x holdout."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.a4 import _read_aligned_video, evaluate_outputs  # noqa: E402
from raw_fusion.finetune.a7_motion_fusion import (  # noqa: E402
    FlowMaskParams,
    build_flow_motion_weights,
    fuse_motion_regions,
    fusion_shift_summary,
    motion_temporal_summary,
)
from audit_motion_temporal_fidelity import _aligned_motion_mask  # noqa: E402


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one value is required")
    return values


def _render(frame: np.ndarray, black: np.ndarray) -> np.ndarray:
    dn = frame * (_WHITE_LEVEL - black[:, None, None]) + black[:, None, None]
    mosaic = np.empty((frame.shape[1] * 2, frame.shape[2] * 2), dtype=np.float32)
    mosaic[0::2, 0::2], mosaic[0::2, 1::2] = dn[0], dn[1]
    mosaic[1::2, 0::2], mosaic[1::2, 1::2] = dn[2], dn[3]
    raw16 = np.rint(np.clip(mosaic / _WHITE_LEVEL, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535.0
    return np.rint(np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 54), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        text,
        (18, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _save_previews(
    output_dir: Path,
    video: np.ndarray,
    baseline: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    black: np.ndarray,
    frame_offset: int,
) -> list[str]:
    ranking = np.argsort(weights.reshape(weights.shape[0], -1).mean(axis=1))[::-1]
    selected = sorted(int(value) for value in ranking[:6])
    paths: list[str] = []
    for index in selected:
        input_rgb = _label(_render(video[index], black), f"input frame {frame_offset + index}")
        baseline_rgb = _label(_render(baseline[index], black), "A3")
        teacher_rgb = _label(_render(teacher[index], black), "official teacher")
        candidate_rgb = _label(_render(candidate[index], black), "restricted fusion")
        mask = np.rint(np.clip(weights[index], 0.0, 1.0) * 255.0).astype(np.uint8)
        mask_rgb = cv2.applyColorMap(mask, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
        mask_rgb = cv2.resize(mask_rgb, (input_rgb.shape[1], input_rgb.shape[0]))
        mask_rgb = _label(mask_rgb, "fusion weight")
        panel = np.concatenate(
            (input_rgb, baseline_rgb, teacher_rgb, candidate_rgb, mask_rgb),
            axis=1,
        )
        path = output_dir / f"frame_{frame_offset + index:04d}_panel.png"
        cv2.imwrite(str(path), panel[:, :, ::-1])
        paths.append(str(path.resolve()))
    return paths


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start, frames = args.start_frame, args.frames
    video, overlap = _read_aligned_video(
        args.supervision_dir,
        "645x",
        frames,
        start_frame=start,
    )
    baseline_all = np.load(args.baseline_prediction, mmap_mode="r")
    teacher_all = np.load(args.teacher_prediction, mmap_mode="r")
    baseline = baseline_all[start : start + frames].astype(np.float32)
    teacher = teacher_all[start : start + frames].astype(np.float32)
    if baseline.shape != video.shape or teacher.shape != video.shape:
        raise ValueError(
            f"prediction shapes {baseline.shape}/{teacher.shape} do not match input {video.shape}"
        )

    motion = _aligned_motion_mask(
        args.supervision_dir,
        "645x",
        start + frames,
    )[start : start + frames]
    reference = np.load(
        args.supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy"
    )
    reference_report = _load_json(
        args.supervision_dir / "references" / "a1_reference_report.json"
    )
    black = _channel_vector(reference_report["black_levels"]).astype(np.float32)
    saturation_limits = (3800.0 - black) / (_WHITE_LEVEL - black)
    mask_params = FlowMaskParams(
        temporal_gap=args.temporal_gap,
        min_flow_px=args.min_flow_px,
        max_flow_px=args.max_flow_px,
        max_forward_backward_error_px=args.max_flow_error,
        max_photometric_error=args.max_photometric_error,
        seed_dilation_radius=args.seed_dilation_radius,
        feather_sigma=args.feather_sigma,
    )
    weights, mask_records = build_flow_motion_weights(
        video,
        motion,
        saturation_limits.astype(np.float32),
        mask_params,
    )
    np.save(args.output_dir / "fusion_weights_645x_holdout.npy", weights.astype(np.float16))

    baseline_metrics = evaluate_outputs(
        baseline,
        video,
        reference,
        overlap,
        motion,
        black,
    )
    baseline_motion = motion_temporal_summary(baseline, video, motion)
    sweep: list[dict[str, object]] = []
    preview_candidate: np.ndarray | None = None
    for alpha in _parse_float_list(args.alphas):
        for clip_dn in _parse_float_list(args.residual_clips_dn):
            candidate = fuse_motion_regions(
                baseline,
                teacher,
                weights,
                black,
                alpha,
                clip_dn,
            )
            candidate_metrics = evaluate_outputs(
                candidate,
                video,
                reference,
                overlap,
                motion,
                black,
            )
            candidate_motion = motion_temporal_summary(candidate, video, motion)
            row = {
                "alpha": alpha,
                "residual_clip_dn": clip_dn,
                "static_metrics": candidate_metrics,
                "motion": candidate_motion,
                "motion_ratio_gain": (
                    candidate_motion["motion_to_input_ratio"]
                    - baseline_motion["motion_to_input_ratio"]
                ),
                "shift_from_a3": fusion_shift_summary(
                    candidate,
                    baseline,
                    motion,
                    black,
                ),
            }
            sweep.append(row)
            if alpha == args.preview_alpha and clip_dn == args.preview_residual_clip_dn:
                preview_candidate = candidate.copy()

    if preview_candidate is None:
        raise ValueError("preview alpha/clip pair must be included in the sweep")
    prediction_path = args.output_dir / "prediction_645x_holdout_preview.npy"
    np.save(prediction_path, preview_candidate.astype(np.float16))
    preview_paths = _save_previews(
        args.output_dir,
        video,
        baseline,
        teacher,
        preview_candidate,
        weights,
        black,
        start,
    )
    sweep.sort(key=lambda row: float(row["motion_ratio_gain"]), reverse=True)
    result: dict[str, object] = {
        "protocol": "A7 deterministic restricted motion-region fusion",
        "frame_range": [start, start + frames],
        "baseline_prediction": str(args.baseline_prediction.resolve()),
        "teacher_prediction": str(args.teacher_prediction.resolve()),
        "mask_params": asdict(mask_params),
        "mask_summary": {
            "strict_confidence_fraction_mean": float(
                np.mean([row["strict_confidence_fraction"] for row in mask_records])
            ),
            "fusion_weight_fraction_mean": float(
                np.mean([row["fusion_weight_fraction"] for row in mask_records])
            ),
            "fusion_weight_mean": float(weights.mean()),
            "static_nonzero_weights": int(np.count_nonzero(weights[motion < 0.5])),
        },
        "baseline_static_metrics": baseline_metrics,
        "baseline_motion": baseline_motion,
        "sweep": sweep,
        "preview_setting": {
            "alpha": args.preview_alpha,
            "residual_clip_dn": args.preview_residual_clip_dn,
            "prediction": str(prediction_path.resolve()),
            "panels": preview_paths,
        },
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--baseline-prediction",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/prediction_645x.npy",
    )
    parser.add_argument(
        "--teacher-prediction",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/old_teacher_645x.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/a7_motion_fusion",
    )
    parser.add_argument("--start-frame", type=int, default=160)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--residual-clips-dn", default="32,64,96")
    parser.add_argument("--preview-alpha", type=float, default=0.3)
    parser.add_argument("--preview-residual-clip-dn", type=float, default=64.0)
    parser.add_argument("--temporal-gap", type=int, default=4)
    parser.add_argument("--min-flow-px", type=float, default=1.0)
    parser.add_argument("--max-flow-px", type=float, default=30.0)
    parser.add_argument("--max-flow-error", type=float, default=1.5)
    parser.add_argument("--max-photometric-error", type=float, default=0.08)
    parser.add_argument("--seed-dilation-radius", type=int, default=3)
    parser.add_argument("--feather-sigma", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate(args)
    print(
        json.dumps(
            {
                "mask_summary": result["mask_summary"],
                "baseline_motion": result["baseline_motion"],
                "best_motion_rows": result["sweep"][:3],
                "preview_setting": result["preview_setting"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

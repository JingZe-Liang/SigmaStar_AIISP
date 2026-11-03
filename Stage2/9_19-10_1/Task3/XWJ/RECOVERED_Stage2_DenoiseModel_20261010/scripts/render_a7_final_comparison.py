#!/usr/bin/env python3
"""Render unified fixed-ISP panels for the recovered A7 fusion variants."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.a7_motion_fusion import fuse_motion_regions  # noqa: E402


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one frame is required")
    return values


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    coordinates = _parse_csv_ints(value)
    if len(coordinates) != 4:
        raise ValueError("crop must be x0,y0,x1,y1")
    x0, y0, x1, y1 = coordinates
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("crop coordinates must define a positive image region")
    return x0, y0, x1, y1


def _parse_settings(value: str) -> list[tuple[str, float, float]]:
    settings: list[tuple[str, float, float]] = []
    for item in value.split(","):
        name, alpha, clip_dn = item.strip().split(":")
        settings.append((name, float(alpha), float(clip_dn)))
    if not settings:
        raise ValueError("at least one setting is required")
    return settings


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


def _write_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image[:, :, ::-1]):
        raise OSError(f"failed to write {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = _parse_csv_ints(args.frames)
    full_frames = set(_parse_csv_ints(args.full_frames))
    if not full_frames.issubset(frames):
        raise ValueError("full frames must be included in the review frame list")
    crop = _parse_crop(args.crop)
    settings = _parse_settings(args.settings)

    baseline = np.load(args.baseline_prediction, mmap_mode="r")
    teacher = np.load(args.teacher_prediction, mmap_mode="r")
    if baseline.shape != teacher.shape:
        raise ValueError(f"prediction shapes do not match: {baseline.shape} and {teacher.shape}")

    report = _load_json(args.reference_report)
    black = _channel_vector(report["black_levels"]).astype(np.float32)
    variants: list[tuple[str, Path, np.ndarray]] = []
    for label, path in (
        ("r3", args.r3_weights),
        ("r7", args.r7_weights),
        ("r11", args.r11_weights),
    ):
        weights = np.load(path, mmap_mode="r")
        expected = (args.holdout_frames, baseline.shape[2], baseline.shape[3])
        if weights.shape != expected:
            raise ValueError(f"{label} weights have shape {weights.shape}, expected {expected}")
        variants.append((label, path, weights))

    x0, y0, x1, y1 = crop
    outputs: list[dict[str, object]] = []
    for frame in frames:
        local_index = frame - args.holdout_start
        if frame < 0 or frame >= baseline.shape[0]:
            raise ValueError(f"frame {frame} is outside prediction range [0, {baseline.shape[0]})")
        if local_index < 0 or local_index >= args.holdout_frames:
            raise ValueError(
                f"frame {frame} is outside A7 holdout range "
                f"[{args.holdout_start}, {args.holdout_start + args.holdout_frames})"
            )

        baseline_frame = baseline[frame : frame + 1].astype(np.float32)
        teacher_frame = teacher[frame : frame + 1].astype(np.float32)
        baseline_rgb = _render(baseline_frame[0], black)
        teacher_rgb = _render(teacher_frame[0], black)
        if x1 > baseline_rgb.shape[1] or y1 > baseline_rgb.shape[0]:
            raise ValueError(f"crop {crop} exceeds rendered shape {baseline_rgb.shape[:2][::-1]}")

        weight_tiles: list[np.ndarray] = []
        for label, _, weights in variants:
            weight = weights[local_index].astype(np.float32)
            weight_u8 = np.rint(np.clip(weight, 0.0, 1.0) * 255.0).astype(np.uint8)
            weight_rgb = cv2.applyColorMap(weight_u8, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            weight_rgb = cv2.resize(
                weight_rgb,
                (baseline_rgb.shape[1], baseline_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            weight_tiles.append(_label(weight_rgb, f"{label} fusion weight"))
        weight_path = args.output_dir / f"frame_{frame:04d}_weights.png"
        _write_rgb(weight_path, np.concatenate(weight_tiles, axis=1))
        outputs.append({"frame": frame, "kind": "weights", "path": str(weight_path.resolve())})

        for setting_name, alpha, clip_dn in settings:
            labels = ["A3 baseline", "old teacher (diagnostic only)"]
            rendered_raw = [baseline_rgb, teacher_rgb]
            for label, _, weights in variants:
                candidate = fuse_motion_regions(
                    baseline_frame,
                    teacher_frame,
                    weights[local_index : local_index + 1].astype(np.float32),
                    black,
                    alpha,
                    clip_dn,
                )
                labels.append(f"{label} a={alpha:g} c={clip_dn:g}")
                rendered_raw.append(_render(candidate[0], black))

            full_path = args.output_dir / f"frame_{frame:04d}_{setting_name}_full.png"
            crop_path = args.output_dir / f"frame_{frame:04d}_{setting_name}_person_crop.png"
            crop_tiles = [
                _label(tile[y0:y1, x0:x1], label)
                for tile, label in zip(rendered_raw, labels, strict=True)
            ]
            _write_rgb(crop_path, np.concatenate(crop_tiles, axis=1))
            if frame in full_frames:
                full_tiles = [
                    _label(tile, label)
                    for tile, label in zip(rendered_raw, labels, strict=True)
                ]
                _write_rgb(full_path, np.concatenate(full_tiles, axis=1))
                outputs.append(
                    {
                        "frame": frame,
                        "kind": "full",
                        "setting": setting_name,
                        "alpha": alpha,
                        "residual_clip_dn": clip_dn,
                        "path": str(full_path.resolve()),
                    }
                )
            outputs.append(
                {
                    "frame": frame,
                    "kind": "person_crop",
                    "setting": setting_name,
                    "alpha": alpha,
                    "residual_clip_dn": clip_dn,
                    "path": str(crop_path.resolve()),
                }
            )

    manifest: dict[str, object] = {
        "protocol": "A7 recovered existing-data unified fixed-ISP review",
        "holdout_range": [args.holdout_start, args.holdout_start + args.holdout_frames],
        "frames": frames,
        "full_frames": sorted(full_frames),
        "crop_xyxy": list(crop),
        "settings": [
            {"name": name, "alpha": alpha, "residual_clip_dn": clip_dn}
            for name, alpha, clip_dn in settings
        ],
        "sources": {
            "baseline_prediction": str(args.baseline_prediction.resolve()),
            "teacher_prediction": str(args.teacher_prediction.resolve()),
            "reference_report": str(args.reference_report.resolve()),
            "weights": {
                label: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for label, path, _ in variants
            },
        },
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    evaluation_root = REPO_ROOT / "artifacts/finetune_mis20s1/evaluation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-prediction",
        type=Path,
        default=evaluation_root / "prediction_645x.npy",
    )
    parser.add_argument(
        "--teacher-prediction",
        type=Path,
        default=evaluation_root / "old_teacher_645x.npy",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/supervision/references/a1_reference_report.json",
    )
    parser.add_argument(
        "--r3-weights",
        type=Path,
        default=evaluation_root / "a7_motion_fusion/fusion_weights_645x_holdout.npy",
    )
    parser.add_argument(
        "--r7-weights",
        type=Path,
        default=evaluation_root / "a7_motion_fusion_r7/fusion_weights_645x_holdout.npy",
    )
    parser.add_argument(
        "--r11-weights",
        type=Path,
        default=evaluation_root / "a7_motion_fusion_r11/fusion_weights_645x_holdout.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=evaluation_root / "a7_motion_fusion_final_review_20261011",
    )
    parser.add_argument("--holdout-start", type=int, default=160)
    parser.add_argument("--holdout-frames", type=int, default=40)
    parser.add_argument("--frames", default="167,168,169,170,171,172,173")
    parser.add_argument("--full-frames", default="168,170,172")
    parser.add_argument("--crop", default="520,150,1150,850")
    parser.add_argument("--settings", default="preview:0.3:64,strongest:0.4:96")
    args = parser.parse_args()
    manifest = render(args)
    print(json.dumps({"outputs": len(manifest["outputs"]), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

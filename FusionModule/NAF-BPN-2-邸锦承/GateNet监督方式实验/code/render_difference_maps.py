from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from data import CODE_MAX, FRAME_COUNT, HEIGHT, WIDTH, NR_BLACK_LEVEL, discover_sequences, read_candidate


def read_raw(path: Path) -> np.ndarray:
    mapped = np.memmap(path, dtype="<u2", mode="r", shape=(HEIGHT, WIDTH))
    return np.asarray(mapped).copy()


def make_heatmap(
    difference: np.ndarray,
    scale: float,
    dead_zone: float,
    frame_index: int,
    only_positive: bool = False,
    only_negative: bool = False,
) -> np.ndarray:
    magnitude = np.clip(
        (np.abs(difference) - dead_zone) / max(scale - dead_zone, 1e-8),
        0.0,
        1.0,
    )
    intensity = np.rint(magnitude * 255.0).astype(np.uint8)
    heatmap = np.zeros((*difference.shape, 3), dtype=np.uint8)
    lower = difference < -dead_zone
    higher = difference > dead_zone
    if not only_positive:
        heatmap[lower, 1] = intensity[lower]
    if not only_negative:
        heatmap[higher, 2] = intensity[higher]
    cv2.rectangle(heatmap, (0, 0), (620, 72), (0, 0, 0), thickness=-1)
    cv2.putText(
        heatmap,
        f"645x frame {frame_index:04d}: AI fused - 2DNR",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        heatmap,
        (
            "black=not lower  green=AI lower than 2DNR"
            if only_negative
            else (
                "black=not higher  red=AI higher than 2DNR"
                if only_positive
                else f"black=|diff|<{dead_zone:.5f}  green=lower  red=higher"
            )
        ),
        (16, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return heatmap


def main() -> int:
    parser = argparse.ArgumentParser(description="Render signed AI-minus-2DNR RAW difference maps")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", default="645x")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--dead-zone", type=float, default=0.001, help="不标注的线性差异阈值")
    parser.add_argument("--only-positive", action="store_true", help="只标注 AI 高于 2DNR 的红色区域")
    parser.add_argument("--only-negative", action="store_true", help="只标注 AI 低于 2DNR 的绿色区域")
    args = parser.parse_args()
    if not 50.0 < args.percentile <= 100.0:
        raise ValueError("percentile 必须在 (50, 100] 范围内")
    if args.dead_zone < 0:
        raise ValueError("dead-zone 不能为负数")
    if args.only_positive and args.only_negative:
        raise ValueError("only-positive 与 only-negative 不能同时使用")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    sequence = next(
        item
        for item in discover_sequences(
            data_root,
            (args.sequence,),
            None,
            str(config.get("cfa_pattern", "RGGB")),
            int(config.get("source_black_level", 252)),
            int(config.get("dnr_black_level", NR_BLACK_LEVEL)),
            int(config.get("white_level", CODE_MAX)),
        )
        if item.name == args.sequence
    )
    manifest_path = args.manifest or (args.inference_root / "video_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence_manifest = next(item for item in manifest["sequences"] if item["name"] == args.sequence)
    frame_indices = [int(index) for index in sequence_manifest["random_frame_indices"]]
    if not frame_indices:
        raise ValueError("manifest 中没有 random_frame_indices")

    differences: dict[int, np.ndarray] = {}
    all_values = []
    for frame_index in frame_indices:
        ai_path = args.inference_root / args.sequence / f"out_{frame_index:04d}.raw"
        if not ai_path.is_file():
            raise FileNotFoundError(f"缺少 AI RAW: {ai_path}")
        dnr2 = read_candidate(sequence.dnr2_paths[frame_index]).astype(np.float32)
        ai = read_raw(ai_path).astype(np.float32)
        dnr2_linear = np.clip(
            (dnr2 - sequence.dnr_black_level) / max(sequence.white_level - sequence.dnr_black_level, 1),
            0.0,
            1.0,
        )
        ai_linear = np.clip(
            (ai - sequence.dnr_black_level) / max(sequence.white_level - sequence.dnr_black_level, 1),
            0.0,
            1.0,
        )
        difference = ai_linear - dnr2_linear
        differences[frame_index] = difference
        all_values.append(np.abs(difference).reshape(-1))

    scale = float(np.percentile(np.concatenate(all_values), args.percentile))
    scale = max(scale, 1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    for frame_index, difference in differences.items():
        output = args.output / f"frame_{frame_index:04d}_ai_minus_2dnr.png"
        if not cv2.imwrite(
            str(output),
            make_heatmap(
                difference,
                scale,
                args.dead_zone,
                frame_index,
                args.only_positive,
                args.only_negative,
            ),
        ):
            raise RuntimeError(f"写入失败: {output}")

    summary = {
        "sequence": args.sequence,
        "operation": "AI_fused - 2DNR",
        "domain": "black-level-corrected linear RAW",
        "frame_indices": frame_indices,
        "scale_percentile": args.percentile,
        "symmetric_scale": scale,
        "dead_zone": args.dead_zone,
        "only_positive": args.only_positive,
        "only_negative": args.only_negative,
        "color_mapping": {
            "black": (
                "AI fused is not lower than 2DNR or is inside the dead zone"
                if args.only_negative
                else (
                    "AI fused is not higher than 2DNR or is inside the dead zone"
                    if args.only_positive
                    else f"absolute linear difference below {args.dead_zone}"
                )
            ),
            "green": "AI fused lower than 2DNR" if not args.only_positive else "not used",
            "red": "AI fused higher than 2DNR" if not args.only_negative else "not used",
            "saturation": "absolute difference larger than the symmetric scale is clipped",
        },
        "output_files": [f"frame_{index:04d}_ai_minus_2dnr.png" for index in frame_indices],
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

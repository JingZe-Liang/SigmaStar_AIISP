#!/usr/bin/env python3
"""Build inspectable eight-frame dynamic Noise2Noise windows."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from raw_fusion.finetune.a3 import _channel_vector, _load_json
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.stream_io import open_stream
from build_dynamic_n2n_probe import (
    _PACKED_SHAPE,
    _aligned_motion_mask,
    _build_pair,
    _parse_crop,
    _read_aligned_frame,
    _render_packed,
)


def build_window_probe(
    *,
    supervision_dir: Path,
    gain: str,
    starts: list[int],
    temporal_frames: int,
    temporal_gap: int,
    frame_range_start: int,
    frame_range_end: int,
    crop: tuple[int, int, int, int],
    output_dir: Path,
    min_flow_px: float = 0.0,
) -> dict[str, object]:
    if temporal_frames != 8:
        raise ValueError("the RViDeformer training contract requires 8 temporal frames")
    if frame_range_start < 0 or frame_range_end <= frame_range_start:
        raise ValueError("invalid allowed frame range")
    if any(
        start < frame_range_start
        or start + temporal_frames - 1 + temporal_gap >= frame_range_end
        for start in starts
    ):
        raise ValueError(f"a window leaves the allowed frame range [{frame_range_start}, {frame_range_end})")

    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    reference_black = _channel_vector(report["black_levels"]).astype(np.float32)
    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    needed_frames = sorted({frame for start in starts for frame in range(start, start + temporal_frames + temporal_gap)})
    loaded = {frame: _read_aligned_frame(reader, adapter, manifest, reference_black, frame) for frame in needed_frames}
    loaded_motion = {frame: _aligned_motion_mask(manifest, frame) for frame in needed_frames}
    x, y, width, height = crop
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for start in starts:
        source_frames: list[np.ndarray] = []
        target_frames: list[np.ndarray] = []
        confidence_masks: list[np.ndarray] = []
        frame_stats: list[dict[str, float]] = []
        for offset in range(temporal_frames):
            source_index = start + offset
            target_index = source_index + temporal_gap
            source, target, confidence, stats = _build_pair(
                loaded[source_index],
                loaded[target_index],
                loaded_motion[source_index],
                loaded_motion[target_index],
                reference_black,
                min_flow_px=min_flow_px,
            )
            source_frames.append(source[:, y:y + height, x:x + width])
            target_frames.append(target[:, y:y + height, x:x + width])
            confidence_masks.append(confidence[y:y + height, x:x + width])
            frame_stats.append(stats)

        window_name = f"window_{start:04d}_{start + temporal_frames - 1:04d}_gap{temporal_gap}"
        sample_path = output_dir / f"{window_name}.npz"
        np.savez_compressed(
            sample_path,
            source=np.stack(source_frames).astype(np.float32),
            target=np.stack(target_frames).astype(np.float32),
            confidence=np.stack(confidence_masks).astype(np.uint8),
        )
        source_preview_full = _render_packed(loaded[start], reference_black)
        target_preview_full = _render_packed(loaded[start + temporal_gap], reference_black)
        source_preview = source_preview_full[2 * y:2 * (y + height), 2 * x:2 * (x + width)]
        target_preview = target_preview_full[2 * y:2 * (y + height), 2 * x:2 * (x + width)]
        mask = cv2.resize(confidence_masks[0].astype(np.uint8) * 255, (source_preview.shape[1], source_preview.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_preview = np.zeros_like(source_preview)
        mask_preview[:, :, 1] = mask
        cv2.imwrite(str(output_dir / f"{window_name}_preview.png"), np.hstack((source_preview, target_preview, mask_preview))[:, :, ::-1])
        records.append({
            "start_frame": start,
            "source_frame_range": [start, start + temporal_frames - 1],
            "target_frame_range": [start + temporal_gap, start + temporal_frames - 1 + temporal_gap],
            "sample": str(sample_path.resolve()),
            "preview": str((output_dir / f"{window_name}_preview.png").resolve()),
            "crop_packed_xywh": list(crop),
            "frame_stats": frame_stats,
            "mean_confidence_fraction": float(np.mean([stats["confidence_fraction"] for stats in frame_stats])),
            "mean_confidence_fraction_of_dynamic": float(np.mean([stats["confidence_fraction_of_dynamic"] for stats in frame_stats])),
        })

    result: dict[str, object] = {
        "protocol": "eight-frame dynamic Noise2Noise window probe",
        "gain": gain,
        "temporal_frames": temporal_frames,
        "temporal_gap": temporal_gap,
        "source_domain": "A1 exposure-mapped packed RGGB normalized domain",
        "windows": records,
        "allowed_frame_range": [frame_range_start, frame_range_end],
        "target_semantics": "each later noisy frame is flow-warped to the corresponding source frame; targets are not clean references",
        "confidence_definition": {"flow_magnitude_px_gte": min_flow_px},
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--gain", default="645x")
    parser.add_argument("--starts", default="30,46,140")
    parser.add_argument("--temporal-frames", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=4)
    parser.add_argument("--frame-range-start", type=int, default=0)
    parser.add_argument("--frame-range-end", type=int, default=160)
    parser.add_argument("--crop", default="650,100,240,420")
    parser.add_argument("--min-flow-px", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/dynamic_window_probe_645x"))
    args = parser.parse_args()
    starts = [int(value.strip()) for value in args.starts.split(",") if value.strip()]
    result = build_window_probe(
        supervision_dir=args.supervision_dir,
        gain=args.gain,
        starts=starts,
        temporal_frames=args.temporal_frames,
        temporal_gap=args.temporal_gap,
        frame_range_start=args.frame_range_start,
        frame_range_end=args.frame_range_end,
        crop=_parse_crop(args.crop),
        output_dir=args.output_dir,
        min_flow_px=args.min_flow_px,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

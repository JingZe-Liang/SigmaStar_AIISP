#!/usr/bin/env python3
"""Build a small motion-aware Noise2Noise target probe from one RAW stream."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json, _normalise_dn
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.stream_io import open_stream, pack_frame


_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
_PACKED_SHAPE = (4, 540, 960)


def _parse_pairs(value: str, train_end_frame: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        start_text, end_text = item.strip().split(":")
        start, end = int(start_text), int(end_text)
        if end <= start:
            raise ValueError(f"dynamic probe pair must move forward in time: {item}")
        if start < 0 or end >= train_end_frame:
            raise ValueError(f"pair {item} overlaps the excluded holdout [{train_end_frame}, 200)")
        pairs.append((start, end))
    if not pairs:
        raise ValueError("at least one pair is required")
    return pairs


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError("crop must be x,y,width,height with positive size")
    x, y, width, height = values
    if x < 0 or y < 0 or x + width > _PACKED_SHAPE[2] or y + height > _PACKED_SHAPE[1]:
        raise ValueError(f"crop {values} is outside packed shape {_PACKED_SHAPE[1:]}")
    return values


def _read_aligned_frame(
    reader: object,
    adapter: object,
    manifest: dict,
    reference_black: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    packed = pack_frame(reader.read_frame(frame_index), adapter).astype(np.float32)
    slopes = _channel_vector(manifest["exposure_fit"]["slopes"])
    intercepts = _channel_vector(manifest["exposure_fit"]["intercepts"])
    mapped = slopes[:, None, None] * packed + intercepts[:, None, None]
    normalized = _normalise_dn(mapped, reference_black)
    shift_x = int(manifest["registration"]["integer_shift_x"])
    shift_y = int(manifest["registration"]["integer_shift_y"])
    aligned = np.zeros(_PACKED_SHAPE, dtype=np.float32)
    y0, y1 = max(0, -shift_y), min(_PACKED_SHAPE[1], _PACKED_SHAPE[1] - shift_y)
    x0, x1 = max(0, -shift_x), min(_PACKED_SHAPE[2], _PACKED_SHAPE[2] - shift_x)
    aligned[:, y0:y1, x0:x1] = normalized[:, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
    return aligned


def _aligned_motion_mask(manifest: dict, frame_index: int) -> np.ndarray:
    source = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")[frame_index, 0, ::2, ::2]
    shift_x = int(manifest["registration"]["integer_shift_x"])
    shift_y = int(manifest["registration"]["integer_shift_y"])
    aligned = np.zeros((_PACKED_SHAPE[1], _PACKED_SHAPE[2]), dtype=np.float32)
    y0, y1 = max(0, -shift_y), min(_PACKED_SHAPE[1], _PACKED_SHAPE[1] - shift_y)
    x0, x1 = max(0, -shift_x), min(_PACKED_SHAPE[2], _PACKED_SHAPE[2] - shift_x)
    aligned[y0:y1, x0:x1] = source[y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
    return aligned


def _flow_warp(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = frame.shape[-2:]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = np.stack([
        cv2.remap(channel, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        for channel in frame
    ])
    return warped


def _flow_confidence(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_gray = np.clip(np.mean(source[1:3], axis=0), 0.0, 1.0).astype(np.float32)
    target_gray = np.clip(np.mean(target[1:3], axis=0), 0.0, 1.0).astype(np.float32)
    source_gray_u8 = np.rint(source_gray * 255.0).astype(np.uint8)
    target_gray_u8 = np.rint(target_gray * 255.0).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setVariationalRefinementIterations(2)
    forward = dis.calc(source_gray_u8, target_gray_u8, None)
    backward = dis.calc(target_gray_u8, source_gray_u8, None)
    warped_target = _flow_warp(target, forward)
    warped_backward = _flow_warp(backward.transpose(2, 0, 1), forward).transpose(1, 2, 0)
    flow_error = np.linalg.norm(forward + warped_backward, axis=-1)
    photometric_error = np.abs(source_gray - np.mean(warped_target[1:3], axis=0))
    flow_magnitude = np.linalg.norm(forward, axis=-1)
    return warped_target, forward, flow_error, photometric_error, flow_magnitude


def _render_packed(frame: np.ndarray, black: np.ndarray) -> np.ndarray:
    dn = frame * (_WHITE_LEVEL - black[:, None, None]) + black[:, None, None]
    mosaic = np.empty((_PACKED_SHAPE[1] * 2, _PACKED_SHAPE[2] * 2), dtype=np.float32)
    mosaic[0::2, 0::2], mosaic[0::2, 1::2] = dn[0], dn[1]
    mosaic[1::2, 0::2], mosaic[1::2, 1::2] = dn[2], dn[3]
    raw16 = np.rint(np.clip(mosaic / _WHITE_LEVEL, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535.0
    return np.rint(np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)


def _build_pair(
    source: np.ndarray,
    target: np.ndarray,
    source_motion: np.ndarray,
    target_motion: np.ndarray,
    reference_black: np.ndarray,
    min_flow_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    warped, forward, flow_error, photometric_error, flow_magnitude = _flow_confidence(source, target)
    height, width = flow_error.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    target_motion_warped = cv2.remap(target_motion, grid_x + forward[..., 0], grid_y + forward[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    source_valid = np.all(source < ((3800.0 - reference_black) / (_WHITE_LEVEL - reference_black))[:, None, None], axis=0)
    target_valid = np.all(target < ((3800.0 - reference_black) / (_WHITE_LEVEL - reference_black))[:, None, None], axis=0)
    inside = (grid_x + forward[..., 0] >= 1.0) & (grid_x + forward[..., 0] < width - 1.0)
    inside &= (grid_y + forward[..., 1] >= 1.0) & (grid_y + forward[..., 1] < height - 1.0)
    dynamic = (source_motion >= 0.5) | (target_motion_warped >= 0.5)
    confidence = dynamic & source_valid & target_valid & inside
    confidence &= flow_error < 1.5
    confidence &= photometric_error < 0.08
    confidence &= flow_magnitude >= min_flow_px
    confidence &= flow_magnitude < 30.0
    if not confidence.any():
        raise ValueError("dynamic pair has no pixels passing the confidence mask")
    stats = {
        "dynamic_fraction": float(dynamic.mean()),
        "confidence_fraction": float(confidence.mean()),
        "confidence_fraction_of_dynamic": float(confidence.sum() / max(dynamic.sum(), 1)),
        "flow_error_median": float(np.median(flow_error[confidence])) if confidence.any() else float("nan"),
        "flow_error_p95": float(np.percentile(flow_error[confidence], 95)) if confidence.any() else float("nan"),
        "photometric_error_median": float(np.median(photometric_error[confidence])) if confidence.any() else float("nan"),
        "photometric_error_p95": float(np.percentile(photometric_error[confidence], 95)) if confidence.any() else float("nan"),
        "flow_magnitude_median": float(np.median(flow_magnitude[confidence])) if confidence.any() else float("nan"),
        "flow_magnitude_p95": float(np.percentile(flow_magnitude[confidence], 95)) if confidence.any() else float("nan"),
        "min_flow_px": float(min_flow_px),
        "photometric_error_before_warp_median": float(np.median(np.abs(np.mean(source[1:3], axis=0) - np.mean(target[1:3], axis=0))[dynamic])) if dynamic.any() else float("nan"),
    }
    return source, warped, confidence, stats


def build_probe(
    *,
    supervision_dir: Path,
    gain: str,
    pairs: list[tuple[int, int]],
    crop: tuple[int, int, int, int],
    output_dir: Path,
    min_flow_px: float = 0.0,
) -> dict[str, object]:
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    reference_black = _channel_vector(report["black_levels"]).astype(np.float32)
    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames = sorted({frame for pair in pairs for frame in pair})
    loaded = {frame: _read_aligned_frame(reader, adapter, manifest, reference_black, frame) for frame in frames}
    loaded_motion = {frame: _aligned_motion_mask(manifest, frame) for frame in frames}
    x, y, width, height = crop
    pair_records: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for start, end in pairs:
        source, warped, confidence, stats = _build_pair(
            loaded[start], loaded[end], loaded_motion[start], loaded_motion[end], reference_black,
            min_flow_px=min_flow_px,
        )
        pair_name = f"pair_{start:04d}_{end:04d}"
        np.savez_compressed(
            output_dir / f"{pair_name}.npz",
            source=source[:, y:y + height, x:x + width].astype(np.float32),
            target=warped[:, y:y + height, x:x + width].astype(np.float32),
            confidence=confidence[y:y + height, x:x + width].astype(np.uint8),
        )
        source_rgb = _render_packed(source, reference_black)
        target_rgb = _render_packed(warped, reference_black)
        mask_rgb = np.zeros_like(source_rgb)
        mask_rgb[::2, ::2, 1] = cv2.resize(confidence.astype(np.uint8) * 255, (source_rgb.shape[1] // 2, source_rgb.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        preview = np.hstack((source_rgb, target_rgb, mask_rgb))
        cv2.imwrite(str(output_dir / f"{pair_name}_preview.png"), preview[:, :, ::-1])
        pair_records.append({
            "pair": [start, end],
            "sample": str((output_dir / f"{pair_name}.npz").resolve()),
            "preview": str((output_dir / f"{pair_name}_preview.png").resolve()),
            "crop_packed_xywh": list(crop),
            "stats": stats,
        })
    result: dict[str, object] = {
        "protocol": "dynamic Noise2Noise target probe",
        "gain": gain,
        "source_domain": "A1 exposure-mapped packed RGGB normalized domain",
        "pairs": pair_records,
        "holdout_exclusion": {"train_end_frame_exclusive": 160, "excluded_range": [160, 200]},
        "confidence_definition": {
            "forward_backward_error_px_lt": 1.5,
            "warped_green_photometric_error_normalized_lt": 0.08,
            "flow_magnitude_px_lt": 30.0,
            "flow_magnitude_px_gte": min_flow_px,
            "source_and_target_saturation_cut": 3800,
        },
        "target_semantics": "later-frame noisy observation warped to current-frame coordinates; not a clean reference",
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--gain", default="645x")
    parser.add_argument("--pairs", default="34:38,50:54,148:152")
    parser.add_argument("--train-end-frame", type=int, default=160)
    parser.add_argument("--crop", default="650,100,240,420")
    parser.add_argument("--min-flow-px", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/evaluation/dynamic_n2n_probe_645x"))
    args = parser.parse_args()
    pairs = _parse_pairs(args.pairs, args.train_end_frame)
    crop = _parse_crop(args.crop)
    result = build_probe(
        supervision_dir=args.supervision_dir,
        gain=args.gain,
        pairs=pairs,
        crop=crop,
        output_dir=args.output_dir,
        min_flow_px=args.min_flow_px,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

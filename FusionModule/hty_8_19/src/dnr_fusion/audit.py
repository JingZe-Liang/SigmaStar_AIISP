from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import load_config, project_root, validate_scene
from .dataset_fast import open_scene_streams


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def _green_preview(frame: np.ndarray, source_shift: int = 0) -> np.ndarray:
    if source_shift:
        frame = np.right_shift(frame, source_shift)
    green = 0.5 * (
        frame[0::2, 1::2].astype(np.float32)
        + frame[1::2, 0::2].astype(np.float32)
    )
    green = cv2.resize(green, (240, 135), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(green, (0, 0), 1.2)


def _correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    x = left[mask].astype(np.float64)
    y = right[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def _temporal_lag(source: np.ndarray, candidate: np.ndarray) -> dict:
    scores = []
    for lag in range(-10, 11):
        low = max(1, 1 - lag)
        high = min(source.shape[0], source.shape[0] - lag)
        candidate_delta = candidate[low:high] - candidate[low - 1 : high - 1]
        source_delta = source[low + lag : high + lag] - source[low + lag - 1 : high + lag - 1]
        mask = np.abs(source_delta) > np.quantile(np.abs(source_delta), 0.75)
        scores.append({
            "lag": lag,
            "correlation": _correlation(candidate_delta, source_delta, mask),
        })
    best = max(scores, key=lambda item: item["correlation"])
    return {"best_lag": best["lag"], "best_correlation": best["correlation"], "scores": scores}


def _stream_stats(stream, sample_frames: list[int], shift: int = 0) -> dict:
    sample = np.stack([stream.frame(index) for index in sample_frames])
    stored = sample if not shift else np.left_shift(sample, shift)
    values = sample
    return {
        "path": str(stream.path.resolve()),
        "bytes": stream.path.stat().st_size,
        "frames": stream.frame_count,
        "sample_frames": sample_frames,
        "stored_min": int(stored.min()),
        "stored_max": int(stored.max()),
        "value_min": int(values.min()),
        "value_max": int(values.max()),
        "value_quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)
        },
        "stored_low4_nonzero_fraction": float(np.mean((stored & 15) != 0)),
    }


def _candidate_concat_check(path: Path, frame_count: int, height: int, width: int) -> dict:
    branch = path.stem
    directory = path.parent / branch
    checks = []
    stream = np.memmap(path, dtype="<u2", mode="r", shape=(frame_count, height, width))
    for index in (0, frame_count - 1):
        frame_path = directory / f"out_{index:04d}.raw"
        if not frame_path.is_file():
            checks.append({"frame": index, "exists": False, "exact_match": False})
            continue
        frame = np.fromfile(frame_path, dtype="<u2").reshape(height, width)
        checks.append({
            "frame": index,
            "exists": True,
            "exact_match": bool(np.array_equal(frame, stream[index])),
        })
    return {"directory": str(directory.resolve()), "checks": checks}


def _threshold_samples(streams, frame_count: int) -> dict:
    temporal_values = []
    disagreement_values = []
    for index in range(1, frame_count, 4):
        current = streams.denoised.frame(index).astype(np.float32)
        previous = streams.denoised.frame(index - 1).astype(np.float32)
        fused = streams.fused.frame(index).astype(np.float32)
        current_green = 0.5 * (current[0::2, 1::2] + current[1::2, 0::2])
        previous_green = 0.5 * (previous[0::2, 1::2] + previous[1::2, 0::2])
        fused_green = 0.5 * (fused[0::2, 1::2] + fused[1::2, 0::2])
        temporal = cv2.GaussianBlur(current_green - previous_green, (0, 0), 1.2)
        disagreement = cv2.GaussianBlur(fused_green - current_green, (0, 0), 1.2)
        temporal_values.append(np.abs(temporal)[::4, ::4].ravel())
        disagreement_values.append(np.abs(disagreement)[::4, ::4].ravel())
    temporal = np.concatenate(temporal_values)
    disagreement = np.concatenate(disagreement_values)
    temporal_median, temporal_q75 = np.quantile(temporal, (0.5, 0.75))
    disagreement_median, disagreement_q75 = np.quantile(disagreement, (0.5, 0.75))
    return {
        "temporal_blur_dn_quantiles": {
            str(q): float(np.quantile(temporal, q))
            for q in (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)
        },
        "candidate_blur_dn_quantiles": {
            str(q): float(np.quantile(disagreement, q))
            for q in (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)
        },
        "recommended_motion_threshold_dn": float(
            temporal_median + 8.0 * (temporal_q75 - temporal_median)
        ),
        "recommended_disagreement_threshold_dn": float(
            disagreement_median + 12.0 * (disagreement_q75 - disagreement_median)
        ),
    }


def audit_scene(config: dict, scene_name: str) -> dict:
    scene = validate_scene(config, scene_name)
    streams = open_scene_streams(config, scene_name)
    raw = config["raw"]
    sample_frames = sorted({0, 1, 2, 10, 50, 100, 150, streams.frame_count - 1})
    previews = {
        "source": np.stack([
            _green_preview(streams.source.frame(index)) for index in range(streams.frame_count)
        ]),
        "denoised": np.stack([
            _green_preview(streams.denoised.frame(index)) for index in range(streams.frame_count)
        ]),
        "fused": np.stack([
            _green_preview(streams.fused.frame(index)) for index in range(streams.frame_count)
        ]),
    }
    return {
        "scene": scene_name,
        "configured_motion_threshold_dn": scene["motion_threshold_dn"],
        "configured_disagreement_threshold_dn": scene["disagreement_threshold_dn"],
        "streams": {
            "source": _stream_stats(
                streams.source, sample_frames, shift=int(raw["source_shift"])
            ),
            "denoised": _stream_stats(streams.denoised, sample_frames),
            "fused": _stream_stats(streams.fused, sample_frames),
        },
        "concat_checks": {
            "denoised": _candidate_concat_check(
                Path(scene["denoised"]), streams.frame_count, raw["height"], raw["width"]
            ),
            "fused": _candidate_concat_check(
                Path(scene["fused"]), streams.frame_count, raw["height"], raw["width"]
            ),
        },
        "temporal_alignment": {
            "denoised_to_source": _temporal_lag(previews["source"], previews["denoised"]),
            "fused_to_source": _temporal_lag(previews["source"], previews["fused"]),
        },
        "threshold_estimation": _threshold_samples(streams, streams.frame_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = args.output or project_root(config) / "outputs" / "audit" / "data_audit.json"
    report = {
        "raw_contract": config["raw"],
        "scenes": {name: audit_scene(config, name) for name in sorted(config["scenes"])},
        "conclusions": {
            "source_storage": "uint16 little-endian with 12-bit values left-aligned by 4 bits",
            "candidate_storage": "uint16 little-endian with direct 12-bit values",
            "expected_frames_per_scene": 200,
            "clean_ground_truth_available": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


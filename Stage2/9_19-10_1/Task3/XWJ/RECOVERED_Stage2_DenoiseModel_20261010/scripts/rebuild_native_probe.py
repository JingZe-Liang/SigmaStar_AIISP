#!/usr/bin/env python3
"""Rebuild the native 1000x temporal-reference probe from recovered inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.reference import (  # noqa: E402
    _SIGNAL_MARGIN_DN,
    _accumulate_stream,
    _channel_vector,
    _robust_sigma,
    _save_array,
    _save_json,
)
from raw_fusion.finetune.stream_io import CHANNEL_NAMES  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(
    *,
    supervision_dir: Path,
    output_dir: Path,
    chunk_frames: int,
    recovered_candidate: Path | None,
) -> dict[str, object]:
    manifest_path = supervision_dir / "stream_manifests/04_shdarkroom_1000x.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motion_path = Path(manifest["motion"]["mask_path"])
    if not motion_path.is_absolute():
        motion_path = REPO_ROOT / motion_path
    manifest["motion"]["mask_path"] = str(motion_path.resolve())

    temporal = _accumulate_stream(
        manifest,
        chunk_frames,
        use_motion_mask=True,
        normalize_exposure=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "temporal_mean_native_dn.npy": temporal.mean,
        "even_mean_native_dn.npy": temporal.even_mean,
        "odd_mean_native_dn.npy": temporal.odd_mean,
        "confidence.npy": temporal.confidence,
        "valid_region.npy": temporal.valid_region,
        "exposure_scales.npy": temporal.exposure_scales,
    }
    for name, array in arrays.items():
        _save_array(output_dir / name, array)

    black = _channel_vector(temporal.black_levels)
    signal = temporal.mean > (black + _SIGNAL_MARGIN_DN)
    split_residual = (temporal.even_mean - temporal.odd_mean) / 2.0
    split_sigma: dict[str, float] = {}
    split_robust_sigma: dict[str, float] = {}
    split_pixels: dict[str, int] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        selected = temporal.valid_region & signal[channel]
        values = split_residual[channel][selected]
        split_sigma[name] = float(np.std(values))
        split_robust_sigma[name] = _robust_sigma(values)
        split_pixels[name] = int(values.size)

    candidate_comparison: dict[str, object] | None = None
    if recovered_candidate is not None and recovered_candidate.is_file():
        candidate = np.load(recovered_candidate, mmap_mode="r")
        if candidate.shape != temporal.mean.shape:
            candidate_comparison = {
                "path": str(recovered_candidate.resolve()),
                "shape_match": False,
                "candidate_shape": list(candidate.shape),
            }
        else:
            difference = np.abs(candidate.astype(np.float32, copy=False) - temporal.mean)
            candidate_comparison = {
                "path": str(recovered_candidate.resolve()),
                "shape_match": True,
                "exact_equal": bool(np.array_equal(candidate, temporal.mean)),
                "max_abs_difference_dn": float(difference.max()),
                "mean_abs_difference_dn": float(difference.mean()),
                "sha256": _sha256(recovered_candidate),
            }

    report: dict[str, object] = {
        "protocol": "recovered native 1000x motion-aware temporal reference",
        "source_manifest": str(manifest_path.resolve()),
        "source_raw": str(Path(manifest["path"]).resolve()),
        "source_motion_mask": str(motion_path.resolve()),
        "frames_used": temporal.frames_total,
        "shape": list(temporal.mean.shape),
        "black_levels": temporal.black_levels,
        "static_confidence_threshold": 0.9,
        "motion_aware_valid_fraction": float(temporal.valid_region.mean()),
        "confidence_mean": float(temporal.confidence.mean()),
        "exposure_scale_min": {
            name: float(temporal.exposure_scales[:, channel].min())
            for channel, name in enumerate(CHANNEL_NAMES)
        },
        "exposure_scale_max": {
            name: float(temporal.exposure_scales[:, channel].max())
            for channel, name in enumerate(CHANNEL_NAMES)
        },
        "split_half_residual_sigma_dn": split_sigma,
        "split_half_robust_sigma_dn": split_robust_sigma,
        "split_half_signal_pixels": split_pixels,
        "recovered_candidate_comparison": candidate_comparison,
        "outputs": {},
    }
    report["outputs"] = {
        name: {
            "path": str((output_dir / name).resolve()),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(output_dir / name),
        }
        for name, array in arrays.items()
    }
    _save_json(output_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe",
    )
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument(
        "--recovered-candidate",
        type=Path,
        default=REPO_ROOT
        / ".recovery/evidence/unclassified_artifacts/temporal_mean_native_dn.npy",
    )
    args = parser.parse_args()
    report = rebuild(
        supervision_dir=args.supervision_dir,
        output_dir=args.output_dir,
        chunk_frames=args.chunk_frames,
        recovered_candidate=args.recovered_candidate,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

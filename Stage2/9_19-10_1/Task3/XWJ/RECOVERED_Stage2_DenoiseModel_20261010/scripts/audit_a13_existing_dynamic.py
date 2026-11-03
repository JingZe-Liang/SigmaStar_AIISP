"""Inventory known company data and screen the existing MIS20S1 dynamic source on CPU."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST_SHA256 = "f76739c8538b42d15582ab063e40d673ddeff94cdc81b1f264808f48db7ad1df"
sys.path.insert(0, str(ROOT / "src"))

from raw_fusion.finetune import a7_capture
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.stream_io import RawStreamReader, pack_frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(dataset_root: Path, manifest_path: Path, output_dir: Path) -> dict:
    dataset_root, manifest_path, output_dir = (path.resolve() for path in (dataset_root, manifest_path, output_dir))
    output_dir.relative_to(ROOT / "artifacts")
    if output_dir.exists():
        raise FileExistsError("output directory already exists; prior audits must be preserved")
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("source manifest differs from the frozen historical MIS20S1 10x identity")
    source = json.loads(manifest_bytes)
    if (source["group"], source["scene"], source["gain_label"]) != ("04", "shdarkroom", "10x"):
        raise ValueError("this diagnostic is restricted to the historical MIS20S1 10x source")
    if (source["width"], source["height"], source["frames_total"]) != (1920, 1080, 200):
        raise ValueError("historical MIS20S1 10x geometry or frame count differs")
    raw_path = Path(source["path"]).resolve()
    raw_path.relative_to(dataset_root)
    adapter = resolve_adapter(source["group"], source["scene"], source["gain_label"])
    if (source["bit_shift"], source["cfa"], source["reader_kind"]) != (adapter.bit_shift, adapter.cfa, "single_file"):
        raise ValueError("source manifest disagrees with the established RAW adapter")
    code_paths = [Path(__file__).resolve(), Path(a7_capture.__file__).resolve(), ROOT / "src/raw_fusion/finetune/stream_io.py", ROOT / "src/raw_fusion/finetune/adapters.py", manifest_path]
    provenance = {str(path): sha256(path) for path in code_paths}
    provenance[str(manifest_path)] = manifest_sha256
    raw_before = raw_path.stat()
    reader = RawStreamReader(raw_path, int(source["width"]), int(source["height"]))
    if reader.frames != source["frames_total"] or reader.frames < 2:
        raise ValueError("source frame count differs from the recorded stream manifest")
    limits = a7_capture.CaptureValidationThresholds()
    indices = np.unique(np.linspace(0, reader.frames - 2, min(limits.sample_pairs, reader.frames - 1)).round().astype(int))
    frame_hashes, pairs, background = {}, [], []
    aligned, saturated, samples = 0, 0, 0
    for index in indices:
        raw_a, raw_b = reader.read_frame(int(index)), reader.read_frame(int(index) + 1)
        first, second = pack_frame(raw_a, adapter).astype(np.float32), pack_frame(raw_b, adapter).astype(np.float32)
        for frame_index, raw, packed in ((int(index), raw_a, first), (int(index) + 1, raw_b, second)):
            frame_hashes[str(frame_index)] = hashlib.sha256(raw.tobytes()).hexdigest()
            aligned += int(np.count_nonzero(raw % 16 == 0))
            saturated += int(np.count_nonzero(packed >= 4095))
            samples += raw.size
            background.append(float(np.percentile(a7_capture._block_means(packed.mean(axis=0), limits.motion_block_size), 20)))
        noise = a7_capture._quiet_pair_noise(first, second, limits.noise_block_size)
        pairs.append({
            "indices": [int(index), int(index) + 1], "quiet_pair_noise_dn": noise,
            "motion_fraction": a7_capture._motion_fraction(first, second, noise, limits.motion_block_size),
        })
    levels = np.asarray(background)
    noise = float(np.median([pair["quiet_pair_noise_dn"] for pair in pairs]))
    motion = np.asarray([pair["motion_fraction"] for pair in pairs])
    metrics = {
        "frames": reader.frames, "sampled_adjacent_pairs": len(pairs),
        "sampled_unique_frames": len(frame_hashes), "high12_alignment_fraction": aligned / samples,
        "quiet_pair_noise_dn": noise, "saturation_fraction": saturated / samples,
        "background_cv": float(np.std(levels) / max(levels.mean(), 1e-6)),
        "background_max_deviation": float(np.max(np.abs(levels - np.median(levels))) / max(np.median(levels), 1e-6)),
        "active_pair_fraction": float(np.mean(motion >= limits.min_motion_fraction)),
        "mean_motion_fraction": float(np.mean(motion)),
    }
    checks = {
        "minimum_frames": reader.frames >= limits.min_frames_per_clip,
        "sampled_high12_alignment": metrics["high12_alignment_fraction"] == 1,
        "quiet_pair_noise": noise <= limits.max_target_noise_dn,
        "saturation": metrics["saturation_fraction"] <= limits.max_saturation_fraction,
        "background_cv": metrics["background_cv"] <= limits.max_background_cv,
        "background_max_deviation": metrics["background_max_deviation"] <= limits.max_background_deviation,
        "active_motion": metrics["active_pair_fraction"] >= limits.min_active_pair_fraction,
        "motion_area_lower_bound": metrics["mean_motion_fraction"] >= limits.min_motion_fraction,
        "motion_area_upper_bound": metrics["mean_motion_fraction"] <= limits.max_motion_fraction,
    }
    raw_files = [{"relative_path": str(path.relative_to(dataset_root)), "size_bytes": path.stat().st_size} for path in sorted(dataset_root.rglob("*.raw")) if path.is_file()]
    json_files = [str(path.relative_to(dataset_root)) for path in sorted(dataset_root.rglob("*.json")) if path.is_file()]
    raw_after = raw_path.stat()
    stat_keys = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_dev", "st_ino")
    if any(getattr(raw_before, key) != getattr(raw_after, key) for key in stat_keys):
        raise RuntimeError("source RAW changed during audit")
    if any(sha256(path) != provenance[str(path)] for path in code_paths):
        raise RuntimeError("audit source code or source manifest changed during audit")
    result = {
        "schema": "a13_existing_dynamic_readiness_v1", "date": "2026-10-30", "dataset_access": "read_only_cpu",
        "status": "existing_source_not_new_independent_dynamic_pilot", "training_ready": False,
        "scope": "Only the configured company-data root was inventoried. No claim about unknown external directories.",
        "dataset_root": str(dataset_root), "raw_file_count": len(raw_files),
        "raw_files_by_group": dict(sorted(Counter(row["relative_path"].split("/")[0] for row in raw_files).items())),
        "json_files_in_dataset_root": json_files,
        "source": {"path": str(raw_path), "manifest": str(manifest_path), "size_bytes": raw_before.st_size,
                   "mtime_ns": raw_before.st_mtime_ns, "device": raw_before.st_dev, "inode": raw_before.st_ino,
                   "sampled_frame_sha256": frame_hashes, "full_raw_sha256": None},
        "filename_metadata_unverified_capture_claims": source["filename_metadata"],
        "unknown_capture_controls": ["fps", "nr_disabled", "auto_exposure", "auto_gain", "auto_white_balance", "tripod_fixed", "hardware_bypass_evidence", "motion_blur_review"],
        "metrics": metrics, "thresholds": asdict(limits), "signal_screen_checks": checks, "signal_screen_all_pass": all(checks.values()),
        "screen_definition": "A7 quiet adjacent-pair and block-motion screening; no physical-scene, clean-target, NR-bypass, blur, or generalization certification.",
        "limitations": ["Quiet-pair noise is not total target error and may miss fixed-pattern or correlated noise.", "Only the recorded adjacent pairs are screened; no whole-stream cleanliness claim.", "The one-scene 200-frame source was already used in A3/A6 development; temporal splitting cannot create independent scenes."],
        "required_before_new_training": ["Independent multi-scene dynamic targets and scene-level train/validation split", "Actual capture controls and provenance", "Automatic A7 checks followed by fixed-ISP visual target review"],
        "provenance_sha256": provenance, "sampled_pairs": pairs,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in (("readiness.json", result), ("raw_inventory.json", raw_files)):
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/finetune_mis20s1/supervision/stream_manifests/04_shdarkroom_10x.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_root, args.manifest, args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "training_ready", "raw_file_count", "metrics", "signal_screen_checks")}, indent=2))

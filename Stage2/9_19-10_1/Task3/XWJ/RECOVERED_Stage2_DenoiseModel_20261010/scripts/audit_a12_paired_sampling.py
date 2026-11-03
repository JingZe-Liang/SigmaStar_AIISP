"""Read every sample in the two-cycle A12 paired experiment on CPU."""

import argparse
from collections import Counter
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import A3Config  # noqa: E402
from raw_fusion.finetune.a12_paired_sampling import A12PairedSamplingDataset  # noqa: E402
from raw_fusion.finetune.a12_sampling import A12SamplingDataset  # noqa: E402
from raw_fusion.finetune.a12_train import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.parent != REPO_ROOT / "artifacts" or not output_dir.name.startswith("a12_"):
        parser.error("output-dir must be a new artifacts/a12_* directory in this repository")
    if output_dir.exists():
        parser.error("output-dir already exists; preserve the old audit and choose a new directory")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    period = int(payload.get("samples_per_epoch", 512))
    if period != 512 or payload.get("batch_size", 1) != 1 or payload.get("real_fraction", 0.5) != 0.5:
        parser.error("the paired experiment requires a 512-sample cycle, batch 1 and a 1:1 source ratio")
    names = {field.name for field in fields(A3Config)}
    config_data = {key: value for key, value in payload.items() if key in names}
    os.chdir(REPO_ROOT)
    for key in ("dataset_root", "supervision_dir", "noise_dir", "output_dir", "checkpoint"):
        config_data[key] = str(Path(config_data[key]).resolve())
    override_value = payload.get("sample_overrides_path")
    override_sha = payload.get("expected_sample_overrides_sha256")
    if (override_value is None) != (override_sha is None):
        parser.error("sample_overrides_path and expected_sample_overrides_sha256 must be supplied together")
    override_path = Path(override_value).resolve() if override_value is not None else None
    if override_path is not None and sha256_file(override_path) != override_sha:
        parser.error("sample override manifest SHA-256 mismatch")
    config_data.update(steps=period * 2, device="cpu", resume_checkpoint=None)
    config = A3Config(**config_data)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    report = {
        "schema": "a12_paired_sampling_audit_v1", "status": "running",
        "training_started": False, "config_path": str(config_path),
        "source_config_sha256": sha256_file(config_path), "config": asdict(config),
        "samples_per_epoch": period, "cycles": 2,
        "sample_overrides_path": str(override_path) if override_path is not None else None,
        "expected_sample_overrides_sha256": override_sha,
        "sampling_policy": "Repeat the first-cycle source/record/gain schedule; only geometry/window/noise refresh differs.",
        "mask_policy": (
            "Retain A3 masks and every nonempty low-coverage sample. Apply only manifest-recorded offline replacements of audited empty refresh samples. No runtime retries; fail on any remaining empty mask."
            if override_path is not None else
            "Retain A3 masks, report low coverage, and fail on any empty mask; no rejection sampling."
        ),
        "data_identity_policy": "Hash sampled input/target/mask tensors and small metadata; RAW files are identified by path, size and mtime, not full-file hashes.",
        "limitations": [
            "This audit does not train or establish image-quality improvement.",
            "Targets remain static and samples use A3's existing noise and validity models.",
            "The source schedule is frozen for this ablation; this is not a full reproduction of the old A3 sampler.",
            "Sample hashes describe data at audit time; later mutation requires a new audit.",
        ],
    }
    report_path = output_dir / "audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    exit_code = 1
    try:
        datasets = {
            mode: A12PairedSamplingDataset(config, mode, samples_per_epoch=period,
                                          plan_overrides_path=override_path)
            for mode in ("fixed", "refresh")
        }
        original = A12SamplingDataset(config, samples_per_epoch=period)
        plans = {
            mode: [dataset.sample_plan(index) for index in range(len(dataset))]
            for mode, dataset in datasets.items()
        }
        first = plans["fixed"][:period]
        source_schedule = [
            {"position": plan.epoch_index, "source": plan.source,
             "record_index": plan.record_index, "gain": plan.gain}
            for plan in first
        ]
        schedule_hash = hashlib.sha256(json.dumps(source_schedule, sort_keys=True).encode()).hexdigest()
        checks = {
            "first_cycle_matches_a12_plans": all(
                plans[mode][index] == original.sample_plan(index)
                for mode in datasets for index in range(period)
            ),
            "identical_repeating_source_record_gain": all(
                (plan.source, plan.record_index, plan.gain)
                == (first[index % period].source, first[index % period].record_index, first[index % period].gain)
                for mode in datasets for index, plan in enumerate(plans[mode])
            ),
            "exact_source_ratio_every_cycle": all(
                Counter(plan.source for plan in plans[mode][start:start + period])
                == {"real": period // 2, "synthetic": period // 2}
                for mode in datasets for start in (0, period)
            ),
        }
        snapshots: dict[str, list[dict]] = {mode: [] for mode in datasets}
        for index in range(period * 2):
            for mode, dataset in datasets.items():
                inputs, targets, masks = dataset[index]
                arrays = (inputs, targets, masks)
                snapshots[mode].append({
                    "plan": asdict(plans[mode][index]),
                    "tensor_sha256": [hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for tensor in arrays],
                    "valid_fraction": float(masks.mean()),
                    "finite": all(bool(torch.isfinite(tensor).all()) for tensor in arrays),
                    "expected_shapes": inputs.shape == targets.shape == (8, 4, 64, 64) and masks.shape == (8, 64, 64),
                    "input_range_0_1": bool(((inputs >= 0) & (inputs <= 1)).all()),
                    "target_range_0_1": bool(((targets >= 0) & (targets <= 1)).all()),
                    "binary_masks": bool(((masks == 0) | (masks == 1)).all()),
                    "static_target": bool(torch.all(targets == targets[0])),
                })
            if (index + 1) % 128 == 0:
                print(f"[a12 paired audit] {index + 1}/{period * 2} indices read for both modes", flush=True)

        checks["first_cycle_tensor_identity"] = all(
            snapshots["fixed"][index]["tensor_sha256"] == snapshots["refresh"][index]["tensor_sha256"]
            for index in range(period)
        )
        checks["fixed_cycle_tensor_replay"] = all(
            snapshots["fixed"][index]["tensor_sha256"] == snapshots["fixed"][index + period]["tensor_sha256"]
            for index in range(period)
        )
        changed = {
            source: [index for index in range(period)
                     if first[index].source == source
                     and snapshots["refresh"][index]["tensor_sha256"][0]
                     != snapshots["refresh"][index + period]["tensor_sha256"][0]]
            for source in ("real", "synthetic")
        }
        checks["refresh_changes_real_inputs"] = bool(changed["real"])
        checks["refresh_changes_synthetic_inputs"] = bool(changed["synthetic"])
        checks["valid_tensor_contents"] = all(
            all(snapshot[key] for key in ("finite", "expected_shapes", "input_range_0_1",
                                         "target_range_0_1", "binary_masks", "static_target"))
            for records in snapshots.values() for snapshot in records
        )
        coverage = {}
        for mode, records in snapshots.items():
            coverage[mode] = {
                "samples_read": len(records),
                "minimum_valid_fraction": min(row["valid_fraction"] for row in records),
                "mean_valid_fraction": sum(row["valid_fraction"] for row in records) / len(records),
                "zero_valid_sample_indices": [index for index, row in enumerate(records) if row["valid_fraction"] == 0],
                "below_1_percent_sample_indices": [index for index, row in enumerate(records) if row["valid_fraction"] < 0.01],
                "below_6_25_percent_sample_indices": [index for index, row in enumerate(records) if row["valid_fraction"] < 0.0625],
            }
        checks["no_empty_supervision"] = all(not item["zero_valid_sample_indices"] for item in coverage.values())

        metadata = {}
        if override_path is not None:
            metadata[str(override_path)] = sha256_file(override_path)
        raw_files = []
        for record in datasets["fixed"].real:
            manifest_path = record.manifest_path.resolve()
            metadata[str(manifest_path)] = sha256_file(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_path = Path(manifest["path"]).resolve()
            stat = raw_path.stat()
            raw_files.append({"path": str(raw_path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        for path in (
            Path(config.supervision_dir) / "references/a1_reference_manifest.json",
            Path(config.supervision_dir) / "references/a1_reference_report.json",
            Path(config.noise_dir) / "noise_model.json",
            REPO_ROOT / "src/raw_fusion/finetune/a12_paired_sampling.py",
            REPO_ROOT / "src/raw_fusion/finetune/a12_sampling.py",
            REPO_ROOT / "src/raw_fusion/finetune/a3.py",
            Path(__file__).resolve(),
        ):
            metadata[str(path.resolve())] = sha256_file(path)
        with (output_dir / "sample_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump({"source_schedule": source_schedule, "modes": snapshots}, stream, indent=2)
            stream.write("\n")
        report.update({
            "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "source_schedule_sha256": schedule_hash, "coverage": coverage,
            "refresh_changed_input_counts": {source: len(indices) for source, indices in changed.items()},
            "metadata_sha256": metadata, "raw_files": raw_files,
            "sample_manifest": str(output_dir / "sample_manifest.json"),
            "sample_manifest_sha256": sha256_file(output_dir / "sample_manifest.json"),
        })
        exit_code = 0 if report["status"] == "passed" else 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

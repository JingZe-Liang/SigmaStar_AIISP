"""Resolve audited empty real samples offline with a recorded seed sequence."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import A3Config, _PACKED_HEIGHT, _PACKED_WIDTH  # noqa: E402
from raw_fusion.finetune.a12_paired_sampling import (  # noqa: E402
    A12PairedSamplingDataset, paired_sampling_config_identity,
)
from raw_fusion.finetune.a12_train import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    output = args.output_dir.resolve()
    if output.parent != REPO_ROOT / "artifacts" or not output.name.startswith("a12_"):
        parser.error("output-dir must be a new artifacts/a12_* directory")
    if output.exists():
        parser.error("output-dir already exists; preserve the previous resolution")
    os.chdir(REPO_ROOT)
    torch.set_num_threads(1)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failures = sorted(key for key, passed in audit["checks"].items() if not passed)
    if audit["status"] != "failed" or failures != ["no_empty_supervision"]:
        parser.error("only an audit whose sole failure is empty supervision can be resolved")
    if audit.get("sample_overrides_path") is not None:
        parser.error("the resolver requires an original audit without plan overrides")
    if audit["coverage"]["fixed"]["zero_valid_sample_indices"]:
        parser.error("this protocol does not change the fixed group or shared first cycle")
    indices = audit["coverage"]["refresh"]["zero_valid_sample_indices"]
    period = audit["samples_per_epoch"]
    if period != 512 or not indices or any(not period <= index < 2 * period for index in indices):
        parser.error("only audited second-cycle empty refresh samples are eligible")
    samples_path = Path(audit["sample_manifest"]).resolve()
    if sha256_file(samples_path) != audit["sample_manifest_sha256"]:
        parser.error("the failed audit's sample manifest changed")
    snapshots = json.loads(samples_path.read_text(encoding="utf-8"))["modes"]["refresh"]
    config = A3Config(**audit["config"])
    dataset = A12PairedSamplingDataset(config, "refresh", samples_per_epoch=period)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "a12_empty_plan_overrides_v1", "status": "running",
        "sampling_config": paired_sampling_config_identity(config), "samples_per_epoch": period,
        "source_audit": str(audit_path), "source_audit_sha256": sha256_file(audit_path),
        "source_sample_manifest": str(samples_path), "source_sample_manifest_sha256": sha256_file(samples_path),
        "seed_recipe": "SeedSequence([config.seed, epoch, epoch_index, 12, 2, attempt]).generate_state(1, uint64)[0]",
        "attempt_numbering": "0..63", "maximum_attempts_per_sample": 64,
        "selection_rule": "Accept the first candidate with strictly positive valid coverage; retain all nonempty samples regardless of coverage.",
        "training_performed": False, "runtime_retry_permitted": False, "overrides": [],
        "resolver_sha256": sha256_file(Path(__file__).resolve()),
    }
    succeeded = False
    try:
        for index in indices:
            original = dataset.sample_plan(index)
            snapshot = snapshots[index]
            if original.source != "real" or snapshot["plan"] != asdict(original) or snapshot["valid_fraction"] != 0.0:
                raise ValueError(f"index {index} is not the audited empty real plan")
            actual = dataset[index]
            actual_hashes = [hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for tensor in actual]
            if float(actual[2].mean()) != 0.0 or actual_hashes != snapshot["tensor_sha256"]:
                raise ValueError(f"audited empty sample changed at index {index}")
            record = dataset.real[original.record_index]
            entry = {
                "sampling_mode": "refresh", "sample_index": index,
                "original_plan": asdict(original), "original_valid_fraction": 0.0,
                "original_tensor_sha256": actual_hashes, "attempts": [],
            }
            report["overrides"].append(entry)
            for attempt in range(64):
                seed = int(np.random.SeedSequence(
                    [config.seed, original.epoch, original.epoch_index, 12, 2, attempt],
                ).generate_state(1, dtype=np.uint64)[0])
                rng = np.random.default_rng(seed)
                patch = config.patch_size
                y = int(rng.integers(max(0, -record.shift_y), min(_PACKED_HEIGHT - patch, _PACKED_HEIGHT - patch - record.shift_y) + 1))
                x = int(rng.integers(max(0, -record.shift_x), min(_PACKED_WIDTH - patch, _PACKED_WIDTH - patch - record.shift_x) + 1))
                start = int(rng.integers(0, record.frame_count - config.temporal_frames + 1))
                candidate = replace(original, crop_y=y, crop_x=x, frame_start=start, geometry_seed=seed)
                arrays = tuple(array.astype(np.float32, copy=False) for array in dataset._real_sample(record, np.random.default_rng(seed)))
                if not all(np.isfinite(array).all() for array in arrays):
                    raise ValueError(f"non-finite candidate at index {index}, attempt {attempt}")
                coverage = float(arrays[2].mean())
                hashes = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
                entry["attempts"].append({"attempt": attempt, "plan": asdict(candidate),
                                          "valid_fraction": coverage, "tensor_sha256": hashes})
                if coverage > 0.0:
                    entry.update(replacement_plan=asdict(candidate), replacement_valid_fraction=coverage,
                                 replacement_tensor_sha256=hashes, accepted_attempt=attempt)
                    break
            else:
                raise RuntimeError(f"all 64 offline candidates remain empty at index {index}")
            print(json.dumps({"index": index, "accepted_attempt": entry["accepted_attempt"],
                              "valid_fraction": entry["replacement_valid_fraction"]}), flush=True)
        report["status"] = "resolved"
        with (output / "overrides.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        succeeded = True
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        with (output / "resolution_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "output_dir": str(output)}), flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())

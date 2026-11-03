"""Audit A12 sample diversity and read-only CPU loading without training."""

import argparse
from collections import Counter
from dataclasses import asdict
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
from raw_fusion.finetune.a12_sampling import A12SamplingDataset, A12StepSampler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision")
    parser.add_argument("--noise-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/noise_model")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/a12_sampling_audit_20261029")
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--read-samples-per-epoch", type=int, default=8)
    parser.add_argument("--read-first-samples", type=int, help="Read the first N absolute samples instead of per-epoch samples")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--real-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs < 2 or not 2 <= args.read_samples_per_epoch <= args.samples_per_epoch:
        parser.error("epochs must be >= 2; read-samples-per-epoch must be in [2, samples-per-epoch]")
    if args.read_first_samples is not None and not 2 <= args.read_first_samples <= args.epochs * args.samples_per_epoch:
        parser.error("read-first-samples must be in [2, epochs * samples-per-epoch]")
    output_dir = args.output_dir.resolve()
    supervision_dir = args.supervision_dir.resolve()
    noise_dir = args.noise_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "audit.json"
    plans_path = output_dir / "sample_plans.json"
    if report_path.exists() or plans_path.exists():
        parser.error("output already exists; use a new output directory to preserve the audit")

    os.chdir(REPO_ROOT)
    torch.set_num_threads(1)
    config = A3Config(
        dataset_root="/HardDisk/jingzeliang/Data/Company_Dataset/dataset",
        supervision_dir=str(supervision_dir), noise_dir=str(noise_dir),
        output_dir=str(output_dir), checkpoint="not_loaded_cpu_sampling_audit",
        steps=args.epochs * args.samples_per_epoch, batch_size=1,
        seed=args.seed, real_fraction=args.real_fraction, device="cpu",
    )
    started = time.monotonic()
    report = {
        "schema_version": "a12_sampling_audit_v1",
        "status": "running", "training_started": False, "checkpoint_loaded": False,
        "config": asdict(config), "samples_per_epoch": args.samples_per_epoch,
        "validation_policy": "Existing A3 evaluation and validation sampling are unchanged; only the new training sampler is audited.",
        "limitations": [
            "This is a sampling/data-loading audit, not evidence of denoising quality improvement.",
            "The clean target, noise model, and static-only supervision remain those of A3.",
            "Absolute sample IDs reproduce inputs; full training determinism also requires restoring model, optimizer, EMA, and training RNG states.",
        ],
    }
    try:
        dataset = A12SamplingDataset(config, samples_per_epoch=args.samples_per_epoch)
        plans = [dataset.sample_plan(i) for i in range(len(dataset))]
        plans_path.write_text(json.dumps([asdict(plan) for plan in plans], indent=2) + "\n", encoding="utf-8")
        epoch_summaries = []
        for epoch in range(args.epochs):
            selected = plans[epoch * args.samples_per_epoch : (epoch + 1) * args.samples_per_epoch]
            sources = [plan.source for plan in selected]
            epoch_summaries.append({
                "epoch": epoch, "source_counts": dict(Counter(sources)),
                "gain_counts": dict(Counter(f"{plan.source}:{plan.gain}" for plan in selected)),
                "unique_crop_window_configurations": len({(p.source, p.gain, p.crop_y, p.crop_x, p.frame_start) for p in selected}),
                "source_switches": sum(a != b for a, b in zip(sources[:-1], sources[1:])),
            })

        comparisons = list(zip(plans[:args.samples_per_epoch], plans[args.samples_per_epoch : 2 * args.samples_per_epoch]))
        checks = {
            "exact_epoch_ratio": all(s["source_counts"]["real"] == dataset.real_samples_per_epoch for s in epoch_summaries),
            "interleaved_sources": all(s["source_switches"] > 1 for s in epoch_summaries),
            "epoch_changes_crop": any((a.crop_y, a.crop_x) != (b.crop_y, b.crop_x) for a, b in comparisons),
            "epoch_changes_real_window": any(a.frame_start != b.frame_start for a, b in comparisons if a.source == "real"),
            "epoch_changes_synthetic_noise_seed": all(a.noise_seed != b.noise_seed for a, b in comparisons if a.source == "synthetic"),
            "no_full_frame_synthetic_cache": not hasattr(dataset, "_synthetic_temporal"),
        }
        if args.read_first_samples is not None:
            read_indices = list(range(args.read_first_samples))
        else:
            read_indices = [epoch * args.samples_per_epoch + offset for epoch in range(args.epochs)
                            for offset in range(args.read_samples_per_epoch)]
        snapshots = []
        for index in read_indices:
            plan = plans[index]
            arrays = dataset[index]
            snapshot = {
                "plan": asdict(plan),
                "tensor_sha256": [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in arrays],
                "input_shape": list(arrays[0].shape),
                "valid_fraction": float(arrays[2].mean()),
                "finite": all(bool(torch.isfinite(t).all()) for t in arrays),
                "range_0_1": bool((arrays[0] >= 0).all() and (arrays[0] <= 1).all()),
                "target_identical_across_frames": bool(torch.all(arrays[1] == arrays[1][0])),
            }
            if plan.source == "synthetic":
                snapshot["unique_input_frames"] = len({hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in arrays[0]})
            snapshots.append(snapshot)

        restarted = A12SamplingDataset(config, samples_per_epoch=args.samples_per_epoch)
        resumed_indices = list(A12StepSampler(restarted, start_step=3))
        checks["resume_starts_at_next_sample"] = resumed_indices == list(range(3, len(dataset)))
        checks["fresh_dataset_reproduces_samples"] = all(
            [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in restarted[snapshot["plan"]["sample_index"]]]
            == snapshot["tensor_sha256"] for snapshot in snapshots[:args.read_samples_per_epoch]
        )
        checks["sample_tensors_valid"] = all(s["finite"] and s["range_0_1"] and s["target_identical_across_frames"] for s in snapshots)
        checks["independent_synthetic_frames"] = all(s["unique_input_frames"] == 8 for s in snapshots if s["plan"]["source"] == "synthetic")
        coverage = {
            "selection": "first_absolute_samples" if args.read_first_samples is not None else "per_epoch_prefix",
            "samples": len(snapshots),
            "source_counts": dict(Counter(s["plan"]["source"] for s in snapshots)),
            "minimum_valid_fraction": min(s["valid_fraction"] for s in snapshots),
            "mean_valid_fraction": sum(s["valid_fraction"] for s in snapshots) / len(snapshots),
            "zero_valid_sample_indices": [s["plan"]["sample_index"] for s in snapshots if s["valid_fraction"] == 0],
            "below_1_percent_sample_indices": [s["plan"]["sample_index"] for s in snapshots if s["valid_fraction"] < 0.01],
            "below_6_25_percent_sample_indices": [s["plan"]["sample_index"] for s in snapshots if s["valid_fraction"] < 0.0625],
            "threshold_interpretation": "Diagnostic counts only; validity masks and sampling are not altered.",
        }
        checks["no_empty_supervision_in_read_samples"] = not coverage["zero_valid_sample_indices"]
        report.update({
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "epoch_summaries": epoch_summaries, "sample_snapshots": snapshots,
            "coverage_summary": coverage,
            "samples_read": len(snapshots), "sample_plans": str(plans_path),
            "baseline_source_sha256": {
                str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (REPO_ROOT / "src/raw_fusion/finetune/a3.py", REPO_ROOT / "src/raw_fusion/finetune/a4.py")
            },
        })
    except (OSError, ValueError, RuntimeError) as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})

    report["elapsed_seconds"] = time.monotonic() - started
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the standalone ST-Mamba RAW fusion model on aligned RAW streams."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from Mamba.mamba_2d_prior.losses import compute_pseudolabel_loss
from Mamba.mamba_2d_prior.model import Mamba2DPriorConfig, Mamba2DPriorFusionNet
from Mamba.mamba_2d_prior.raw_dataset import RandomPatchDataset, RawFusionStream, RawStreamConfig, split_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--two-d", required=True, type=Path)
    parser.add_argument("--three-d", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=1600)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--prior-cache",
        type=Path,
        default=None,
        help="Directory for the reusable full-frame prior memmap (default: <output-dir>/prior_cache)",
    )
    parser.add_argument("--rebuild-prior-cache", action="store_true", help="Force rebuilding the validated prior memmap")
    parser.add_argument("--disable-prior-cache", action="store_true", help="Compute priors per patch as in the original pipeline")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--guard-frames", type=int, default=10)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--mamba-state-dim", type=int, default=8)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-scan-backend", choices=("auto", "reference", "mamba_ssm"), default="auto")
    parser.add_argument("--mamba-variant", choices=("mamba1", "mamba2"), default="mamba1")
    parser.add_argument("--scan-path-mode", choices=("temporal4", "temporal4_grouped", "multiscale_grouped", "8path"), default="multiscale_grouped")
    parser.add_argument("--mamba2-state-dim", type=int, default=64)
    parser.add_argument("--mamba2-headdim", type=int, default=None)
    parser.add_argument("--mamba2-groups", type=int, default=1)
    parser.add_argument("--mamba2-chunk-size", type=int, default=256)
    parser.add_argument("--max-3dnr-weight", type=float, default=0.35)
    parser.add_argument("--beta-bias-init", type=float, default=-4.0)
    parser.add_argument("--disable-block-motion-gate", action="store_true")
    parser.add_argument("--disable-dynamic-direction-gate", action="store_true")
    parser.add_argument("--temporal-gate-bias-init", type=float, default=4.0)
    parser.add_argument("--source-black-level", type=float, default=252.0)
    parser.add_argument("--denoised-black-level", type=float, default=300.0)
    parser.add_argument("--source-container-scale", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("auto", "none", "bf16", "fp16"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if requested == "none" or device.type != "cuda":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_epoch(model: Mamba2DPriorFusionNet, loader: DataLoader[dict[str, torch.Tensor]], optimizer: torch.optim.Optimizer | None, device: torch.device, dtype: torch.dtype | None) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals: dict[str, float] = {}
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            output = model(batch["prev4"], batch["curr4"], batch["dnr2_4"], batch["dnr3_4"], batch["motion"], batch["flatness"], batch["agreement"])
            loss, parts = compute_pseudolabel_loss(output, batch)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / max(1, len(loader)) for key, value in totals.items()}


def make_loader(dataset: RandomPatchDataset, batch_size: int, shuffle: bool, device: torch.device, num_workers: int, prefetch_factor: int) -> DataLoader[dict[str, torch.Tensor]]:
    """Keep CPU patch construction ahead of the CUDA training stream."""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **options)  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must be in (0, 0.5)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = amp_dtype(args.amp, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    stream = RawFusionStream(
        RawStreamConfig(
            args.source,
            args.two_d,
            args.three_d,
            source_black_level=args.source_black_level,
            denoised_black_level=args.denoised_black_level,
            source_container_scale=args.source_container_scale,
            max_3dnr_weight=args.max_3dnr_weight,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_cache_dir = args.prior_cache or (args.output_dir / "prior_cache")
    if not args.disable_prior_cache:
        stream.enable_prior_cache(prior_cache_dir, rebuild=args.rebuild_prior_cache)
    split = split_sequence(stream.frame_count, args.validation_fraction, args.test_fraction, args.guard_frames)
    config = Mamba2DPriorConfig(
        channels=args.channels,
        num_blocks=args.num_blocks,
        mamba_state_dim=args.mamba_state_dim,
        mamba_expand=args.mamba_expand,
        mamba_scan_backend=args.mamba_scan_backend,
        mamba_variant=args.mamba_variant,
        scan_path_mode=args.scan_path_mode,
        mamba2_state_dim=args.mamba2_state_dim,
        mamba2_headdim=args.mamba2_headdim,
        mamba2_groups=args.mamba2_groups,
        mamba2_chunk_size=args.mamba2_chunk_size,
        max_3dnr_weight=args.max_3dnr_weight,
        beta_bias_init=args.beta_bias_init,
        block_motion_modulation=not args.disable_block_motion_gate,
        dynamic_direction_fusion=not args.disable_dynamic_direction_gate,
        temporal_gate_bias_init=args.temporal_gate_bias_init,
    )
    model = Mamba2DPriorFusionNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    run_config = vars(args) | {"device_resolved": str(device), "amp_dtype": None if dtype is None else str(dtype), "frame_count": stream.frame_count, "prior_cache_resolved": None if args.disable_prior_cache else str(prior_cache_dir.resolve()), "split": {"train": split.train_frames, "guard": split.guard_frames, "validation": split.validation_frames, "test": split.test_frames}, "model": config.serializable()}
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    best_validation = float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_set = RandomPatchDataset(stream, split.train_frames, args.patch_size, args.samples_per_epoch, args.seed + epoch * 100000)
        val_set = RandomPatchDataset(stream, split.validation_frames, args.patch_size, args.validation_samples, args.seed + epoch)
        train_stats = run_epoch(model, make_loader(train_set, args.batch_size, True, device, args.num_workers, args.prefetch_factor), optimizer, device, dtype)
        with torch.no_grad():
            validation_stats = run_epoch(model, make_loader(val_set, args.batch_size, False, device, args.num_workers, args.prefetch_factor), None, device, dtype)
        scheduler.step()
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_stats, "validation": validation_stats}
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {"model": model.state_dict(), "model_config": config.serializable(), "epoch": epoch, "validation_loss": validation_stats["loss"], "stream_config": {key: str(value) for key, value in vars(stream.config).items()}}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if validation_stats["loss"] < best_validation:
            best_validation = validation_stats["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_set = RandomPatchDataset(stream, split.test_frames, args.patch_size, args.test_samples, args.seed + 2_000_003)
    with torch.no_grad():
        test_stats = run_epoch(model, make_loader(test_set, args.batch_size, False, device, args.num_workers, args.prefetch_factor), None, device, dtype)
    test_report = {
        "checkpoint": "best.pt",
        "frames": split.test_frames,
        "metrics": test_stats,
        "note": "No clean ground truth is available; these are held-out pseudo-label and motion-safety metrics, not reference image-quality scores.",
    }
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_io import discover_dataset
from fusion_loss import WeakFusionLoss
from gatenet import GateNet, build_gate_features
from training_dataset import FusionTrainingDataset


FOLDS = {
    "128_to_645": (("128x",), ("645x",)),
    "645_to_128": (("645x",), ("128x",)),
    "all": (("128x", "645x"), ("128x", "645x")),
}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Weakly supervised 2DNR/3DNR convex-fusion training"
    )
    parser.add_argument("--dataset-root", type=Path, default=base / "DATASET")
    parser.add_argument("--md-root", type=Path, default=base / "DERIVED" / "md_mog2")
    parser.add_argument("--output", type=Path, default=base / "runs" / "weak_fusion")
    parser.add_argument("--fold", choices=tuple(FOLDS), default="128_to_645")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--black-source", type=float, default=252.0)
    parser.add_argument("--black-dnr", type=float, default=300.0)
    parser.add_argument("--md-dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if not 0.0 <= args.md_dropout < 1.0:
        parser.error("md-dropout must be in [0, 1)")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def average_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in totals.items()}


def run_epoch(
    *,
    model: GateNet,
    criterion: WeakFusionLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    md_dropout: float,
    amp_enabled: bool,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    progress = tqdm(loader, desc=description, leave=False)
    for batch in progress:
        batch = move_batch(batch, device)
        batch_size = batch["source"].shape[0]
        motion_feature = batch["motion"]
        if training and md_dropout > 0:
            keep = (
                torch.rand(batch_size, 1, 1, 1, device=device) >= md_dropout
            ).to(motion_feature.dtype)
            motion_feature = motion_feature * keep

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                features = build_gate_features(
                    batch["denoised"],
                    batch["fused"],
                    batch["source"],
                    batch["source_prev"],
                    batch["source_next"],
                    motion_feature,
                    batch["noise_sigma"],
                )
                alpha = model(features)
                loss, metrics = criterion(alpha, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {description}: {loss.item()}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        for key, value in metrics.items():
            totals[key] += float(value.item()) * batch_size
        sample_count += batch_size
        progress.set_postfix(
            loss=f"{metrics['total'].item():.4f}",
            alpha=f"{metrics['alpha_mean'].item():.3f}",
            static=f"{metrics['static_fraction'].item():.2f}",
            motion=f"{metrics['motion_fraction'].item():.2f}",
        )
    return average_metrics(totals, sample_count)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def append_history(path: Path, epoch: int, train: dict, val: dict, lr: float) -> None:
    row = {"epoch": epoch, "learning_rate": lr}
    row.update({f"train_{key}": value for key, value in train.items()})
    row.update({f"val_{key}": value for key, value in val.items()})
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = bool(args.amp and device.type == "cuda")
    train_ids, val_ids = FOLDS[args.fold]
    catalog = discover_dataset(args.dataset_root)
    train_dataset = FusionTrainingDataset(
        catalog,
        args.md_root,
        sequence_ids=train_ids,
        samples_per_epoch=args.train_samples,
        crop_size=args.crop_size,
        warmup_frames=args.warmup_frames,
        black_source=args.black_source,
        black_dnr=args.black_dnr,
        training=True,
        seed=args.seed,
    )
    val_dataset = FusionTrainingDataset(
        catalog,
        args.md_root,
        sequence_ids=val_ids,
        samples_per_epoch=args.val_samples,
        crop_size=args.crop_size,
        warmup_frames=args.warmup_frames,
        black_source=args.black_source,
        black_dnr=args.black_dnr,
        training=False,
        seed=args.seed + 100_000,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = GateNet().to(device)
    criterion = WeakFusionLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_val = float("inf")
    epochs_without_improvement = 0

    if not args.resume and any(
        (args.output / name).exists() for name in ("history.csv", "last.pt", "best.pt")
    ):
        raise FileExistsError(
            f"Training output already contains a run: {args.output}. "
            "Choose a new --output or use --resume."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    config.update(
        {
            "train_sequences": list(train_ids),
            "val_sequences": list(val_ids),
            "train_statistics": {
                key: value.as_dict() for key, value in train_dataset.statistics.items()
            },
            "val_statistics": {
                key: value.as_dict() for key, value in val_dataset.statistics.items()
            },
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))

    print(
        json.dumps(
            {
                "device": str(device),
                "amp": amp_enabled,
                "fold": args.fold,
                "train_sequences": train_ids,
                "val_sequences": val_ids,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "model_parameters": config["model_parameters"],
                "statistics": {
                    **config["train_statistics"],
                    **config["val_statistics"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            md_dropout=args.md_dropout,
            amp_enabled=amp_enabled,
            description=f"train {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            md_dropout=0.0,
            amp_enabled=amp_enabled,
            description=f"val {epoch}/{args.epochs}",
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        improved = val_metrics["total"] < best_val
        if improved:
            best_val = val_metrics["total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(args.output / "last.pt", state)
        append_history(
            args.output / "history.csv", epoch, train_metrics, val_metrics, current_lr
        )
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        print(
            f"epoch={epoch} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} "
            f"alpha(static/motion)={val_metrics['alpha_static']:.3f}/"
            f"{val_metrics['alpha_motion']:.3f} "
            f"proxy(output/d2/d3)={val_metrics['output_proxy']:.3f}/"
            f"{val_metrics['d2_proxy']:.3f}/{val_metrics['d3_proxy']:.3f} "
            f"{'best' if improved else ''}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

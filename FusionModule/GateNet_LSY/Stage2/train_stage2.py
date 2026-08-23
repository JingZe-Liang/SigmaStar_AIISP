from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


STAGE2_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STAGE2_ROOT.parents[2]
PHASE2_ROOT = WORKSPACE_ROOT / "Phase2"
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from dataset_io import discover_dataset  # noqa: E402
from fusion_loss import WeakFusionLoss  # noqa: E402
from training_dataset import FusionTrainingDataset  # noqa: E402

from fusion_loss_stage2 import Stage2FusionLoss
from gatenet_stage2 import GateNetStage2, build_gate_features


FOLDS = {
    "128_to_645": (("128x",), ("645x",)),
    "645_to_128": (("645x",), ("128x",)),
    "all": (("128x", "645x"), ("128x", "645x")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage2 MD-distilled 2DNR/3DNR fusion training"
    )
    parser.add_argument("--dataset-root", type=Path, default=PHASE2_ROOT / "DATASET")
    parser.add_argument(
        "--md-root", type=Path, default=PHASE2_ROOT / "DERIVED" / "md_mog2"
    )
    parser.add_argument("--output", type=Path, default=STAGE2_ROOT / "runs" / "stage2")
    parser.add_argument("--fold", choices=tuple(FOLDS), default="128_to_645")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--motion-loss-weight", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--black-source", type=float, default=252.0)
    parser.add_argument("--black-dnr", type=float, default=300.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if args.motion_loss_weight < 0:
        parser.error("motion-loss-weight must be non-negative")
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
    model: GateNetStage2,
    criterion: Stage2FusionLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
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
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                features = build_gate_features(
                    batch["denoised"],
                    batch["fused"],
                    batch["source"],
                    batch["source_prev"],
                    batch["source_next"],
                    batch["noise_sigma"],
                )
                alpha, motion_logit = model(features, return_motion=True)
                loss, metrics = criterion(alpha, motion_logit, batch)
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
            md_f1=(
                f"{2 * metrics['motion_precision'].item() * metrics['motion_recall'].item() / max(metrics['motion_precision'].item() + metrics['motion_recall'].item(), 1e-6):.2f}"
            ),
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
    dataset_options = {
        "catalog": catalog,
        "md_root": args.md_root,
        "crop_size": args.crop_size,
        "warmup_frames": args.warmup_frames,
        "black_source": args.black_source,
        "black_dnr": args.black_dnr,
    }
    train_dataset = FusionTrainingDataset(
        sequence_ids=train_ids,
        samples_per_epoch=args.train_samples,
        training=True,
        seed=args.seed,
        **dataset_options,
    )
    val_dataset = FusionTrainingDataset(
        sequence_ids=val_ids,
        samples_per_epoch=args.val_samples,
        training=False,
        seed=args.seed + 100_000,
        **dataset_options,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = GateNetStage2(base_channels=args.base_channels).to(device)
    criterion = Stage2FusionLoss(
        WeakFusionLoss(), motion_weight=args.motion_loss_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
            "stage": 2,
            "inference_inputs": [
                "2DNR",
                "3DNR",
                "noisy_t",
                "noisy_t-1",
                "noisy_t+1",
            ],
            "md_usage": "training supervision only",
            "train_sequences": list(train_ids),
            "val_sequences": list(val_ids),
            "train_statistics": {
                key: value.as_dict() for key, value in train_dataset.statistics.items()
            },
            "val_statistics": {
                key: value.as_dict() for key, value in val_dataset.statistics.items()
            },
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "model_config": model.model_config(),
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
                "model_parameters": config["model_parameters"],
                "md_usage": config["md_usage"],
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
            "model_config": model.model_config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(args.output / "last.pt", state)
        append_history(args.output / "history.csv", epoch, train_metrics, val_metrics, current_lr)
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        print(
            f"epoch={epoch} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} "
            f"alpha(static/motion)={val_metrics['alpha_static']:.3f}/"
            f"{val_metrics['alpha_motion']:.3f} "
            f"motion(P/R)={val_metrics['motion_precision']:.3f}/"
            f"{val_metrics['motion_recall']:.3f} "
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

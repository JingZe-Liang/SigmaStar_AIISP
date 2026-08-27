from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

STAGE3_ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = STAGE3_ROOT.parent / "Stage2"
WORKSPACE_ROOT = STAGE3_ROOT.parents[2]
DATA_ROOT = STAGE2_ROOT / "data"
if str(STAGE3_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE3_ROOT))
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

from dataset_io import discover_dataset  # noqa: E402
from weak_fusion_loss import WeakFusionLoss  # noqa: E402
from training_dataset import FusionTrainingDataset  # noqa: E402

from fusion_loss_stage2 import Stage2FusionLoss  # noqa: E402
from fusion_loss_stage3 import Stage3FusionLoss  # noqa: E402
from gatenet_stage2 import GateNetStage2  # noqa: E402
from motionnet_stage3 import MotionFocalLoss, TemporalMotionNet, build_motion_features  # noqa: E402
from train_stage2 import (  # noqa: E402
    FOLDS,
    move_batch,
    run_epoch,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage3 staged training: fusion-only, then motion-head-only"
    )
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "DATASET")
    parser.add_argument("--md-root", type=Path, default=DATA_ROOT / "md_mog2")
    parser.add_argument("--output", type=Path, default=STAGE3_ROOT / "runs" / "stage3")
    parser.add_argument("--fold", choices=tuple(FOLDS), default="128_to_645")
    parser.add_argument("--phase1-epochs", type=int, default=50)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--motion-learning-rate", type=float, default=2e-4)
    parser.add_argument("--static-alpha-weight", type=float, default=0.1)
    parser.add_argument("--static-d2-weight", type=float, default=0.0)
    parser.add_argument("--motion-positive-weight", type=float, default=4.0)
    parser.add_argument("--motion-focal-gamma", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--black-source", type=float, default=252.0)
    parser.add_argument("--black-dnr", type=float, default=300.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.phase1_epochs <= 0 or args.phase2_epochs <= 0 or args.batch_size <= 0:
        parser.error("phase epochs and batch-size must be positive")
    if args.static_alpha_weight < 0 or args.static_d2_weight < 0:
        parser.error("static loss weights must be non-negative")
    return args


def make_loaders(args: argparse.Namespace, device: torch.device):
    train_ids, val_ids = FOLDS[args.fold]
    catalog = discover_dataset(args.dataset_root)
    options = {
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
        **options,
    )
    val_dataset = FusionTrainingDataset(
        sequence_ids=val_ids,
        samples_per_epoch=args.val_samples,
        training=False,
        seed=args.seed + 100_000,
        **options,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    return (
        train_dataset,
        val_dataset,
        DataLoader(train_dataset, shuffle=False, **loader_options),
        DataLoader(val_dataset, shuffle=False, **loader_options),
        train_ids,
        val_ids,
    )


def write_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_motion_epoch(*, model, criterion, loader, device, optimizer, scaler, amp_enabled, description):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        batch_size = batch["source"].shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(build_motion_features(
                    batch["source"], batch["source_prev"], batch["source_next"], batch["noise_sigma"]
                ))
                loss, metrics = criterion(logits, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite motion loss in {description}: {loss.item()}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = bool(args.amp and device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("config.json", "phase1_best.pt")):
        raise FileExistsError(f"Output already contains a Stage3 run: {args.output}")

    train_dataset, val_dataset, train_loader, val_loader, train_ids, val_ids = make_loaders(args, device)
    model = GateNetStage2(base_channels=args.base_channels).to(device)
    criterion = Stage3FusionLoss(
        Stage2FusionLoss(WeakFusionLoss(), motion_weight=0.0),
        static_alpha_weight=args.static_alpha_weight,
        static_d2_weight=args.static_d2_weight,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_rows: list[dict[str, float | int | str]] = []
    run_config = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    run_config.update({
        "stage": 3,
        "strategy": "fusion_only_then_independent_temporal_motion",
        "inference_inputs": ["2DNR", "3DNR", "noisy_t", "noisy_t-1", "noisy_t+1"],
        "md_usage": "training supervision only",
        "train_sequences": list(train_ids),
        "val_sequences": list(val_ids),
        "train_statistics": {key: value.as_dict() for key, value in train_dataset.statistics.items()},
        "val_statistics": {key: value.as_dict() for key, value in val_dataset.statistics.items()},
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_config": model.model_config(),
        "static_alpha_weight": args.static_alpha_weight,
        "static_d2_weight": args.static_d2_weight,
    })

    # Phase 1: optimize only the fusion objective. The motion head receives no gradient.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)
    best_fusion = float("inf")
    for epoch in range(1, args.phase1_epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model=model, criterion=criterion, loader=train_loader, device=device,
            optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled,
            description=f"stage3 phase1 train {epoch}/{args.phase1_epochs}",
        )
        val_metrics = run_epoch(
            model=model, criterion=criterion, loader=val_loader, device=device,
            optimizer=None, scaler=scaler, amp_enabled=amp_enabled,
            description=f"stage3 phase1 val {epoch}/{args.phase1_epochs}",
        )
        scheduler.step()
        row = {"phase": "fusion", "epoch": epoch, "train_fusion_total": train_metrics["fusion_total"],
               "val_fusion_total": val_metrics["fusion_total"], "val_output_proxy": val_metrics["output_proxy"],
               "val_d2_proxy": val_metrics["d2_proxy"], "val_d3_proxy": val_metrics["d3_proxy"]}
        metrics_rows.append(row)
        state = {"stage3_phase": 1, "epoch": epoch, "model": model.state_dict(),
                 "model_config": model.model_config(), "config": run_config, "val_metrics": val_metrics}
        save_checkpoint(args.output / "phase1_last.pt", state)
        if val_metrics["fusion_total"] < best_fusion:
            best_fusion = val_metrics["fusion_total"]
            save_checkpoint(args.output / "phase1_best.pt", state)

    # Phase 2: independent temporal branch cannot modify the frozen fusion model.
    checkpoint = torch.load(args.output / "phase1_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    motion_model = TemporalMotionNet(base_channels=args.base_channels).to(device)
    motion_criterion = MotionFocalLoss(
        positive_weight=args.motion_positive_weight, gamma=args.motion_focal_gamma
    ).to(device)
    optimizer = torch.optim.AdamW(
        motion_model.parameters(),
        lr=args.motion_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
    best_motion = float("inf")
    for epoch in range(1, args.phase2_epochs + 1):
        train_dataset.set_epoch(args.phase1_epochs + epoch)
        train_metrics = run_motion_epoch(
            model=motion_model, criterion=motion_criterion, loader=train_loader, device=device,
            optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled,
            description=f"stage3 phase2 train {epoch}/{args.phase2_epochs}",
        )
        val_metrics = run_motion_epoch(
            model=motion_model, criterion=motion_criterion, loader=val_loader, device=device,
            optimizer=None, scaler=scaler, amp_enabled=amp_enabled,
            description=f"stage3 phase2 val {epoch}/{args.phase2_epochs}",
        )
        scheduler.step()
        row = {"phase": "motion", "epoch": epoch, "train_motion_loss": train_metrics["motion_loss"],
               "val_motion_loss": val_metrics["motion_loss"], "val_motion_precision": val_metrics["motion_precision"],
               "val_motion_recall": val_metrics["motion_recall"]}
        metrics_rows.append(row)
        state = {"stage3_phase": 2, "epoch": epoch, "model": model.state_dict(),
                 "model_config": model.model_config(), "motion_model": motion_model.state_dict(),
                 "motion_model_config": motion_model.model_config(), "config": run_config,
                 "val_metrics": val_metrics, "frozen_fusion": True}
        save_checkpoint(args.output / "phase2_last.pt", state)
        if val_metrics["motion_loss"] < best_motion:
            best_motion = val_metrics["motion_loss"]
            save_checkpoint(args.output / "phase2_best.pt", state)

    run_config["motion_inference_dependency"] = False
    run_config["phase1_motion_loss_weight"] = 0.0
    (args.output / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics(args.output / "history.csv", metrics_rows)
    print(json.dumps({"output": str(args.output.resolve()), "phase1_best_fusion": best_fusion,
                      "phase2_best_motion": best_motion, "frozen_fusion": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

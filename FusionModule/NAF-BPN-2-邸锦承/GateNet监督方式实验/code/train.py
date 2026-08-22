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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    FRAME_COUNT,
    CleanH5Dataset,
    WeakFusionDataset,
    discover_sequences,
)
from losses import NAFBPNWeakFusionLoss, WeakLossWeights, stage1_supervised_loss
from model import NAFBPNMotionFusionNet, extract_model_state


ROOT = Path(__file__).resolve().parent
FOLDS = {
    "128_to_645": (("128x",), ("645x",)),
    "645_to_128": (("645x",), ("128x",)),
    "all": (("128x", "645x"), ("128x", "645x")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAF-BPN Stage 1/2 training")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--stage", choices=("1", "2"), default="2")
    parser.add_argument("--fold", choices=tuple(FOLDS), default="128_to_645")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"device", "seed", "batch_size", "patch_size", "num_workers"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config 缺少字段: {sorted(missing)}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def load_checkpoint(model: NAFBPNMotionFusionNet, path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_model_state(payload)
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {"model": state}


def make_loader(dataset, config: dict[str, Any], device: torch.device, shuffle: bool) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def run_stage1_epoch(
    model: NAFBPNMotionFusionNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in tqdm(loader, desc=description, leave=False):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp_enabled):
            prediction = model(
                batch["image_2dnr"],
                batch["image_3dnr"],
                batch["noisy_current"],
                batch["noisy_previous"],
            )
            loss, metrics = stage1_supervised_loss(prediction, batch["target"])
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        batch_size = batch["target"].shape[0]
        for key, value in metrics.items():
            totals[key] += float(value.item()) * batch_size
        count += batch_size
    return average_metrics(totals, count)


def run_stage2_epoch(
    model: NAFBPNMotionFusionNet,
    criterion: NAFBPNWeakFusionLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in tqdm(loader, desc=description, leave=False):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp_enabled):
            prediction = model(
                batch["image_2dnr"],
                batch["image_3dnr"],
                batch["noisy_current"],
                batch["noisy_previous"],
            )
            loss, metrics = criterion(prediction, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"非有限 loss: {loss.item()}")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        batch_size = batch["image_2dnr"].shape[0]
        for key, value in metrics.items():
            totals[key] += float(value.item()) * batch_size
        count += batch_size
    return average_metrics(totals, count)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_model(config: dict[str, Any], device: torch.device) -> NAFBPNMotionFusionNet:
    model = NAFBPNMotionFusionNet(
        num_basis=int(config.get("num_basis", 15)),
        kernel_size=int(config.get("kernel_size", 7)),
        width=int(config.get("model_width", 32)),
    ).to(device)
    return model


def run_stage1(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> int:
    data_root = config.get("stage1_data_root", config.get("data_root"))
    if not data_root:
        raise ValueError("Stage 1 需要 stage1_data_root")
    model = build_model(config, device)
    output = args.output or Path(config.get("stage1_output", ROOT / "runs" / "stage1"))
    train_data = CleanH5Dataset(data_root, int(config["patch_size"]), training=True)
    val_root = config.get("stage1_validation_root")
    val_data = CleanH5Dataset(val_root, int(config["patch_size"]), training=False) if val_root else None
    train_loader = make_loader(train_data, config, device, shuffle=True)
    val_loader = make_loader(val_data, config, device, shuffle=False) if val_data else None
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("stage1_learning_rate", config.get("learning_rate", 1e-3))),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    epochs = 1 if args.smoke_test else int(config.get("stage1_epochs", config.get("epochs", 800)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-7)
    scaler = make_scaler(device, bool(config.get("amp", True)))
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(extract_model_state(payload), strict=True)
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = run_stage1_epoch(model, train_loader, device, optimizer, scaler, scaler.is_enabled(), f"stage1 train {epoch}/{epochs}")
        val_metrics = run_stage1_epoch(model, val_loader, device, None, scaler, scaler.is_enabled(), f"stage1 val {epoch}/{epochs}") if val_loader else {}
        scheduler.step()
        score = val_metrics.get("total", train_metrics["total"])
        row = {"stage": 1, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        append_history(output / "history.csv", row)
        state = {"variant": "naf_bpn_stage1", "stage": 1, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": config, "metrics": row}
        save_checkpoint(output / "last.pth", state)
        if score < best:
            best = score
            save_checkpoint(output / "best.pth", state)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


def run_stage2(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> int:
    data_root = config.get("data_root")
    motion_root = config.get("motion_cache_root")
    if not data_root or not motion_root:
        raise ValueError("Stage 2 需要 data_root 和训练用 motion_cache_root")
    train_names, validation_names = FOLDS[args.fold]
    names = tuple(config.get("sequence_names", ("128x", "645x")))
    sequences = discover_sequences(
        Path(data_root),
        names,
        Path(motion_root),
        str(config.get("cfa_pattern", "RGGB")),
        int(config.get("source_black_level", 252)),
        int(config.get("dnr_black_level", 300)),
        int(config.get("white_level", 4095)),
    )
    train_sequences = tuple(sequence for sequence in sequences if sequence.name in train_names)
    val_sequences = tuple(sequence for sequence in sequences if sequence.name in validation_names)
    proxy_radius = int(config.get("proxy_radius", 3))
    split = int(config.get("train_frame_end", 160))
    train_frames = tuple(range(proxy_radius, split))
    val_frames = tuple(range(split, FRAME_COUNT - proxy_radius))
    steps_per_epoch = 1 if args.smoke_test else int(config.get("steps_per_epoch", 1000))
    validation_batches = 1 if args.smoke_test else int(config.get("validation_batches", 16))
    train_data = WeakFusionDataset(
        train_sequences,
        train_frames,
        steps_per_epoch * int(config["batch_size"]),
        int(config["patch_size"]),
        int(config["seed"]),
        proxy_radius=proxy_radius,
        training=True,
        motion_dropout=float(config.get("motion_dropout", 0.2)),
        motion_jitter=bool(config.get("motion_jitter", True)),
    )
    val_data = WeakFusionDataset(
        val_sequences,
        val_frames,
        validation_batches * int(config["batch_size"]),
        int(config["patch_size"]),
        int(config["seed"]) + 100000,
        proxy_radius=proxy_radius,
        training=False,
        motion_dropout=0.0,
        motion_jitter=False,
    )
    train_loader = make_loader(train_data, config, device, shuffle=False)
    val_loader = make_loader(val_data, config, device, shuffle=False)
    model = build_model(config, device)
    checkpoint_path = args.init_checkpoint or (Path(config["init_checkpoint"]) if config.get("init_checkpoint") else None)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, device)
        print(f"已严格加载 Stage 1 checkpoint: {checkpoint_path}")
    else:
        print("警告：Stage 2 未提供 init_checkpoint，将从随机权重开始", flush=True)
    weights = WeakLossWeights(
        proxy_static=float(config.get("lambda_proxy_static", 1.0)),
        gradient_static=float(config.get("lambda_gradient_static", 0.25)),
        motion_anchor=float(config.get("lambda_motion_anchor", 0.10)),
        candidate_stability=float(config.get("lambda_candidate_stability", 0.10)),
        masked_noisy=float(config.get("lambda_masked_noisy", 0.05)),
    )
    criterion = NAFBPNWeakFusionLoss(
        weights,
        static_temporal_threshold=float(config.get("static_temporal_threshold", 0.015)),
        static_range_threshold=float(config.get("static_range_threshold", 0.035)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("stage2_learning_rate", config.get("learning_rate", 2e-4))),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = 1 if args.smoke_test else int(config.get("stage2_epochs", config.get("epochs", 50)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-7)
    scaler = make_scaler(device, bool(config.get("amp", True)))
    output = args.output or ROOT / "runs" / f"stage2_{args.fold}"
    start_epoch = 1
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(extract_model_state(payload), strict=True)
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if isinstance(payload, dict) and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
    config_used = {
        **config,
        "stage": 2,
        "fold": args.fold,
        "train_sequences": list(train_names),
        "validation_sequences": list(validation_names),
        "weak_supervision": True,
        "strict_j_invariant": False,
        "train_statistics": {key: value.as_dict() for key, value in train_data.statistics.items()},
        "validation_statistics": {key: value.as_dict() for key, value in val_data.statistics.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(json.dumps(config_used, ensure_ascii=False, indent=2), encoding="utf-8")
    best = float("inf")
    for epoch in range(start_epoch, epochs + 1):
        train_data.set_epoch(epoch)
        train_metrics = run_stage2_epoch(model, criterion, train_loader, device, optimizer, scaler, scaler.is_enabled(), f"stage2 train {epoch}/{epochs}")
        val_metrics = run_stage2_epoch(model, criterion, val_loader, device, None, scaler, scaler.is_enabled(), f"stage2 val {epoch}/{epochs}")
        scheduler.step()
        row = {"stage": 2, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        append_history(output / "history.csv", row)
        state = {"variant": "naf_bpn_weak_stage2", "stage": 2, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": config_used, "metrics": row}
        save_checkpoint(output / "last.pth", state)
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            save_checkpoint(output / "best.pth", state)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 要求 CUDA，但当前 PyTorch 未检测到 CUDA")
    return run_stage1(args, config, device) if args.stage == "1" else run_stage2(args, config, device)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import FRAME_COUNT, CleanH5Dataset, WeakFusionDataset, discover_sequences
from losses import NAFBPNWeakFusionLoss, WeakLossWeights, stage1_supervised_loss
from model import NAFBPNMotionFusionNet, extract_model_state


ROOT = Path(__file__).resolve().parent
FOLDS = {
    "128_to_645": (("128x",), ("645x",)),
    "645_to_128": (("645x",), ("128x",)),
    "all": (("128x", "645x"), ()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAF-BPN cloud Stage 1/2 training")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cloud.json")
    parser.add_argument("--stage", choices=("1", "2"), required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), default="128_to_645")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_config(path: Path) -> dict[str, Any]:
    config_path = path if path.is_absolute() else ROOT / path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"device", "seed", "batch_size", "patch_size", "num_workers"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config 缺少字段: {sorted(missing)}")
    return config


def config_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"config 缺少路径字段: {key}")
    return resolve_path(str(value))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_acceleration(config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
    allow_tf32 = bool(config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def initialize_worker(_: int) -> None:
    """Avoid nested CPU thread pools when twelve loader workers read RAW crops."""
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def load_checkpoint(model: NAFBPNMotionFusionNet, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(extract_model_state(payload), strict=True)
    return payload if isinstance(payload, dict) else {"model": extract_model_state(payload)}


def restore_training_state(
    model: NAFBPNMotionFusionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    path: Path,
) -> tuple[int, float]:
    payload = load_checkpoint(model, path)
    if "optimizer" not in payload or "scheduler" not in payload:
        raise ValueError(f"断点缺少 optimizer/scheduler，不能恢复训练: {path}")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("epoch", 0)) + 1, float(payload.get("best_score", float("inf")))


def make_loader(dataset, config: dict[str, Any], device: torch.device, shuffle: bool) -> DataLoader:
    workers = int(config["num_workers"])
    options: dict[str, Any] = {
        "batch_size": int(config["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "drop_last": shuffle,
        "worker_init_fn": initialize_worker if workers > 0 else None,
    }
    if workers > 0:
        options["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(dataset, **options)


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
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(tqdm(loader, desc=description, leave=False), start=1):
        if max_batches is not None and batch_index > max_batches:
            break
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
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Stage 1 非有限 loss: {loss.item()}")
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
    if count == 0:
        raise RuntimeError("Stage 1 epoch 没有处理任何 batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = average_metrics(totals, count)
    elapsed = max(time.perf_counter() - started_at, 1e-6)
    metrics["samples_per_second"] = count / elapsed
    if device.type == "cuda":
        metrics["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return metrics


def run_stage2_epoch(
    model: NAFBPNMotionFusionNet,
    criterion: NAFBPNWeakFusionLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
    description: str,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(tqdm(loader, desc=description, leave=False), start=1):
        if max_batches is not None and batch_index > max_batches:
            break
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
            raise FloatingPointError(f"Stage 2 非有限 loss: {loss.item()}")
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
    if count == 0:
        raise RuntimeError("Stage 2 epoch 没有处理任何 batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = average_metrics(totals, count)
    elapsed = max(time.perf_counter() - started_at, 1e-6)
    metrics["samples_per_second"] = count / elapsed
    if device.type == "cuda":
        metrics["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return metrics


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model(config: dict[str, Any], device: torch.device) -> NAFBPNMotionFusionNet:
    return NAFBPNMotionFusionNet(
        num_basis=int(config.get("num_basis", 15)),
        kernel_size=int(config.get("kernel_size", 7)),
        width=int(config.get("model_width", 32)),
    ).to(device)


def checkpoint_state(
    variant: str,
    stage: int,
    epoch: int,
    model: NAFBPNMotionFusionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict[str, Any],
    metrics: dict[str, Any],
    best_score: float,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "metrics": metrics,
        "best_score": best_score,
    }


def effective_max_batches(args: argparse.Namespace) -> int | None:
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches 必须大于 0")
    return args.max_batches if args.max_batches is not None else (1 if args.smoke_test else None)


def run_stage1(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> int:
    data_root = config_path(config, "stage1_data_root")
    output = resolve_path(args.output or config.get("stage1_output", "runs/stage1"))
    train_data = CleanH5Dataset(data_root, int(config["patch_size"]), split="train", seed=int(config["seed"]))
    validation_data = CleanH5Dataset(data_root, int(config["patch_size"]), split="validation", seed=int(config["seed"]))
    train_loader = make_loader(train_data, config, device, shuffle=True)
    validation_loader = make_loader(validation_data, config, device, shuffle=False)
    model = build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=float(config.get("stage1_learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    epochs = 1 if args.smoke_test else int(config.get("stage1_epochs", 800))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-7)
    scaler = make_scaler(device, bool(config.get("amp", True)))
    start_epoch, best_score = 1, float("inf")
    if args.resume:
        start_epoch, best_score = restore_training_state(model, optimizer, scheduler, scaler, resolve_path(args.resume))
        print(f"恢复 Stage 1: epoch={start_epoch}, checkpoint={resolve_path(args.resume)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    config_used = {
        **config,
        "stage": 1,
        "stage1_validation_policy": "last_h5_per_scene",
        "train_h5_files": [str(path.relative_to(data_root)) for path in train_data.h5_files],
        "validation_h5_files": [str(path.relative_to(data_root)) for path in validation_data.h5_files],
    }
    write_json(output / "config_used.json", config_used)
    max_batches = effective_max_batches(args)
    for epoch in range(start_epoch, epochs + 1):
        train_data.set_epoch(epoch)
        train_metrics = run_stage1_epoch(model, train_loader, device, optimizer, scaler, scaler.is_enabled(), f"stage1 train {epoch}/{epochs}", max_batches)
        validation_metrics = run_stage1_epoch(model, validation_loader, device, None, scaler, scaler.is_enabled(), f"stage1 validation {epoch}/{epochs}", max_batches)
        scheduler.step()
        score = validation_metrics["total"]
        row = {"stage": 1, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        append_history(output / "history.csv", row)
        state = checkpoint_state("naf_bpn_stage1", 1, epoch, model, optimizer, scheduler, scaler, config_used, row, min(best_score, score))
        save_checkpoint(output / "last.pth", state)
        if score < best_score:
            best_score = score
            state["best_score"] = best_score
            save_checkpoint(output / "best.pth", state)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


def run_stage2(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> int:
    data_root = config_path(config, "data_root")
    motion_root = config_path(config, "motion_cache_root")
    sequences = discover_sequences(
        data_root,
        tuple(config.get("sequence_names", ("128x", "645x"))),
        motion_root,
        str(config["cfa_pattern"]),
        int(config["source_black_level"]),
        int(config["dnr_black_level"]),
        int(config["white_level"]),
    )
    radius = int(config["proxy_radius"])
    final_fit = args.fold == "all"
    train_names, validation_names = FOLDS[args.fold]
    train_sequences = tuple(item for item in sequences if item.name in train_names)
    validation_sequences = tuple(item for item in sequences if item.name in validation_names)
    train_frames = tuple(range(radius, FRAME_COUNT - radius)) if final_fit else tuple(range(radius, int(config["train_frame_end"])))
    validation_frames = () if final_fit else tuple(range(int(config["train_frame_end"]), FRAME_COUNT - radius))
    return run_stage2_job(args, config, device, train_sequences, validation_sequences, train_frames, validation_frames, final_fit=final_fit)


def run_stage2_job(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    train_sequences,
    validation_sequences,
    train_frames: tuple[int, ...],
    validation_frames: tuple[int, ...],
    *,
    final_fit: bool,
) -> int:
    if not train_sequences:
        raise ValueError("Stage 2 没有训练序列")
    batch_size = int(config["batch_size"])
    steps_per_epoch = 1 if args.smoke_test else int(config["steps_per_epoch"])
    validation_batches = 1 if args.smoke_test else int(config["validation_batches"])
    train_data = WeakFusionDataset(train_sequences, train_frames, steps_per_epoch * batch_size, int(config["patch_size"]), int(config["seed"]), proxy_radius=int(config["proxy_radius"]), training=True, motion_dropout=float(config.get("motion_dropout", 0.2)), motion_jitter=bool(config.get("motion_jitter", True)))
    validation_data = None
    if not final_fit:
        validation_data = WeakFusionDataset(validation_sequences, validation_frames, validation_batches * batch_size, int(config["patch_size"]), int(config["seed"]) + 100000, proxy_radius=int(config["proxy_radius"]), training=False, motion_dropout=0.0, motion_jitter=False)
    train_loader = make_loader(train_data, config, device, shuffle=False)
    validation_loader = make_loader(validation_data, config, device, shuffle=False) if validation_data else None
    model = build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=float(config.get("stage2_learning_rate", 2e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    epochs = 1 if args.smoke_test else int(config.get("stage2_epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-7)
    scaler = make_scaler(device, bool(config.get("amp", True)))
    output = resolve_path(args.output or f"runs/stage2_{'final_all' if final_fit else args.fold}")
    start_epoch, best_score = 1, float("inf")
    if args.resume:
        start_epoch, best_score = restore_training_state(model, optimizer, scheduler, scaler, resolve_path(args.resume))
        print(f"恢复 Stage 2: epoch={start_epoch}, checkpoint={resolve_path(args.resume)}", flush=True)
    else:
        if args.init_checkpoint is None:
            raise ValueError("Stage 2 必须提供 --init-checkpoint")
        checkpoint = resolve_path(args.init_checkpoint)
        load_checkpoint(model, checkpoint)
        print(f"已严格加载 Stage 1 checkpoint: {checkpoint}", flush=True)
    weights = WeakLossWeights(proxy_static=float(config.get("lambda_proxy_static", 1.0)), gradient_static=float(config.get("lambda_gradient_static", 0.25)), motion_anchor=float(config.get("lambda_motion_anchor", 0.10)), candidate_stability=float(config.get("lambda_candidate_stability", 0.10)), masked_noisy=float(config.get("lambda_masked_noisy", 0.05)))
    criterion = NAFBPNWeakFusionLoss(weights, static_temporal_threshold=float(config.get("static_temporal_threshold", 0.015)), static_range_threshold=float(config.get("static_range_threshold", 0.035))).to(device)
    config_used = {
        **config,
        "stage": 2,
        "fold": args.fold,
        "final_fit": final_fit,
        "train_sequences": [item.name for item in train_sequences],
        "validation_sequences": [item.name for item in validation_sequences],
        "train_frames": [min(train_frames), max(train_frames)],
        "validation_frames": [min(validation_frames), max(validation_frames)] if validation_frames else [],
        "weak_supervision": True,
        "strict_j_invariant": False,
        "train_statistics": {key: value.as_dict() for key, value in train_data.statistics.items()},
        "validation_statistics": {} if validation_data is None else {key: value.as_dict() for key, value in validation_data.statistics.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_used.json", config_used)
    max_batches = effective_max_batches(args)
    last_state: dict[str, Any] | None = None
    for epoch in range(start_epoch, epochs + 1):
        train_data.set_epoch(epoch)
        train_metrics = run_stage2_epoch(model, criterion, train_loader, device, optimizer, scaler, scaler.is_enabled(), f"stage2 train {args.fold} {epoch}/{epochs}", max_batches)
        validation_metrics = {}
        if validation_loader is not None:
            validation_metrics = run_stage2_epoch(model, criterion, validation_loader, device, None, scaler, scaler.is_enabled(), f"stage2 validation {args.fold} {epoch}/{epochs}", max_batches)
        scheduler.step()
        score = train_metrics["total"] if final_fit else validation_metrics["total"]
        row = {"stage": 2, "fold": args.fold, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        append_history(output / "history.csv", row)
        last_state = checkpoint_state("naf_bpn_weak_stage2", 2, epoch, model, optimizer, scheduler, scaler, config_used, row, min(best_score, score))
        save_checkpoint(output / "last.pth", last_state)
        if not final_fit and score < best_score:
            best_score = score
            last_state["best_score"] = best_score
            save_checkpoint(output / "best.pth", last_state)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if final_fit:
        if last_state is None:
            last_state = load_checkpoint(model, output / "last.pth")
        save_checkpoint(output / "final.pth", last_state)
    return 0


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 要求 CUDA，但当前 PyTorch 未检测到 CUDA")
    configure_acceleration(config, device)
    return run_stage1(args, config, device) if args.stage == "1" else run_stage2(args, config, device)


if __name__ == "__main__":
    raise SystemExit(main())

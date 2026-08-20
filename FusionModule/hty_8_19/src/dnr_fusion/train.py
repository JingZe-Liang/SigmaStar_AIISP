from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, project_root, validate_scene
from .dataset_fast import FusionPatchDataset
from .model import SafeGateUNet


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gate_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    hard_mask: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    gate = torch.sigmoid(logits)
    risk_weight = float(settings["risk_weight"])
    pixel_weight = 1.0 + risk_weight * (1.0 - target)
    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        * pixel_weight
    ).mean()
    l1 = (torch.abs(gate - target) * pixel_weight).mean()
    conservative = (gate * hard_mask).sum() / hard_mask.sum().clamp_min(1.0)
    tv = torch.mean(torch.abs(gate[:, :, :, 1:] - gate[:, :, :, :-1]))
    tv = tv + torch.mean(torch.abs(gate[:, :, 1:, :] - gate[:, :, :-1, :]))
    total = (
        bce
        + float(settings["l1_weight"]) * l1
        + float(settings["conservative_weight"]) * conservative
        + float(settings["tv_weight"]) * tv
    )
    return total, {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "l1": float(l1.detach()),
        "hard_gate": float(conservative.detach()),
        "tv": float(tv.detach()),
        "gate_mean": float(gate.detach().mean()),
        "target_mean": float(target.detach().mean()),
    }


def merge_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {
        key: float(np.mean([item[key] for item in items])) for key in items[0]
    }


@torch.inference_mode()
def validate(
    model: SafeGateUNet,
    loader: DataLoader,
    device: torch.device,
    settings: dict[str, Any],
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        target = batch["target_gate"].to(device, non_blocking=True)
        hard_mask = batch["hard_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(features)
            _, batch_metrics = gate_loss(logits, target, hard_mask, settings)
        gate = torch.sigmoid(logits.float())
        static = target >= 0.8
        motion = hard_mask > 0
        batch_metrics["gate_mae"] = float(torch.mean(torch.abs(gate - target)))
        batch_metrics["static_gate"] = float(
            gate[static].mean() if static.any() else torch.tensor(0.0, device=device)
        )
        batch_metrics["motion_gate"] = float(
            gate[motion].mean() if motion.any() else torch.tensor(0.0, device=device)
        )
        metrics.append(batch_metrics)
    return merge_metrics(metrics)


def save_checkpoint(
    path: Path,
    model: SafeGateUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fold_name: str,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "fold": fold_name,
        "epoch": epoch,
        "model": {
            "input_channels": model.input_channels,
            "width": model.width,
            "state_dict": model.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "raw": config["raw"],
        "safety": config["safety"],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_fold(
    config: dict[str, Any],
    fold_name: str,
    output_dir: Path,
    device: torch.device,
) -> Path:
    fold = config["folds"][fold_name]
    train_scene = str(fold["train_scene"])
    validate_scene(config, train_scene)
    training = config["training"]
    seed = int(training["seed"]) + ord(fold_name[0])
    seed_everything(seed)

    batch_size = int(training["batch_size"])
    train_dataset = FusionPatchDataset(
        config,
        train_scene,
        fold["train_frames"],
        patch_size=int(training["patch_size"]),
        samples=int(training["steps_per_epoch"]) * batch_size,
        seed=seed,
        augment=True,
    )
    val_dataset = FusionPatchDataset(
        config,
        train_scene,
        fold["val_frames"],
        patch_size=int(training["patch_size"]),
        samples=int(training["val_patches"]),
        seed=seed + 10_000,
        augment=False,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_settings = {
        "batch_size": batch_size,
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_settings
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_settings)

    model = SafeGateUNet(
        input_channels=int(config["model"]["input_channels"]),
        width=int(config["model"]["width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(training["learning_rate"]) * 0.05
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    history: list[dict[str, Any]] = []
    best_score = math.inf
    started = time.perf_counter()

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_metrics: list[dict[str, float]] = []
        progress = tqdm(
            train_loader,
            desc=f"fold {fold_name} epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        for batch in progress:
            features = batch["features"].to(device, non_blocking=True)
            target = batch["target_gate"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(features)
                loss, batch_metrics = gate_loss(
                    logits, target, hard_mask, training
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_metrics.append(batch_metrics)
            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                gate=f"{batch_metrics['gate_mean']:.3f}",
            )

        train_metrics = merge_metrics(epoch_metrics)
        val_metrics = validate(
            model, val_loader, device, training, amp_enabled=amp_enabled
        )
        scheduler.step()
        score = val_metrics["gate_mae"] + 2.0 * val_metrics["motion_gate"]
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
            "selection_score": score,
        }
        history.append(record)
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(record, ensure_ascii=False))
        if score < best_score:
            best_score = score
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                fold_name,
                config,
                val_metrics,
            )

    summary = {
        "fold": fold_name,
        "train_scene": train_scene,
        "target_scene": fold["target_scene"],
        "epochs": epochs,
        "best_selection_score": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--fold", choices=("A", "B"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        config["training"]["steps_per_epoch"] = args.steps_per_epoch
    output_dir = args.output_dir or (
        project_root(config) / "outputs" / "checkpoints" / f"fold_{args.fold}"
    )
    checkpoint = train_fold(
        config, args.fold, output_dir.resolve(), torch.device(args.device)
    )
    print(f"Saved best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()


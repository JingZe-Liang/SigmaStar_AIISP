from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from .config import load_config, project_root
from .dataset_v2 import ThresholdNormalizedPatchDataset
from .model import SafeGateUNet
from .train import gate_loss, merge_metrics, save_checkpoint, seed_everything, validate


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def train_joint(
    config: dict,
    output_dir: Path,
    device: torch.device,
    *,
    epochs_override: int | None = None,
    steps_override: int | None = None,
) -> Path:
    settings = dict(config["training"])
    epochs = int(epochs_override or settings["epochs"])
    steps = int(steps_override or settings["steps_per_epoch"])
    batch_size = int(settings["batch_size"])
    seed = int(settings["seed"]) + 200
    seed_everything(seed)

    train_sets: list[ThresholdNormalizedPatchDataset] = []
    val_sets: list[ThresholdNormalizedPatchDataset] = []
    per_scene_samples = max(batch_size, steps * batch_size // 2)
    per_scene_val = max(32, int(settings["val_patches"]) // 2)
    for offset, (fold_name, fold) in enumerate(config["folds"].items()):
        scene = str(fold["train_scene"])
        train_sets.append(
            ThresholdNormalizedPatchDataset(
                config,
                scene,
                fold["train_frames"],
                patch_size=int(settings["patch_size"]),
                samples=per_scene_samples,
                seed=seed + 1000 * offset,
                augment=True,
            )
        )
        val_sets.append(
            ThresholdNormalizedPatchDataset(
                config,
                scene,
                fold["val_frames"],
                patch_size=int(settings["patch_size"]),
                samples=per_scene_val,
                seed=seed + 10_000 + 1000 * offset,
                augment=False,
            )
        )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": int(settings["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ConcatDataset(train_sets), shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(ConcatDataset(val_sets), shuffle=False, **loader_options)
    model = SafeGateUNet(
        input_channels=int(config["model"]["input_channels"]),
        width=int(config["model"]["width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(settings["learning_rate"]) * 0.05,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    history: list[dict] = []
    best_score = math.inf
    started = time.perf_counter()

    for epoch in range(epochs):
        for dataset in train_sets:
            dataset.set_epoch(epoch)
        model.train()
        batches: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"joint v2 epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            features = batch["features"].to(device, non_blocking=True)
            target = batch["target_gate"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(features)
                loss, metrics = gate_loss(logits, target, hard_mask, settings)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batches.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}", gate=f"{metrics['gate_mean']:.3f}")

        train_metrics = merge_metrics(batches)
        val_metrics = validate(model, val_loader, device, settings, amp_enabled)
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
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False))
        if score < best_score:
            best_score = score
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                "joint_v2",
                config,
                val_metrics,
            )

    summary = {
        "training_mode": "joint scenes with temporal holdout validation",
        "scenes": [str(fold["train_scene"]) for fold in config["folds"].values()],
        "epochs": epochs,
        "steps_per_epoch": len(train_loader),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or project_root(config) / "outputs" / "checkpoints" / "joint_v2"
    checkpoint = train_joint(
        config,
        output_dir.resolve(),
        torch.device(args.device),
        epochs_override=args.epochs,
        steps_override=args.steps_per_epoch,
    )
    print(f"Saved best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

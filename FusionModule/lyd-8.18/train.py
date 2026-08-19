from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from lite_unet_fusion.data import RandomPatchDataset
from lite_unet_fusion.losses import fusion_loss
from lite_unet_fusion.model import LiteFusionUNet
from lite_unet_fusion.raw import FusionStream, StreamPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2DNR-prior lightweight U-Net fusion confidence predictor.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--two-d", required=True, type=Path)
    parser.add_argument("--three-d", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=1200)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--resume", type=Path, help="Path to a last.pt checkpoint to resume.")
    parser.add_argument("--init-from", type=Path, help="Load model weights only, then start a new training run.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def worker_init_fn(_: int) -> None:
    # Each worker already uses a CPU core; avoid OpenCV oversubscribing all cores.
    cv2.setNumThreads(1)


def checkpoint_payload(
    model: LiteFusionUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    best_loss: float,
    frame_count: int,
    history: list[dict[str, float | int]],
) -> dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "model_config": model.config,
        "training_config": vars(args),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_loss,
        "frame_count": frame_count,
        "history": history,
    }


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    stream = FusionStream(StreamPaths(args.source, args.two_d, args.three_d))
    validation_count = max(1, round(stream.frame_count * 0.1))
    train_frames = list(range(stream.frame_count - validation_count))
    validation_frames = list(range(stream.frame_count - validation_count, stream.frame_count))
    model = LiteFusionUNet(base_channels=args.base_channels).to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    start_epoch = 1

    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from, not both.")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" not in checkpoint or "scheduler_state_dict" not in checkpoint:
            raise ValueError("The resume checkpoint does not contain optimizer and scheduler state.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_loss = float(checkpoint.get("best_validation_loss", best_loss))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if start_epoch > args.epochs:
            raise ValueError(f"Checkpoint is already at epoch {start_epoch - 1}, which meets --epochs {args.epochs}.")
        print(f"Resuming from epoch {start_epoch} using {args.resume}", flush=True)
    elif args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Initialized model weights from {args.init_from}; optimizer and schedule start fresh.", flush=True)

    loader_kwargs: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        loader_kwargs.update({
            "prefetch_factor": 1,
            "worker_init_fn": worker_init_fn,
        })

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        dataset = RandomPatchDataset(stream, train_frames, args.patch_size, args.samples_per_epoch, args.seed + epoch * 100000)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        totals = {"loss": 0.0, "distill": 0.0, "motion": 0.0, "tv": 0.0}
        for batch in loader:
            batch = {name: value.to(device, non_blocking=use_cuda) for name, value in batch.items()}
            if use_cuda:
                batch["input"] = batch["input"].contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                beta = model(batch["input"])
                loss, components = fusion_loss(beta, batch["teacher"], batch["motion"], batch["flatness"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += value
        scheduler.step()

        model.eval()
        validation = RandomPatchDataset(stream, validation_frames, args.patch_size, 160, args.seed + epoch)
        validation_loader = DataLoader(validation, batch_size=args.batch_size, **loader_kwargs)
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {name: value.to(device, non_blocking=use_cuda) for name, value in batch.items()}
                if use_cuda:
                    batch["input"] = batch["input"].contiguous(memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    loss, _ = fusion_loss(model(batch["input"]), batch["teacher"], batch["motion"], batch["flatness"])
                validation_loss += float(loss)
        row = {name: value / len(loader) for name, value in totals.items()}
        row.update({"epoch": epoch, "validation_loss": validation_loss / len(validation_loader), "learning_rate": optimizer.param_groups[0]["lr"]})
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_loss"] < best_loss:
            best_loss = float(row["validation_loss"])
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, args, epoch, best_loss, stream.frame_count, history)
        save_checkpoint(args.output_dir / "last.pt", payload)
        if row["validation_loss"] <= best_loss:
            save_checkpoint(args.output_dir / "best.pt", payload)

    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Wrote checkpoints: {args.output_dir / 'best.pt'} and {args.output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

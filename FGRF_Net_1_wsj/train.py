"""Train FGRF-Net with the three requested self-supervised losses."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset import build_dataset
from losses import total_loss
from model import FGRFNet


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_device(name: str, expected_gpu: str) -> torch.device:
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        gpu_name = torch.cuda.get_device_name(device)
        if expected_gpu and expected_gpu.casefold() not in gpu_name.casefold():
            raise RuntimeError(
                f"Selected GPU is {gpu_name!r}, not the required {expected_gpu!r}. "
                "Use CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1."
            )
        print(f"gpu={gpu_name} device={device}")
    return device


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Repeat this option to train on multiple scene configs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-gpu", default="RTX 4090",
                        help="Required substring for a CUDA device; empty disables the check.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override config.optimization.learning_rate.")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override config.optimization.weight_decay.")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--resume", default="")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    config_paths = args.config or ["config_example_645.json"]
    configs = [load_config(path) for path in config_paths]
    config = configs[0]
    set_seed(int(config.get("seed", 1234)))
    device = choose_device(args.device, args.expected_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    datasets = [build_dataset(scene_config, training=True) for scene_config in configs]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=max(1, args.prefetch_factor),
        )
    loader = DataLoader(dataset, **loader_kwargs)
    model = FGRFNet(
        input_channels=12,
        base_channels=int(config.get("model", {}).get("base_channels", 24)),
        threshold_floor=float(config.get("model", {}).get("threshold_floor", 0.008)),
        initial_threshold=float(config.get("model", {}).get("initial_threshold", 0.01)),
    ).to(device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(config.get("optimization", {}).get("learning_rate", 2e-4))
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(config.get("optimization", {}).get("weight_decay", 1e-4))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    packed_hw = datasets[0].packed_hw
    print(
        f"device={device} scenes={len(datasets)} frames={len(dataset)} packed={packed_hw} "
        f"batch={args.batch_size} workers={args.workers} prefetch={args.prefetch_factor} "
        f"lr={learning_rate:g} amp={use_amp}"
    )
    run_started = time.perf_counter()
    last_step_started = run_started
    for epoch in range(start_epoch, start_epoch + args.epochs):
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                prediction = model(
                    batch["base"], batch["temporal_residual"], batch["noisy_residual"],
                    batch["static_mask"],
                )
                losses = total_loss(
                    prediction, batch,
                    pseudo_gt_weight=float(config.get("loss", {}).get("pseudo_gt_weight", 1.0)),
                    raw_weight=float(config.get("loss", {}).get("raw_weight", 0.25)),
                    base_weight=float(config.get("loss", {}).get("base_weight", 1.0)),
                )
            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            now = time.perf_counter()
            if global_step == 1 or global_step % int(config.get("logging", {}).get("print_every", 10)) == 0:
                values = " ".join(f"{name}={value.detach().item():.6f}" for name, value in losses.items())
                interval = max(now - last_step_started, 1e-6)
                print(
                    f"epoch={epoch} step={global_step} {values} "
                    f"steps_per_sec={max(1, int(config.get('logging', {}).get('print_every', 10))) / interval:.2f}"
                )
                last_step_started = now
            if args.max_steps and global_step >= args.max_steps:
                break
        checkpoint_path = save_dir / f"fgrf_epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "global_step": global_step,
                "configs": configs,
            },
            checkpoint_path,
        )
        print(f"saved {checkpoint_path}")
        if device.type == "cuda":
            print(f"peak_cuda_mib={torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f}")
        if args.max_steps and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()

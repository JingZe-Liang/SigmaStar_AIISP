"""Train FGRF-Net v2.0 with three-input inference and flow-only supervision."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset import build_dataset
from losses import LossWeights, TextureFusionLoss
from model import TextureGateNet


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str, expected_gpu: str) -> torch.device:
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        gpu_name = torch.cuda.get_device_name(device)
        if expected_gpu and expected_gpu.casefold() not in gpu_name.casefold():
            raise RuntimeError(f"Selected GPU is {gpu_name!r}, expected substring {expected_gpu!r}")
        print(f"device={device} gpu={gpu_name}", flush=True)
    return device


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_loss(config: dict[str, Any]) -> TextureFusionLoss:
    values = config["loss"]
    return TextureFusionLoss(
        LossWeights(
            gate=float(values.get("gate_weight", 1.0)),
            texture=float(values.get("texture_weight", 0.35)),
            motion=float(values.get("motion_weight", 0.25)),
        ),
        oracle_kernel=int(values.get("oracle_kernel", 5)),
        texture_kernels=tuple(int(value) for value in values.get("texture_kernels", (3, 7))),
        texture_threshold=float(values.get("texture_threshold", 0.003)),
        candidate_min_norm=float(values.get("candidate_min_norm", 0.001)),
        candidate_range=float(values.get("candidate_range", 0.01)),
    )


def run_epoch(
    model: TextureGateNet,
    criterion: TextureFusionLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    print_every: int,
    max_steps: int,
    global_step: int,
) -> tuple[dict[str, float], int, bool]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    completed = True
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                alpha = model(batch["noisy"], batch["base"], batch["temporal"])
                loss, metrics = criterion(alpha, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            if training:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                global_step += 1
        batch_size = int(batch["noisy"].shape[0])
        for key, value in metrics.items():
            totals[key] += float(value.detach().item()) * batch_size
        samples += batch_size
        if training and (global_step == 1 or global_step % print_every == 0):
            elapsed = max(time.perf_counter() - started, 1e-6)
            values = " ".join(f"{key}={value.detach().item():.5f}" for key, value in metrics.items())
            print(f"step={global_step} {values} samples_per_sec={samples / elapsed:.2f}", flush=True)
        if training and max_steps and global_step >= max_steps:
            completed = False
            break
    return {key: value / max(samples, 1) for key, value in totals.items()}, global_step, completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, help="Repeat for each training scene.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-gpu", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints_v2"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = [load_config(path) for path in args.config]
    config = configs[0]
    set_seed(int(config.get("seed", 1234)))
    device = choose_device(args.device, args.expected_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    train_sets = []
    for scene_index, scene in enumerate(configs, start=1):
        print(f"building train dataset {scene_index}/{len(configs)} (flow cache may take a few minutes)", flush=True)
        train_sets.append(build_dataset(scene, samples_per_epoch=args.train_samples, training=True, seed=int(scene.get("seed", 1234))))
    val_sets = []
    for scene_index, (scene, train_set) in enumerate(zip(configs, train_sets), start=1):
        print(f"building validation dataset {scene_index}/{len(configs)}", flush=True)
        val_sets.append(build_dataset(scene, samples_per_epoch=args.val_samples, training=False, seed=int(scene.get("seed", 1234)) + 100_000, flow_sequences=(train_set.forward, train_set.backward)))
    print("creating data loaders and model", flush=True)
    train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    loader_args: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "shuffle": False,
    }
    if args.workers > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_dataset, **loader_args)
    val_loader = DataLoader(val_dataset, **loader_args)

    model = TextureGateNet(base_channels=int(config["model"].get("base_channels", 32))).to(device)
    criterion = make_loss(config).to(device)
    learning_rate = float(args.learning_rate if args.learning_rate is not None else config["optimization"].get("learning_rate", 2e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else config["optimization"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint.get("best_val", best_val))

    args.save_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"configs": configs, "args": vars(args), "model_parameters": sum(p.numel() for p in model.parameters())}
    (args.save_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"scenes={len(configs)} train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
        f"batch={args.batch_size} model_parameters={run_config['model_parameters']} amp={amp_enabled}",
        flush=True,
    )
    print("model_input=noisy,base,temporal; flow=training supervision only", flush=True)
    print_every = int(config.get("logging", {}).get("print_every", 10))
    for epoch in range(start_epoch, args.epochs + 1):
        for dataset in train_sets:
            dataset.set_epoch(epoch)
        train_metrics, global_step, completed = run_epoch(
            model, criterion, train_loader, device, optimizer, scaler, amp_enabled,
            print_every, args.max_steps, global_step,
        )
        val_metrics, _, _ = run_epoch(
            model, criterion, val_loader, device, None, None, amp_enabled,
            print_every, 0, global_step,
        )
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.6f} val_total={val_metrics['total']:.6f} "
            f"val_alpha={val_metrics['alpha']:.4f} static={val_metrics['static_fraction']:.4f}",
            flush=True,
        )
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "global_step": global_step, "best_val": best_val,
            "model_config": {"base_channels": int(config["model"].get("base_channels", 32))},
            "configs": configs, "train_metrics": train_metrics, "val_metrics": val_metrics,
        }
        torch.save(state, args.save_dir / f"fgrf_v2_epoch_{epoch:03d}.pt")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            state["best_val"] = best_val
            torch.save(state, args.save_dir / "fgrf_v2_best.pt")
        if args.max_steps and not completed:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

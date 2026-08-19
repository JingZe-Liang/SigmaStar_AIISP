from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import FRAME_COUNT, FullMosaicMotionDataset, discover_sequences
from infer_samples import export_checkpoint_samples
from losses import candidate_gradient_loss, masked_noisy_loss
from masking import SameCFAMasker
from model import load_pretrained_model


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="BLCFA + robust MD + 候选梯度的严格 masked 微调")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"data_root", "motion_cache_root", "init_checkpoint", "epochs", "steps_per_epoch", "batch_size", "patch_size", "learning_rate", "mask_points_per_patch", "mask_min_distance", "train_frame_end", "noisy_loss_weight", "gradient_loss_weight", "algorithm_threshold", "algorithm_transition"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config 缺少字段: {sorted(missing)}")
    if not 2 <= int(config["train_frame_end"]) < FRAME_COUNT:
        raise ValueError("train_frame_end 必须在 2~199")
    if abs(float(config["noisy_loss_weight"]) + float(config["gradient_loss_weight"]) - 1.0) > 1e-6:
        raise ValueError("noisy_loss_weight 与 gradient_loss_weight 之和必须为 1")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, model, optimizer, epoch: int, step: int, record: dict, config: dict, initial_hash: str) -> None:
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step, "record": record, "config": config, "variant": "strict_blcfa_md_grad", "initial_checkpoint_sha256": initial_hash, "created_at": datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def batch_losses(model, batch, device: torch.device, masker: SameCFAMasker, config: dict) -> tuple[torch.Tensor, dict]:
    image_2dnr, image_3dnr, noisy_current, motion_mask = [item.to(device, non_blocking=True) for item in batch]
    target_noisy = noisy_current.clone()
    masked_2dnr, masked_3dnr, masked_noisy, supervised = masker.mask(image_2dnr, image_3dnr, noisy_current)
    prediction = model(masked_2dnr, masked_3dnr, masked_noisy, motion_mask)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("模型输出出现 NaN 或 Inf")
    noisy_value = masked_noisy_loss(prediction, target_noisy, supervised, float(config["epsilon"]))
    gradient_value, candidate_2d_fraction = candidate_gradient_loss(prediction, image_2dnr, image_3dnr, motion_mask, float(config["algorithm_threshold"]), float(config["algorithm_transition"]), float(config["epsilon"]))
    total = float(config["noisy_loss_weight"]) * noisy_value + float(config["gradient_loss_weight"]) * gradient_value
    if not torch.isfinite(total):
        raise FloatingPointError("训练 loss 出现 NaN 或 Inf")
    values = {"masked_noisy_loss": float(noisy_value.detach().cpu()), "candidate_gradient_loss": float(gradient_value.detach().cpu()), "motion_coverage": float(motion_mask.mean().detach().cpu()), "candidate_2dnr_fraction": float(candidate_2d_fraction.detach().cpu()), "total_loss": float(total.detach().cpu())}
    return total, values


@torch.no_grad()
def evaluate(model, loader, device: torch.device, masker: SameCFAMasker, config: dict, batches: int) -> dict:
    model.eval()
    values = []
    iterator = iter(loader)
    for _ in range(batches):
        _, item = batch_losses(model, next(iterator), device, masker, config)
        values.append(item)
    model.train()
    return {f"validation_{key}": float(np.mean([item[key] for item in values])) for key in values[0]}


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 要求 CUDA，但当前 PyTorch 未检测到 CUDA")
    initial_checkpoint = Path(config["init_checkpoint"])
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f"初始权重不存在: {initial_checkpoint}")
    sequences = discover_sequences(Path(config["data_root"]), tuple(config["sequence_names"]), Path(config["motion_cache_root"]))
    split = int(config["train_frame_end"])
    train_data = FullMosaicMotionDataset(sequences, tuple(range(1, split)), int(config["steps_per_epoch"]) * int(config["batch_size"]), int(config["patch_size"]), int(config["seed"]))
    validation_data = FullMosaicMotionDataset(sequences, tuple(range(split, FRAME_COUNT)), int(config["validation_batches"]) * int(config["batch_size"]), int(config["patch_size"]), int(config["seed"]) + 100000)
    loader_args = {"batch_size": int(config["batch_size"]), "num_workers": int(config["num_workers"]), "pin_memory": device.type == "cuda", "drop_last": True}
    train_loader, validation_loader = DataLoader(train_data, **loader_args), DataLoader(validation_data, **loader_args)
    model = load_pretrained_model(str(initial_checkpoint), device)
    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    run_root = ROOT / "runs_v3" / "strict_blcfa_md_grad"
    checkpoints, logs = run_root / "checkpoints", run_root / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    train_masker = SameCFAMasker(int(config["mask_points_per_patch"]), int(config["mask_min_distance"]), int(config["seed"]))
    validation_masker = SameCFAMasker(int(config["mask_points_per_patch"]), int(config["mask_min_distance"]), int(config["seed"]) + 1)
    steps_per_epoch = 1 if args.smoke_test else int(config["steps_per_epoch"])
    epochs = 1 if args.smoke_test else int(config["epochs"])
    global_step = 0
    print("variant=strict_blcfa_md_grad; input=linear BLCFA; motion=precomputed robust MD; BPN=center/cross-CFA forbidden")
    print(f"train frames=1~{split - 1}; validation frames={split}~199; steps={epochs * steps_per_epoch}")
    log_name = "smoke_test.jsonl" if args.smoke_test else "train.jsonl"
    with (logs / log_name).open("a", encoding="utf-8") as log_file:
        for epoch in range(epochs):
            train_data.set_epoch(epoch)
            model.train()
            iterator = iter(train_loader)
            for _ in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                total, record = batch_losses(model, next(iterator), device, train_masker, config)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1
                record.update({"epoch": epoch, "step": global_step})
                if global_step % int(config["log_every_steps"]) == 0 or args.smoke_test:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if global_step % int(config["validation_every_steps"]) == 0 or args.smoke_test:
                    record.update(evaluate(model, validation_loader, device, validation_masker, config, int(config["validation_batches"])))
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if not args.smoke_test and global_step % int(config["save_every_steps"]) == 0:
                    checkpoint_path = checkpoints / f"checkpoint_step_{global_step:06d}.pth"
                    save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, record, config, hash_file(initial_checkpoint))
                    export_checkpoint_samples(checkpoint_path, config, model=model)
            if not args.smoke_test:
                save_checkpoint(checkpoints / "checkpoint_last.pth", model, optimizer, epoch, global_step, record, config, hash_file(initial_checkpoint))
    print("smoke test 结束；未保存 checkpoint" if args.smoke_test else f"训练结束: {checkpoints / 'checkpoint_last.pth'}")


if __name__ == "__main__":
    main()

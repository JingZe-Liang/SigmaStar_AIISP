"""CPU-testable training loop and command-line entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint_strict, save_checkpoint_atomic
from .config import (
    DatasetConfig,
    ExperimentConfig,
    config_fingerprint,
    load_dataset_config,
    load_experiment_config,
)
from .data import FusionPatchDataset
from .losses import FusionLoss, LossBreakdown
from .model import CausalRawFusionNet
from .raw import normalize_raw, pack_rggb


TRAIN_IMAGE_KEYS: tuple[str, ...] = (
    "prev_noisy",
    "curr_noisy",
    "denoised",
    "fused",
    "target",
)


def train_step(
    model: nn.Module,
    batch: Mapping[str, object],
    loss_fn: FusionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None = None,
    device: torch.device | str = torch.device("cpu"),
    amp: bool = False,
) -> dict[str, float]:
    """Run one finite-checked optimization step over a paired batch."""
    target_device = torch.device(device)
    tensors = _move_image_batch(batch, target_device)
    _validate_model_device(model, target_device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        device_type=target_device.type,
        enabled=bool(amp),
    ):
        output = model(
            tensors["prev_noisy"],
            tensors["curr_noisy"],
            tensors["denoised"],
            tensors["fused"],
        )
        _ensure_finite_model_output(output.prediction, output.gate, output.correction)
        breakdown = loss_fn(
            output,
            tensors["denoised"],
            tensors["fused"],
            tensors["target"],
        )
    _ensure_finite_breakdown(breakdown)

    if scaler is not None:
        if not amp:
            raise ValueError("GradScaler 只能与 amp=True 一起使用")
        scaler.scale(breakdown.total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = _gradient_norm(model)
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("gradient norm is not finite")
        scaler.step(optimizer)
        scaler.update()
    else:
        breakdown.total.backward()
        gradient_norm = _gradient_norm(model)
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("gradient norm is not finite")
        optimizer.step()

    return {
        "total": _scalar(breakdown.total),
        "reconstruction": _scalar(breakdown.reconstruction),
        "gradient": _scalar(breakdown.gradient),
        "gate": _scalar(breakdown.gate),
        "residual": _scalar(breakdown.residual),
        "range": _scalar(breakdown.range),
        "gradient_norm": gradient_norm,
    }


def run_training(
    config_path: Path,
    output_dir: Path,
    resume_path: Path | None = None,
) -> Path:
    """Train from a strict experiment config and return the best checkpoint."""
    experiment = load_experiment_config(Path(config_path))
    dataset_config = load_dataset_config(experiment.dataset_path)
    _validate_train_config(experiment)
    device = torch.device(experiment.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 训练，但当前环境没有可用 GPU")
    _seed_everything(experiment.train.seed)

    model = CausalRawFusionNet(experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.train.learning_rate,
        weight_decay=experiment.train.weight_decay,
    )
    loss_fn = FusionLoss(
        experiment.loss,
        white_level=dataset_config.layout.white_level,
        target_black_level=dataset_config.layout.target_black_level,
    )
    scaler = _make_scaler(device, experiment.train.amp)
    fingerprint = config_fingerprint(dataset_config, experiment)
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        restored = load_checkpoint_strict(
            Path(resume_path),
            expected_fingerprint=fingerprint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        start_epoch = restored.epoch + 1
        global_step = restored.global_step
    train_dataset = FusionPatchDataset(
        dataset_config,
        sequence_name=experiment.split.train_sequence,
        frame_range=experiment.split.train_frames,
        patch_size_packed=experiment.train.patch_size_packed,
        samples_per_epoch=experiment.train.samples_per_epoch,
        seed=experiment.train.seed,
    )
    validation_dataset = _CenterValidationDataset(
        dataset_config,
        sequence_name=experiment.split.validation_sequence,
        frame_range=experiment.split.validation_frames,
        patch_size_packed=experiment.train.patch_size_packed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment.train.batch_size,
        shuffle=False,
        num_workers=experiment.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "train.jsonl"
    best_path = destination / "best.pt"
    best_value = (
        _resume_log_best(
            log_path,
            checkpoint_epoch=start_epoch - 1,
            checkpoint_step=global_step,
        )
        if resume_path is not None
        else math.inf
    )
    if start_epoch >= experiment.train.epochs:
        if best_path.is_file():
            load_checkpoint_strict(
                best_path,
                expected_fingerprint=fingerprint,
                device="cpu",
            )
            return best_path
        raise ValueError("恢复检查点已达到配置的训练 epoch，且输出目录没有 best.pt")
    log_mode = "a" if resume_path is not None else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for epoch in range(start_epoch, experiment.train.epochs):
            train_dataset.set_epoch(epoch)
            train_totals: list[dict[str, float]] = []
            for batch in train_loader:
                train_totals.append(
                    train_step(
                        model,
                        batch,
                        loss_fn,
                        optimizer,
                        scaler=scaler,
                        device=device,
                        amp=experiment.train.amp,
                    )
                )
                global_step += 1
            validation = _evaluate(model, validation_loader, loss_fn, device, experiment.train.amp)
            train_summary = _average_metrics(train_totals)
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_summary,
                "validation": validation,
            }
            log_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log_file.flush()
            state = _checkpoint_state(
                model,
                optimizer,
                scaler,
                epoch=epoch,
                global_step=global_step,
                fingerprint=fingerprint,
                experiment=experiment,
                dataset=dataset_config,
            )
            save_checkpoint_atomic(destination / "last.pt", state)
            if validation["total"] < best_value:
                best_value = validation["total"]
                save_checkpoint_atomic(best_path, state)
    if not best_path.is_file():
        raise RuntimeError("训练未生成 best.pt")
    return best_path


class _CenterValidationDataset(FusionPatchDataset):
    """Use one deterministic center crop for every validation frame."""

    def __init__(
        self,
        config: DatasetConfig,
        *,
        sequence_name: str,
        frame_range: tuple[int, int],
        patch_size_packed: int,
    ) -> None:
        super().__init__(
            config,
            sequence_name=sequence_name,
            frame_range=frame_range,
            patch_size_packed=patch_size_packed,
            samples_per_epoch=len(range(frame_range[0], frame_range[1] + 1)),
            seed=0,
            force_transform=(False, False, False),
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self.frame_indices):
            raise IndexError(f"validation index out of range: {index}")
        frame_index = self.frame_indices[index]
        sensor_size = self.patch_size_packed * 2
        sensor_height = self.config.layout.height
        sensor_width = self.config.layout.width
        sensor_top = _center_even_start(sensor_height, sensor_size)
        sensor_left = _center_even_start(sensor_width, sensor_size)
        crops = {
            "prev_noisy": self.readers.noisy.read_crop(
                frame_index - 1, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "curr_noisy": self.readers.noisy.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "denoised": self.readers.denoised.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "fused": self.readers.fused.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
            "target": self.readers.target.read_crop(
                frame_index, sensor_top, sensor_left, sensor_size, sensor_size
            ),
        }
        black_levels = {
            "prev_noisy": self.config.layout.noisy_black_level,
            "curr_noisy": self.config.layout.noisy_black_level,
            "denoised": self.config.layout.candidate_black_level,
            "fused": self.config.layout.candidate_black_level,
            "target": self.config.layout.target_black_level,
        }
        result = {
            name: torch.from_numpy(
                pack_rggb(
                    normalize_raw(
                        crop,
                        black_levels[name],
                        self.config.layout.white_level,
                    )
                )
            )
            for name, crop in crops.items()
        }
        result["sequence_index"] = torch.tensor(self.sequence_index, dtype=torch.int64)
        result["frame_index"] = torch.tensor(frame_index, dtype=torch.int64)
        return result


def _move_image_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key in TRAIN_IMAGE_KEYS:
        if key not in batch:
            raise KeyError(f"batch 缺少 {key}")
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"batch {key} 必须是 Tensor")
        result[key] = value.to(device=device, non_blocking=device.type == "cuda")
    return result


def _validate_model_device(model: nn.Module, device: torch.device) -> None:
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("model 必须包含参数") from error
    param_device = parameter.device
    if param_device.type != device.type:
        raise ValueError(f"model 参数位于 {param_device}，但训练设备为 {device}")
    if param_device.type == "cuda" and device.index is not None and param_device.index != device.index:
        raise ValueError(f"model 参数位于 {param_device}，但训练设备为 {device}")


def _ensure_finite_breakdown(breakdown: LossBreakdown) -> None:
    values = (
        breakdown.total,
        breakdown.reconstruction,
        breakdown.gradient,
        breakdown.gate,
        breakdown.residual,
        breakdown.range,
    )
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("loss contains non-finite value")


def _ensure_finite_model_output(*values: Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("loss input contains non-finite model output")


def _gradient_norm(model: nn.Module) -> float:
    squared: list[Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared.append(parameter.grad.detach().float().norm(2).square())
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().item())


def _scalar(value: Tensor) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise FloatingPointError("loss contains non-finite value")
    return result


def _evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    loss_fn: FusionLoss,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            tensors = _move_image_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                output = model(
                    tensors["prev_noisy"],
                    tensors["curr_noisy"],
                    tensors["denoised"],
                    tensors["fused"],
                )
                _ensure_finite_model_output(
                    output.prediction,
                    output.gate,
                    output.correction,
                )
                breakdown = loss_fn(
                    output,
                    tensors["denoised"],
                    tensors["fused"],
                    tensors["target"],
                )
            _ensure_finite_breakdown(breakdown)
            records.append(
                {
                    "total": _scalar(breakdown.total),
                    "reconstruction": _scalar(breakdown.reconstruction),
                    "gradient": _scalar(breakdown.gradient),
                    "gate": _scalar(breakdown.gate),
                    "residual": _scalar(breakdown.residual),
                    "range": _scalar(breakdown.range),
                }
            )
    return _average_metrics(records)


def _average_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("训练或验证 loader 没有样本")
    names = records[0].keys()
    return {name: sum(record[name] for record in records) / len(records) for name in names}


def _center_even_start(length: int, crop: int) -> int:
    if length <= 0 or crop <= 0 or crop > length:
        raise ValueError("center crop 尺寸必须为正且不能超过图像尺寸")
    return ((length - crop) // 2) // 2 * 2


def _checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    *,
    epoch: int,
    global_step: int,
    fingerprint: str,
    experiment: ExperimentConfig,
    dataset: DatasetConfig,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config_fingerprint": fingerprint,
        "experiment_config": {
            "experiment": _plain_config(experiment),
            "dataset": _plain_config(dataset),
        },
        "seed": experiment.train.seed,
    }


def _plain_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _plain_config(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_config(item) for item in value]
    return value


def _resume_log_best(
    log_path: Path,
    *,
    checkpoint_epoch: int,
    checkpoint_step: int,
) -> float:
    if not log_path.is_file():
        return math.inf
    best = math.inf
    previous_epoch = -1
    previous_step = -1
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            record = json.loads(line)
            epoch = int(record["epoch"])
            step = int(record["global_step"])
            value = float(record["validation"]["total"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"训练日志第 {line_number} 行无效: {log_path}") from error
        if not math.isfinite(value):
            raise ValueError(f"训练日志第 {line_number} 行验证损失非有限")
        if epoch <= previous_epoch or step <= previous_step:
            raise ValueError(f"训练日志第 {line_number} 行 epoch/global_step 未严格递增")
        previous_epoch = epoch
        previous_step = step
        best = min(best, value)
    if previous_epoch >= 0 and (
        previous_epoch != checkpoint_epoch or previous_step != checkpoint_step
    ):
        raise ValueError("训练日志末尾 epoch/global_step 与 checkpoint 不一致")
    return best


def _make_scaler(device: torch.device, amp: bool) -> Any | None:
    if not amp or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler(device="cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_train_config(experiment: ExperimentConfig) -> None:
    train = experiment.train
    if train.batch_size <= 0 or train.epochs <= 0 or train.num_workers < 0:
        raise ValueError("训练 batch_size/epochs 必须为正，num_workers 不能为负")
    if train.learning_rate <= 0 or train.weight_decay < 0:
        raise ValueError("learning_rate 必须为正，weight_decay 不能为负")
    if train.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 训练，但当前环境没有可用 GPU")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练 RAW 融合网络")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args(argv)
    best = run_training(arguments.config, arguments.output_dir, arguments.resume)
    print(best)


if __name__ == "__main__":
    main()

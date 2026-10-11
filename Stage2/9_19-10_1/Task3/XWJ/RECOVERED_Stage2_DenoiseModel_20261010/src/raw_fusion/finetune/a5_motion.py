"""A5 motion-supervision smoke training from the frozen A3 baseline."""

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from raw_fusion.finetune.a3 import (
    A3Config,
    PhaseADataset,
    build_rvideformer,
    compute_loss,
    update_ema,
)


@dataclass(frozen=True, slots=True)
class A5MotionConfig:
    supervision_dir: str
    noise_dir: str
    a3_checkpoint: str
    dynamic_train_manifest: str
    dynamic_holdout_manifest: str
    output_dir: str
    steps: int = 20
    learning_rate: float = 1e-5
    dynamic_weight: float = 0.1
    ema_decay: float = 0.99
    static_anchor_weight: float = 1.0
    seed: int = 20260925
    device: str = "cuda:1"
    patch_size: int = 64
    eval_crops_per_window: int = 4
    trainable_scope: str = "all"
    static_target_mode: str = "a1"
    motion_boundary_radius: int = 1


class DynamicWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Deterministic 64-pixel crops from saved eight-frame dynamic windows."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        patch_size: int = 64,
        crops_per_window: int = 1,
        seed: int = 0,
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("temporal_frames") != 8:
            raise ValueError("dynamic manifest must contain eight-frame windows")
        if manifest.get("confidence_definition", {}).get("flow_magnitude_px_gte") != 1.0:
            raise ValueError("A5 smoke requires the approved flow >= 1 px confidence rule")
        allowed = manifest.get("allowed_frame_range")
        if not isinstance(allowed, list) or len(allowed) != 2:
            raise ValueError("dynamic manifest must declare an allowed_frame_range")
        self.allowed_frame_range = (int(allowed[0]), int(allowed[1]))
        self.records = manifest["windows"]
        if not self.records:
            raise ValueError("dynamic manifest has no windows")
        if patch_size <= 0 or crops_per_window <= 0:
            raise ValueError("patch_size and crops_per_window must be positive")
        self.patch_size = patch_size
        self.crops_per_window = crops_per_window
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_window

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record_index = (index // self.crops_per_window) % len(self.records)
        record = self.records[record_index]
        sample = np.load(record["sample"])
        source = sample["source"].astype(np.float32, copy=False)
        target = sample["target"].astype(np.float32, copy=False)
        confidence = sample["confidence"].astype(bool, copy=False)
        if source.shape != target.shape or source.ndim != 4 or source.shape[0:2] != (8, 4):
            raise ValueError(f"invalid dynamic sample shape {source.shape}")
        if confidence.shape != (8, source.shape[2], source.shape[3]):
            raise ValueError(f"invalid confidence shape {confidence.shape}")
        if source.shape[2] < self.patch_size or source.shape[3] < self.patch_size:
            raise ValueError("dynamic sample is smaller than the training patch")

        positions = np.argwhere(confidence)
        if positions.size == 0:
            raise ValueError(f"dynamic sample has no confidence pixels: {record['sample']}")
        rng = np.random.default_rng(self.seed + index * 104729)
        _, center_y, center_x = positions[int(rng.integers(0, len(positions)))]
        y_low = max(0, int(center_y) - self.patch_size + 1)
        y_high = min(int(center_y), source.shape[2] - self.patch_size)
        x_low = max(0, int(center_x) - self.patch_size + 1)
        x_high = min(int(center_x), source.shape[3] - self.patch_size)
        y = int(rng.integers(y_low, y_high + 1))
        x = int(rng.integers(x_low, x_high + 1))
        crop = np.s_[..., y:y + self.patch_size, x:x + self.patch_size]
        source_crop = np.asarray(source[crop])
        target_crop = np.asarray(target[crop])
        mask_crop = np.asarray(confidence[:, y:y + self.patch_size, x:x + self.patch_size])
        if not mask_crop.any():
            raise RuntimeError("selected dynamic crop unexpectedly has no confidence pixels")
        return (
            torch.from_numpy(source_crop.copy()),
            torch.from_numpy(target_crop.copy()),
            torch.from_numpy(mask_crop[:, None].astype(np.float32)),
        )


def dynamic_charbonnier_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, float]]:
    expanded = mask.expand_as(prediction)
    pixels = expanded.sum()
    if float(pixels.detach().cpu()) <= 0.0:
        raise ValueError("dynamic loss mask is empty")
    loss = (torch.sqrt((prediction - target).square() + 1e-6) * expanded).sum() / pixels
    return loss, {
        "dynamic_loss": float(loss.detach().cpu()),
        "dynamic_mask_fraction": float(mask.mean().detach().cpu()),
    }


def dilate_motion_mask(mask: Tensor, radius: int) -> Tensor:
    """Dilate each frame's motion mask spatially without mixing time steps."""
    if radius < 0:
        raise ValueError("motion boundary radius must be non-negative")
    if mask.ndim not in {4, 5}:
        raise ValueError(f"expected a 4D or 5D mask, got shape {tuple(mask.shape)}")
    if mask.shape[-3] != 1:
        raise ValueError(f"expected a single mask channel, got shape {tuple(mask.shape)}")
    if radius == 0:
        return mask.clone()

    kernel = 2 * radius + 1
    if mask.ndim == 4:
        frames = mask
        leading_shape: tuple[int, ...] = (mask.shape[0],)
    else:
        leading_shape = (mask.shape[0], mask.shape[1])
        frames = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    dilated = F.max_pool2d(frames.float(), kernel_size=kernel, stride=1, padding=radius)
    return dilated.reshape(*leading_shape, 1, mask.shape[-2], mask.shape[-1])


def teacher_anchor_loss(
    prediction: Tensor,
    teacher_target: Tensor,
    static_mask: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Apply the A3 teacher only where the dynamic exclusion mask is absent."""
    pixels = static_mask.expand_as(prediction).sum()
    mask_fraction = float(static_mask.detach().mean().cpu())
    if float(pixels.detach().cpu()) <= 0.0:
        # A crop can be fully covered by motion plus its boundary band.
        return prediction.sum() * 0.0, {
            "loss": 0.0,
            "charbonnier": 0.0,
            "gradient": 0.0,
            "highpass": 0.0,
            "static_mask_fraction": mask_fraction,
            "static_mask_empty": 1.0,
        }
    loss, metrics = compute_loss(
        prediction,
        teacher_target,
        static_mask,
        gradient_weight=0.1,
        highpass_weight=0.1,
    )
    metrics["static_mask_fraction"] = mask_fraction
    metrics["static_mask_empty"] = 0.0
    return loss, metrics


def _load_a3_ema(checkpoint: Path, device: torch.device) -> tuple[nn.Module, nn.Module]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise ValueError("A5 requires an A3 checkpoint containing EMA weights")
    model = build_rvideformer()
    model.load_state_dict(payload["ema"], strict=True)
    ema = build_rvideformer()
    ema.load_state_dict(payload["ema"], strict=True)
    return model.to(device), ema.to(device)


def set_trainable_scope(model: nn.Module, scope: str) -> dict[str, int]:
    if scope not in {"all", "mutual_attention"}:
        raise ValueError(f"unknown trainable scope {scope!r}")
    tensors = 0
    elements = 0
    for name, parameter in model.named_parameters():
        trainable = scope == "all" or any(
            token in name for token in (".hi_qkv_mut.", ".g_q_mut.", ".g_kv_mut.")
        )
        parameter.requires_grad_(trainable)
        if trainable:
            tensors += 1
            elements += parameter.numel()
    if tensors == 0:
        raise ValueError(f"trainable scope {scope!r} selected no parameters")
    return {"tensors": tensors, "elements": elements}


@torch.no_grad()
def _evaluate_dynamic(model: nn.Module, dataset: DynamicWindowDataset, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    mask_fractions: list[float] = []
    for index in range(len(dataset)):
        inputs, targets, masks = dataset[index]
        prediction = model(inputs[None].to(device)).clamp(0.0, 1.0)
        loss, metrics = dynamic_charbonnier_loss(
            prediction, targets[None].to(device), masks[None].to(device)
        )
        losses.append(float(loss.detach().cpu()))
        mask_fractions.append(metrics["dynamic_mask_fraction"])
    return {
        "loss": float(np.mean(losses)),
        "mask_fraction": float(np.mean(mask_fractions)),
        "samples": len(dataset),
    }


def _static_config(config: A5MotionConfig) -> A3Config:
    return A3Config(
        dataset_root="/HardDisk/jingzeliang/Data/Company_Dataset/dataset",
        supervision_dir=config.supervision_dir,
        noise_dir=config.noise_dir,
        output_dir=config.output_dir,
        checkpoint=config.a3_checkpoint,
        seed=config.seed,
        steps=config.steps,
        learning_rate=config.learning_rate,
        holdout_gain="1000x",
        device=config.device,
    )


def train_motion_smoke(config: A5MotionConfig) -> dict[str, object]:
    if config.steps <= 0 or config.learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    if not 0.0 <= config.dynamic_weight <= 1.0:
        raise ValueError("dynamic_weight must be in [0, 1]")
    if config.static_anchor_weight <= 0.0:
        raise ValueError("static_anchor_weight must be positive")
    if not 0.0 < config.ema_decay < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    if config.static_target_mode not in {"a1", "a3_teacher", "same_window_a3_teacher"}:
        raise ValueError(
            "static_target_mode must be 'a1', 'a3_teacher', or 'same_window_a3_teacher'"
        )
    if config.motion_boundary_radius < 0:
        raise ValueError("motion_boundary_radius must be non-negative")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A5 motion smoke requires CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_dynamic = DynamicWindowDataset(
        Path(config.dynamic_train_manifest), patch_size=config.patch_size,
        crops_per_window=1, seed=config.seed,
    )
    holdout_dynamic = DynamicWindowDataset(
        Path(config.dynamic_holdout_manifest), patch_size=config.patch_size,
        crops_per_window=config.eval_crops_per_window, seed=config.seed + 1_000_003,
    )
    if train_dynamic.allowed_frame_range[1] > holdout_dynamic.allowed_frame_range[0]:
        raise ValueError("dynamic training and holdout frame ranges overlap")
    same_window_teacher = config.static_target_mode == "same_window_a3_teacher"
    static_dataset = None
    if not same_window_teacher:
        static_dataset = PhaseADataset(_static_config(config), samples=max(config.steps, 32))

    model, ema = _load_a3_ema(Path(config.a3_checkpoint), device)
    teacher: nn.Module | None = None
    if config.static_target_mode in {"a3_teacher", "same_window_a3_teacher"}:
        teacher = copy.deepcopy(ema)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    trainable = set_trainable_scope(model, config.trainable_scope)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    before = _evaluate_dynamic(model, holdout_dynamic, device)
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, config.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            if same_window_teacher:
                dynamic_inputs, dynamic_targets, dynamic_masks = train_dynamic[(step - 1) % len(train_dynamic)]
                dynamic_input_device = dynamic_inputs[None].to(device)
                dynamic_target_device = dynamic_targets[None].to(device)
                dynamic_mask_device = dynamic_masks[None].to(device)
                dynamic_prediction = model(dynamic_input_device).clamp(0.0, 1.0)
                if teacher is None:
                    raise RuntimeError("same-window teacher mode requires a frozen A3 teacher")
                with torch.no_grad():
                    static_target_device = teacher(dynamic_input_device).clamp(0.0, 1.0)
                dynamic_loss, dynamic_metrics = dynamic_charbonnier_loss(
                    dynamic_prediction, dynamic_target_device, dynamic_mask_device
                )
                excluded_mask = dilate_motion_mask(
                    dynamic_mask_device, config.motion_boundary_radius
                )
                static_mask = (1.0 - excluded_mask).clamp(0.0, 1.0)
                static_loss, static_metrics = teacher_anchor_loss(
                    dynamic_prediction, static_target_device, static_mask
                )
                total_loss = (
                    config.static_anchor_weight * static_loss
                    + config.dynamic_weight * dynamic_loss
                )
                total_loss.backward()
                static_metrics["motion_boundary_exclusion_fraction"] = float(
                    excluded_mask.detach().mean().cpu()
                )
            else:
                if static_dataset is None:
                    raise RuntimeError("static dataset is required for the selected target mode")
                static_inputs, static_targets, static_masks = static_dataset[(step - 1) % len(static_dataset)]
                static_input_device = static_inputs[None].to(device)
                static_prediction = model(static_input_device).clamp(0.0, 1.0)
                if teacher is not None:
                    with torch.no_grad():
                        static_targets_device = teacher(static_input_device).clamp(0.0, 1.0)
                else:
                    static_targets_device = static_targets[None].to(device)
                static_loss, static_metrics = compute_loss(
                    static_prediction,
                    static_targets_device,
                    static_masks[None, :, None].to(device),
                    gradient_weight=0.1,
                    highpass_weight=0.1,
                )
                (config.static_anchor_weight * static_loss).backward()
                if config.dynamic_weight > 0.0:
                    dynamic_inputs, dynamic_targets, dynamic_masks = train_dynamic[(step - 1) % len(train_dynamic)]
                    dynamic_prediction = model(dynamic_inputs[None].to(device)).clamp(0.0, 1.0)
                    dynamic_loss, dynamic_metrics = dynamic_charbonnier_loss(
                        dynamic_prediction, dynamic_targets[None].to(device), dynamic_masks[None].to(device)
                    )
                    (config.dynamic_weight * dynamic_loss).backward()
                else:
                    dynamic_metrics = {"dynamic_loss": 0.0, "dynamic_mask_fraction": 0.0}
                    dynamic_loss = static_loss.detach().new_zeros(())
                total_loss = config.static_anchor_weight * static_loss + config.dynamic_weight * dynamic_loss
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema, model, decay=config.ema_decay)
            record = {
                "step": step,
                "total_objective": float(total_loss.detach().cpu()),
                "static": static_metrics,
                "dynamic": dynamic_metrics,
                "dynamic_weight": config.dynamic_weight,
                "static_anchor_weight": config.static_anchor_weight,
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 5 == 0:
                print(
                    f"[a5-motion] step={step} static={static_metrics['loss']:.6f} "
                    f"dynamic={dynamic_metrics['dynamic_loss']:.6f}",
                    flush=True,
                )

    after_model = _evaluate_dynamic(model, holdout_dynamic, device)
    after_ema = _evaluate_dynamic(ema, holdout_dynamic, device)
    checkpoint_path = output_dir / f"checkpoint_step_{config.steps:06d}.pth"
    torch.save({
        "step": config.steps,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "initialization": {"checkpoint": str(Path(config.a3_checkpoint).resolve()), "weights": "ema"},
    }, checkpoint_path)
    result: dict[str, object] = {
        "protocol": "A5 motion-supervision smoke",
        "checkpoint": str(checkpoint_path.resolve()),
        "dynamic_holdout_before": before,
        "dynamic_holdout_after_model": after_model,
        "dynamic_holdout_after_ema": after_ema,
        "training_frame_range": list(train_dynamic.allowed_frame_range),
        "holdout_frame_range": list(holdout_dynamic.allowed_frame_range),
        "trainable": {"scope": config.trainable_scope, **trainable},
        "static_target_mode": config.static_target_mode,
        "motion_boundary_radius": config.motion_boundary_radius,
        "dynamic_objective_improved": after_ema["loss"] < before["loss"],
        "promotable": False,
        "promotion_status": "requires motion-fidelity and native-regression evaluation",
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

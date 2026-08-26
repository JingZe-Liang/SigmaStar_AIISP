"""Minimal finite-checked data-first training loop."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .data_first_contracts import DataFirstInputBatch, DATA_FIRST_PROTOCOL
from .data_first_dataset import DataFirstDataset, DataFirstSample, DataFirstSampleRow
from .data_first_sampler import DeterministicCellSampler
from .data_first_checkpoint import load_data_first_checkpoint, save_data_first_checkpoint, write_data_first_manifest
from .model import FrequencyFusionConfigV2, FrequencyFusionCore
from .bands import b2, low_pass
from .data_first_fusion import limit_q_to_raw_range
from .selector import select_q
from .schemas.common import ContractError


@dataclass(frozen=True, slots=True)
class DataFirstTrainingBatch:
    model_inputs: DataFirstInputBatch
    target_classes: Tensor
    target_alpha: Tensor
    target_hf: Tensor
    target_hf_valid: Tensor
    target_pixel_valid: Tensor
    target_core_cells: Tensor
    samples: tuple[DataFirstSample, ...]


@dataclass(frozen=True, slots=True)
class DataFirstTrainingResult:
    output_dir: Path
    checkpoint: Path
    global_step: int
    loss_records: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class DataFirstLosses:
    total: Tensor
    alpha: Tensor
    alpha_reg: Tensor
    band: Tensor
    low: Tensor
    zero: Tensor
    range: Tensor
    smooth: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "loss_alpha": float(self.alpha.detach().cpu()),
            "loss_alpha_reg": float(self.alpha_reg.detach().cpu()),
            "loss_band": float(self.band.detach().cpu()),
            "loss_low": float(self.low.detach().cpu()),
            "loss_zero": float(self.zero.detach().cpu()),
            "loss_range": float(self.range.detach().cpu()),
            "loss_smooth": float(self.smooth.detach().cpu()),
        }


def build_data_first_batch(dataset: DataFirstDataset, rows: list[DataFirstSampleRow]) -> DataFirstTrainingBatch:
    samples = tuple(dataset.sample(row) for row in rows)
    if not samples:
        raise ContractError("data-first training batch cannot be empty")
    keys = ("prev_noisy", "curr_noisy", "denoised", "fused", "c_tilde")
    values = {name: torch.cat([getattr(sample.model_inputs, name) for sample in samples], dim=0) for name in keys}
    model_inputs = DataFirstInputBatch(**values)
    classes = []
    alphas = []
    hfs = []
    hf_valids = []
    pixel_valids = []
    core_cells = []
    for sample in samples:
        row = sample.row
        class_grid = sample.supervision.policy_alpha_class[0]
        alpha_grid = sample.supervision.policy_alpha_target[0]
        y = min(row.cell_y, class_grid.shape[0] - 1)
        x = min(row.cell_x, class_grid.shape[1] - 1)
        classes.append(int(class_grid[y, x]))
        alphas.append(float(alpha_grid[y, x]))
        origin_y = min(max(row.cell_y * 32 - 64, 0), 540 - 320)
        origin_x = min(max(row.cell_x * 32 - 64, 0), 960 - 320)
        core_cells.append(((row.cell_y * 32 - (origin_y + 32)) // 32, (row.cell_x * 32 - (origin_x + 32)) // 32))
        hfs.append(sample.supervision.hf_target[:, origin_y + 32 : origin_y + 288, origin_x + 32 : origin_x + 288])
        hf_valids.append(sample.supervision.hf_valid[:, origin_y + 32 : origin_y + 288, origin_x + 32 : origin_x + 288])
        pixel_valids.append(sample.supervision.valid_bits[:, origin_y + 32 : origin_y + 288, origin_x + 32 : origin_x + 288])
    return DataFirstTrainingBatch(
        model_inputs=model_inputs,
        target_classes=torch.tensor(classes, dtype=torch.long),
        target_alpha=torch.tensor(alphas, dtype=torch.float32),
        target_hf=torch.from_numpy(__import__("numpy").stack(hfs)).float(),
        target_hf_valid=torch.from_numpy(__import__("numpy").stack(hf_valids)).bool(),
        target_pixel_valid=torch.from_numpy(__import__("numpy").stack(pixel_valids)).bool(),
        target_core_cells=torch.tensor(core_cells, dtype=torch.long),
        samples=samples,
    )


def _finite_parameters(model: FrequencyFusionCore) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError(f"data-first parameter {name} is non-finite")


def data_first_train_step(batch: DataFirstTrainingBatch, model: FrequencyFusionCore, optimizer: torch.optim.Optimizer) -> DataFirstLosses:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_data_first(batch.model_inputs).q_logits_pixel_core
    valid = torch.ones((output.shape[0], 1, *output.shape[-2:]), device=output.device, dtype=torch.bool)
    selected = select_q(output, valid)
    targets = batch.target_classes.to(device=output.device)
    target_alpha = batch.target_alpha.to(device=output.device)
    target_cells = batch.target_core_cells.to(device=output.device)
    indices = torch.arange(output.shape[0], device=output.device)
    cell_logits = selected.q_logits_cell[indices, :, target_cells[:, 0], target_cells[:, 1]]
    loss_alpha = F.cross_entropy(cell_logits, targets)
    loss_alpha_reg = F.smooth_l1_loss(selected.q_cell[indices, 0, target_cells[:, 0], target_cells[:, 1]], target_alpha)
    denoised = batch.model_inputs.denoised[..., 32:-32, 32:-32]
    fused = batch.model_inputs.fused[..., 32:-32, 32:-32]
    delta = b2(fused - denoised)
    prediction = denoised + limit_q_to_raw_range(denoised, delta, selected.q) * delta
    hf = batch.target_hf.to(device=output.device)
    hf_valid = batch.target_hf_valid.to(device=output.device)
    band_error = (b2(prediction) - hf).abs()
    loss_band = band_error.masked_select(hf_valid.expand_as(band_error)).mean() if bool(hf_valid.any()) else band_error.sum() * 0.0
    loss_low = (low_pass(prediction) - low_pass(denoised)).abs().mean()
    loss_zero = selected.q.square().mean()
    loss_range = (F.relu(-prediction).square() + F.relu(prediction - 1.0).square()).mean()
    loss_smooth = (selected.q[..., 1:] - selected.q[..., :-1]).abs().mean() + (selected.q[..., 1:, :] - selected.q[..., :-1, :]).abs().mean()
    total = loss_alpha + 0.5 * loss_alpha_reg + 0.5 * loss_band + 0.1 * loss_low + 0.05 * loss_zero + 0.05 * loss_range + 0.05 * loss_smooth
    if not bool(torch.isfinite(total.detach())):
        raise FloatingPointError("data-first loss is non-finite")
    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach()).all()):
            raise FloatingPointError(f"data-first gradient {name} is non-finite")
    optimizer.step()
    _finite_parameters(model)
    return DataFirstLosses(total.detach(), loss_alpha.detach(), loss_alpha_reg.detach(), loss_band.detach(), loss_low.detach(), loss_zero.detach(), loss_range.detach(), loss_smooth.detach())


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(value: object, *, device: torch.device) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("torch_cpu"), Tensor):
        raise ContractError("data-first checkpoint has no valid RNG state")
    torch.set_rng_state(value["torch_cpu"])
    cuda_state = value.get("torch_cuda")
    if device.type == "cuda" and isinstance(cuda_state, (list, tuple)) and all(isinstance(item, Tensor) for item in cuda_state):
        torch.cuda.set_rng_state_all(list(cuda_state))


def _resume_training(
    path: Path,
    *,
    model: FrequencyFusionCore,
    optimizer: torch.optim.Optimizer,
    schedule_digest: str,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> int:
    checkpoint = load_data_first_checkpoint(path, map_location=device)
    payload = checkpoint.payload
    provenance = payload.get("provenance")
    state = payload.get("training_state")
    if not isinstance(provenance, Mapping) or provenance.get("schedule_digest") != schedule_digest:
        raise ContractError("resume checkpoint schedule does not match this training run")
    if not isinstance(state, Mapping):
        raise ContractError("resume checkpoint has no training state")
    global_step = state.get("global_step")
    if not isinstance(global_step, int) or global_step < 0:
        raise ContractError("resume checkpoint has invalid global_step")
    if state.get("seed") != seed or state.get("batch_size") != batch_size:
        raise ContractError("resume checkpoint seed or batch size does not match")
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ContractError("resume checkpoint has no model or optimizer state")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    _restore_rng_state(state.get("rng_state"), device=device)
    return global_step


def _write_metric(streams: tuple[object, ...], record: Mapping[str, object]) -> None:
    line = json.dumps(dict(record), sort_keys=True, ensure_ascii=True) + "\n"
    for stream in streams:
        stream.write(line)
        stream.flush()


def run_data_first_training(
    dataset: DataFirstDataset,
    output_dir: Path,
    *,
    device: str = "auto",
    seed: int = 20260826,
    batch_size: int = 2,
    max_steps: int = 2,
    log_interval: int = 1,
    checkpoint_interval: int = 25,
    resume: Path | None = None,
) -> DataFirstTrainingResult:
    if batch_size <= 0 or batch_size % 2:
        raise ContractError("data-first batch size must be positive and even")
    if max_steps <= 0:
        raise ContractError("data-first max_steps must be positive")
    if log_interval <= 0 or checkpoint_interval <= 0:
        raise ContractError("data-first log and checkpoint intervals must be positive")
    target_device = torch.device("cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ContractError("data-first CUDA requested but unavailable")
    torch.manual_seed(int(seed))
    model = FrequencyFusionCore(FrequencyFusionConfigV2.production()).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = DeterministicCellSampler.build(dataset, seed=seed, batch_size=batch_size)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if resume is not None:
        start_step = _resume_training(Path(resume), model=model, optimizer=optimizer, schedule_digest=schedule.digest, seed=seed, batch_size=batch_size, device=target_device)
    if start_step >= max_steps:
        raise ContractError("resume checkpoint has already reached max_steps")
    records: list[dict[str, object]] = []
    started_at = time.monotonic()
    metrics_mode = "a" if resume is not None else "w"
    with (destination / "metrics.jsonl").open(metrics_mode, encoding="ascii") as metrics_stream, (destination / "train.jsonl").open(metrics_mode, encoding="ascii") as legacy_stream:
      for step in range(start_step, max_steps):
        start = (step * batch_size) % len(schedule.rows)
        rows = list(schedule.rows[start : start + batch_size])
        if len(rows) != batch_size:
            rows = list(schedule.rows[:batch_size])
        batch = build_data_first_batch(dataset, rows)
        batch = DataFirstTrainingBatch(
            model_inputs=DataFirstInputBatch(**{name: getattr(batch.model_inputs, name).to(target_device) for name in batch.model_inputs.as_mapping()}),
            target_classes=batch.target_classes.to(target_device),
            target_alpha=batch.target_alpha.to(target_device),
            target_hf=batch.target_hf.to(target_device),
            target_hf_valid=batch.target_hf_valid.to(target_device),
            target_pixel_valid=batch.target_pixel_valid.to(target_device),
            target_core_cells=batch.target_core_cells.to(target_device),
            samples=batch.samples,
        )
        losses = data_first_train_step(batch, model, optimizer)
        global_step = step + 1
        elapsed_seconds = time.monotonic() - started_at
        completed = global_step - start_step
        steps_per_second = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        record: dict[str, object] = {
            "global_step": global_step,
            "protocol": DATA_FIRST_PROTOCOL,
            **losses.scalars(),
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": steps_per_second,
            "eta_seconds": (max_steps - global_step) / steps_per_second if steps_per_second > 0.0 else None,
        }
        if global_step % log_interval == 0 or global_step == max_steps:
            records.append(record)
            _write_metric((metrics_stream, legacy_stream), record)
        if global_step % checkpoint_interval == 0 or global_step == max_steps:
            state = {"global_step": global_step, "seed": seed, "batch_size": batch_size, "rng_state": _capture_rng_state()}
            provenance = {"schedule_digest": schedule.digest, "device": str(target_device)}
            save_data_first_checkpoint(destination / f"checkpoint_step_{global_step:06d}.pt", model, optimizer, state, provenance)
    checkpoint = destination / "data_first_v2.pt"
    final_state = {"global_step": max_steps, "seed": seed, "batch_size": batch_size, "rng_state": _capture_rng_state()}
    ref = save_data_first_checkpoint(checkpoint, model, optimizer, final_state, {"schedule_digest": schedule.digest, "device": str(target_device)})
    write_data_first_manifest(destination, ref, {"global_step": max_steps, "schedule_digest": schedule.digest, "md_used_for_supervision": True, "md_used_as_model_input": False, "formal_v2_compatible": False, "resumed_from": str(resume) if resume is not None else None})
    return DataFirstTrainingResult(destination, checkpoint, max_steps, tuple(records))


__all__ = ["DataFirstLosses", "DataFirstTrainingBatch", "DataFirstTrainingResult", "build_data_first_batch", "data_first_train_step", "run_data_first_training"]

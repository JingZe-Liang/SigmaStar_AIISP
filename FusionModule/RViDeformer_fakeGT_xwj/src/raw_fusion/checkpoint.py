"""Strict, crash-safe checkpoint persistence for fusion training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "model",
        "optimizer",
        "scaler",
        "epoch",
        "global_step",
        "config_fingerprint",
        "experiment_config",
        "seed",
    }
)


@dataclass(frozen=True, slots=True)
class TrainingState:
    """The metadata and state dictionaries restored from a checkpoint."""

    schema_version: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scaler: dict[str, Any] | None
    epoch: int
    global_step: int
    config_fingerprint: str
    experiment_config: dict[str, Any]
    seed: int


def save_checkpoint_atomic(path: Path, state: dict[str, object]) -> None:
    """Persist a validated checkpoint without exposing a partial destination."""
    _validate_state(state)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(state, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def load_checkpoint_strict(
    path: Path,
    *,
    expected_fingerprint: str,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
) -> TrainingState:
    """Load and validate a checkpoint before mutating optional consumers."""
    source = Path(path)
    try:
        raw_state = torch.load(source, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"无法读取检查点: {source}") from error
    if not isinstance(raw_state, dict):
        raise ValueError("检查点必须是字典")
    _validate_state(raw_state)
    if raw_state["config_fingerprint"] != expected_fingerprint:
        raise ValueError(
            "检查点配置指纹不匹配: "
            f"期望 {expected_fingerprint}，实际 {raw_state['config_fingerprint']}"
        )
    state = _to_training_state(raw_state)
    if scaler is not None and state.scaler is None:
        raise ValueError("检查点没有 GradScaler 状态")
    _load_consumers_transactionally(state, model, optimizer, scaler)
    return state


def _load_consumers_transactionally(
    state: TrainingState,
    model: nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
) -> None:
    model_before = None if model is None else _clone_to_cpu(model.state_dict())
    optimizer_before = None if optimizer is None else _clone_to_cpu(optimizer.state_dict())
    scaler_before = None if scaler is None else _clone_to_cpu(scaler.state_dict())
    phase = "模型"
    try:
        if model is not None:
            model.load_state_dict(state.model, strict=True)
        phase = "优化器"
        if optimizer is not None:
            optimizer.load_state_dict(state.optimizer)
        phase = "GradScaler"
        if scaler is not None:
            scaler.load_state_dict(state.scaler)
    except Exception as error:
        try:
            if model is not None:
                model.load_state_dict(model_before, strict=True)
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_before)
            if scaler is not None:
                scaler.load_state_dict(scaler_before)
        except Exception as rollback_error:
            raise RuntimeError("检查点恢复失败且无法回滚调用方状态") from rollback_error
        raise ValueError(f"检查点{phase} state_dict 不匹配") from error


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        cloned = value.__class__((key, _clone_to_cpu(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            cloned._metadata = copy.deepcopy(value._metadata)
        return cloned
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _validate_state(state: dict[str, Any]) -> None:
    actual = set(state)
    if actual != set(CHECKPOINT_KEYS):
        missing = sorted(set(CHECKPOINT_KEYS) - actual)
        unknown = sorted(actual - set(CHECKPOINT_KEYS))
        details: list[str] = []
        if missing:
            details.append(f"缺少 {','.join(missing)}")
        if unknown:
            details.append(f"未知 {','.join(unknown)}")
        raise ValueError(f"检查点字段不符合固定 schema: {'; '.join(details)}")
    if (
        not isinstance(state["schema_version"], int)
        or isinstance(state["schema_version"], bool)
        or state["schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("检查点 schema_version 不支持")
    if not isinstance(state["model"], dict):
        raise ValueError("检查点 model 必须是字典")
    if not isinstance(state["optimizer"], dict):
        raise ValueError("检查点 optimizer 必须是字典")
    if state["scaler"] is not None and not isinstance(state["scaler"], dict):
        raise ValueError("检查点 scaler 必须是字典或 None")
    for name in ("epoch", "global_step", "seed"):
        if not isinstance(state[name], int) or isinstance(state[name], bool):
            raise ValueError(f"检查点 {name} 必须是整数")
        if name != "seed" and state[name] < 0:
            raise ValueError(f"检查点 {name} 必须为非负整数")
    if not isinstance(state["config_fingerprint"], str) or not state["config_fingerprint"]:
        raise ValueError("检查点 config_fingerprint 必须是非空字符串")
    if not isinstance(state["experiment_config"], dict):
        raise ValueError("检查点 experiment_config 必须是字典")


def _to_training_state(state: dict[str, Any]) -> TrainingState:
    return TrainingState(
        schema_version=int(state["schema_version"]),
        model=dict(state["model"]),
        optimizer=dict(state["optimizer"]),
        scaler=None if state["scaler"] is None else dict(state["scaler"]),
        epoch=int(state["epoch"]),
        global_step=int(state["global_step"]),
        config_fingerprint=str(state["config_fingerprint"]),
        experiment_config=dict(state["experiment_config"]),
        seed=int(state["seed"]),
    )


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability on POSIX filesystems."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from raw_fusion.config import LossConfig, ModelConfig
from raw_fusion.losses import FusionLoss
from raw_fusion.model import CausalRawFusionNet
from raw_fusion.train import (
    TRAIN_IMAGE_KEYS,
    _center_even_start,
    _resume_log_best,
    run_training,
    train_step,
)


def make_cpu_training_case() -> tuple[
    CausalRawFusionNet,
    torch.optim.Optimizer,
    FusionLoss,
    dict[str, torch.Tensor],
]:
    model = CausalRawFusionNet(ModelConfig((8, 16, 24), 0.03, True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = FusionLoss(
        LossConfig(0.05, 0.02, 0.01, 0.01, 0.001, 0.02, 0.005, 4),
        white_level=4095,
        target_black_level=252,
    )
    generator = torch.Generator().manual_seed(0)
    batch = {
        key: torch.rand(1, 4, 16, 16, generator=generator)
        for key in TRAIN_IMAGE_KEYS
    }
    return model, optimizer, loss_fn, batch


def test_train_step_updates_parameters_and_returns_finite_losses() -> None:
    model, optimizer, loss_fn, batch = make_cpu_training_case()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    metrics = train_step(
        model,
        batch,
        loss_fn,
        optimizer,
        scaler=None,
        device=torch.device("cpu"),
        amp=False,
    )

    assert set(metrics) == {
        "total",
        "reconstruction",
        "gradient",
        "gate",
        "residual",
        "range",
        "gradient_norm",
    }
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )


def test_train_step_rejects_missing_image_key() -> None:
    model, optimizer, loss_fn, batch = make_cpu_training_case()
    del batch["target"]
    with pytest.raises(KeyError, match="target"):
        train_step(model, batch, loss_fn, optimizer, None, torch.device("cpu"), False)


def test_train_step_rejects_nonfinite_loss_before_optimizer_step() -> None:
    model, optimizer, loss_fn, batch = make_cpu_training_case()
    batch["curr_noisy"].fill_(math.inf)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    with pytest.raises(FloatingPointError, match="loss"):
        train_step(model, batch, loss_fn, optimizer, None, torch.device("cpu"), False)

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_train_step_rejects_non_tensor_image() -> None:
    model, optimizer, loss_fn, batch = make_cpu_training_case()
    invalid: dict[str, object] = dict(batch)
    invalid["target"] = "not a tensor"
    with pytest.raises(TypeError, match="target"):
        train_step(model, invalid, loss_fn, optimizer, None, torch.device("cpu"), False)


def test_center_crop_start_preserves_even_cfa_phase() -> None:
    assert _center_even_start(length=18, crop=16) == 0
    assert _center_even_start(length=20, crop=16) == 2


def test_run_training_exposes_optional_resume_path() -> None:
    parameter = inspect.signature(run_training).parameters["resume_path"]
    assert parameter.default is None


def test_resume_log_must_end_at_checkpoint_progress(tmp_path: Path) -> None:
    log_path = tmp_path / "train.jsonl"
    records = [
        {"epoch": 0, "global_step": 4, "validation": {"total": 0.5}},
        {"epoch": 1, "global_step": 8, "validation": {"total": 0.4}},
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    assert _resume_log_best(log_path, checkpoint_epoch=1, checkpoint_step=8) == 0.4
    with pytest.raises(ValueError, match="日志末尾.*checkpoint"):
        _resume_log_best(log_path, checkpoint_epoch=0, checkpoint_step=4)

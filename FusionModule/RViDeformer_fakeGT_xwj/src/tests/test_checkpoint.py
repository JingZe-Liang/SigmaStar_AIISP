from __future__ import annotations

from pathlib import Path
import copy

import pytest
import torch

from raw_fusion.checkpoint import (
    CHECKPOINT_KEYS,
    TrainingState,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)


def make_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "schema_version": 1,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {}, "param_groups": []},
        "scaler": None,
        "epoch": 2,
        "global_step": 7,
        "config_fingerprint": "abc",
        "experiment_config": {"model": {"channels": [8, 16, 24]}},
        "seed": 123,
    }
    state.update(overrides)
    return state


def test_atomic_save_does_not_leave_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(path, make_state())
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_rejects_fingerprint_mismatch_before_model_mutation(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(path, make_state(model={"weight": torch.tensor([[9.0]])}))

    with pytest.raises(ValueError, match="配置指纹"):
        load_checkpoint_strict(
            path,
            expected_fingerprint="def",
            model=model,
            device=torch.device("cpu"),
        )

    torch.testing.assert_close(model.weight, before)


def test_checkpoint_loads_strict_model_and_returns_training_state(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    destination = torch.nn.Linear(2, 1)
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(path, make_state(model=source.state_dict()))

    result = load_checkpoint_strict(
        path,
        expected_fingerprint="abc",
        model=destination,
        device="cpu",
    )

    assert isinstance(result, TrainingState)
    assert result.epoch == 2
    assert result.global_step == 7
    for name, value in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], value)


def test_checkpoint_rejects_unknown_or_missing_schema_keys(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    invalid = make_state(extra="unexpected")
    with pytest.raises(ValueError, match="检查点字段"):
        save_checkpoint_atomic(path, invalid)
    assert set(make_state()) == CHECKPOINT_KEYS


def test_checkpoint_rejects_negative_progress_counter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epoch.*非负"):
        save_checkpoint_atomic(tmp_path / "model.pt", make_state(epoch=-1))


def test_corrupt_checkpoint_is_reported_as_value_error(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="无法读取检查点"):
        load_checkpoint_strict(path, expected_fingerprint="abc")


def test_failed_atomic_save_preserves_existing_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import raw_fusion.checkpoint as checkpoint_module

    path = tmp_path / "model.pt"
    path.write_bytes(b"existing")

    def fail_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="synthetic"):
        save_checkpoint_atomic(path, make_state())

    assert path.read_bytes() == b"existing"
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_optimizer_restore_rolls_back_model_and_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    source = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(9.0)
        source.bias.fill_(8.0)
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(
        path,
        make_state(model=source.state_dict(), optimizer={"state": {}, "param_groups": []}),
    )

    with pytest.raises(ValueError, match="优化器"):
        load_checkpoint_strict(
            path,
            expected_fingerprint="abc",
            model=model,
            optimizer=optimizer,
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_model[name])
    assert optimizer.state_dict() == before_optimizer


def test_failed_partial_model_restore_rolls_back_model(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(path, make_state(model={"weight": torch.full((1, 2), 9.0)}))

    with pytest.raises(ValueError, match="模型"):
        load_checkpoint_strict(path, expected_fingerprint="abc", model=model)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])

"""A12 preparation, checkpoint integrity, and RNG resume tests without training."""

from dataclasses import asdict, replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from raw_fusion.finetune import a12_train
from raw_fusion.finetune.a12_train import (
    A12TrainConfig,
    capture_rng_state,
    restore_rng_state,
    save_a12_checkpoint,
    sha256_file,
    validate_a12_config,
    validate_resume_payload,
)


@pytest.fixture
def pilot_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> A12TrainConfig:
    monkeypatch.setattr(a12_train, "REPO_ROOT", tmp_path)
    return A12TrainConfig("unused", "unused", "unused", str(tmp_path / "artifacts/a12_test"), "unused")


def test_rng_roundtrip_preserves_stochastic_forward(tmp_path: Path) -> None:
    random.seed(420)
    np.random.seed(421)
    torch.manual_seed(422)
    state = capture_rng_state()
    path = tmp_path / "rng.pth"
    torch.save(state, path)
    expected = (random.random(), np.random.normal(size=12), torch.nn.functional.dropout(torch.ones(128), 0.5, training=True))
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    restore_rng_state(torch.load(path, weights_only=True))
    actual = (random.random(), np.random.normal(size=12), torch.nn.functional.dropout(torch.ones(128), 0.5, training=True))
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


def test_checkpoint_stores_resume_state_without_overwrite(pilot_config: A12TrainConfig, tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    ema = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=pilot_config.learning_rate)
    path = tmp_path / "checkpoint_step_000050.pth"
    save_a12_checkpoint(path, model, ema, optimizer, 50, pilot_config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    resumed_config = replace(pilot_config, output_dir=str(tmp_path / "artifacts/a12_resume"),
                             resume_checkpoint=str(path), steps=300)
    assert validate_resume_payload(payload, resumed_config) == 50
    assert payload["next_sample_index"] == 50
    assert payload["config"] == asdict(pilot_config)
    with pytest.raises(FileExistsError):
        save_a12_checkpoint(path, model, ema, optimizer, 50, pilot_config)
    with pytest.raises(ValueError, match="configuration"):
        validate_resume_payload(payload, replace(resumed_config, seed=99))
    with pytest.raises(ValueError, match="configuration"):
        validate_resume_payload(payload, replace(resumed_config, samples_per_epoch=256))
    del payload["rng"]
    with pytest.raises(ValueError, match="rng"):
        validate_resume_payload(payload, resumed_config)


def test_config_prevents_baseline_overwrite_and_changed_objective(pilot_config: A12TrainConfig, tmp_path: Path) -> None:
    assert validate_a12_config(pilot_config) == Path(pilot_config.output_dir)
    with pytest.raises(ValueError, match="a12_"):
        validate_a12_config(replace(pilot_config, output_dir=str(tmp_path / "artifacts/finetune_mis20s1/training")))
    with pytest.raises(ValueError, match="structure weights"):
        validate_a12_config(replace(pilot_config, gradient_weight=0.2))
    with pytest.raises(ValueError, match="holdout"):
        validate_a12_config(replace(pilot_config, holdout_gain="645x"))
    output = Path(pilot_config.output_dir)
    output.mkdir(parents=True)
    (output / "train.jsonl").write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validate_a12_config(pilot_config)


def test_hash_and_missing_cuda_rng_fail_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    state = capture_rng_state()
    state["torch_cuda"] = []
    with pytest.raises(ValueError, match="CUDA RNG"):
        restore_rng_state(state, require_cuda=True)

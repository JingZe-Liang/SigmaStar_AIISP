"""A14 sampling invariants, immutable cache checks, and fail-stop OOM behavior."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from raw_fusion.finetune import a14_static_train as a14
from raw_fusion.finetune.a12_sampling import A12SamplePlan


class FakeStaticDataset:
    def __init__(self, invalid: set[int] | None = None, *, wrong_kind: bool = False) -> None:
        self.invalid = invalid or set()
        self.config = SimpleNamespace(seed=19)
        self.wrong_kind = wrong_kind
        self.accessed: list[int] = []

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.accessed.append(index)
        inputs = torch.full((8, 4, 64, 64), index / 1000.0)
        targets = torch.full_like(inputs, 0.25)
        masks = torch.ones((8, 64, 64))
        if index in self.invalid:
            masks[:, 8:, :] = 0.0
        return inputs, targets, masks

    def sample_plan(self, index: int) -> A12SamplePlan:
        source = "synthetic" if index % 2 or self.wrong_kind else "real"
        return A12SamplePlan(index, 0, index, source, 0, "128x", 0, 0,
                             0 if source == "real" else None, 1, None if source == "real" else 2)


@pytest.fixture(autouse=True)
def limited_threads() -> object:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config() -> dict:
    sources = []
    for sensor, frames, gain in (("SC465SL", 300, "256x"), ("sc635hai", 150, "131x")):
        sources.append({"name": sensor, "reference_dir": "/refs", "source_sha256": "a" * 64,
                        "source": {"name": f"{sensor}_{gain}", "sensor": sensor,
                                   "path": f"/dataset/{sensor}.raw", "width": 256, "height": 256,
                                   "frames_used": frames, "bit_shift": 0, "cfa": "RGGB",
                                   "nominal_black_dn": 256.0, "reference_mode": "motion_aware_split_half",
                                   "fixed_isp_wb": [1, 1, 1, 1], "fixed_isp_exposure": 1}})
    return {"schema": a14.SCHEMA, "a3_checkpoint_sha256": a14.A3_STEP5000_SHA256,
            "a3_checkpoint": "/initial.pth", "expected_gpu_uuid": a14.EXPECTED_GPU_UUID,
            "dataset_root": "/dataset", "mis_supervision_dir": "/mis", "mis_noise_dir": "/noise",
            "precision": "bfloat16", "steps": 12, "checkpoint_interval": 6,
            "seed": 12, "max_candidates_per_slot": 4, "learning_rate": 1e-6,
            "gradient_clip_norm": 1.0, "min_valid_fraction": 0.25, "ema_decay": 0.99,
            "gradient_weight": 0.1, "highpass_weight": 0.1, "static_sources": sources,
            "prepared_dir": str(a14.REPO_ROOT / "artifacts/a14_test_prepared"),
            "output_dir": str(a14.REPO_ROOT / "artifacts/a14_test_training")}


def make_audit(config: dict, plan: list[dict]) -> dict:
    return {"schema": a14.SCHEMA, "status": "prepared", "config": config,
            "config_sha256": a14.canonical_sha256(config), "plan": plan,
            "plan_sha256": a14.canonical_sha256(plan), "source_inventory": [],
            "source_inventory_sha256": a14.canonical_sha256([])}


def test_replacements_preserve_source_balance_and_both_parities(tmp_path: Path, config: dict) -> None:
    datasets = [FakeStaticDataset({0, 3}), FakeStaticDataset({1}), FakeStaticDataset()]
    plan, rejected = a14.freeze_balanced_samples(datasets, ["mis", "sc465", "sc635"], 12, 0.25, 4, tmp_path / "samples")
    assert [row["source_index"] for row in plan] == [0, 1, 2] * 4
    assert [row["kind"] for row in plan if row["source_index"] == 0] == ["real", "synthetic"] * 2
    assert [row["input_parity"] for row in plan if row["source_index"] == 1] == [0, 1, 0, 1]
    assert [row["candidate_index"] for row in plan if row["source_index"] == 0] == [2, 1, 4, 5]
    assert len(rejected) == 3
    assert rejected[0]["gain"] == "128x"
    assert all(row["valid_fraction"] >= 0.25 for row in plan)
    assert len({(row["source_index"], row["candidate_index"]) for row in plan}) == 12
    a14.validate_prepared_audit(make_audit(config, plan), config)
    for row in plan:
        assert a14.validate_sample(a14.load_frozen_sample(tmp_path, row, 0.25)) == 1.0


def test_same_seed_and_sources_freeze_identical_tensors(tmp_path: Path) -> None:
    plans = []
    for name in ("first", "second"):
        datasets = [FakeStaticDataset({0}), FakeStaticDataset(), FakeStaticDataset()]
        plan, _ = a14.freeze_balanced_samples(datasets, ["mis", "sc465", "sc635"], 6, 0.25, 4, tmp_path / name / "samples")
        plans.append(plan)
    assert plans[0] == plans[1]


def test_bounded_search_fails_without_relaxing_mask(tmp_path: Path) -> None:
    datasets = [FakeStaticDataset({0, 2, 4}), FakeStaticDataset(), FakeStaticDataset()]
    with pytest.raises(RuntimeError, match="within budget"):
        a14.freeze_balanced_samples(datasets, ["mis", "sc465", "sc635"], 6, 0.25, 3, tmp_path / "samples")
    assert datasets[0].accessed == [0, 2, 4]
    assert not datasets[1].accessed


def test_wrong_mis_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real/synthetic balance"):
        a14.freeze_balanced_samples([FakeStaticDataset(wrong_kind=True), FakeStaticDataset(), FakeStaticDataset()],
                                    ["mis", "sc465", "sc635"], 6, 0.25, 3, tmp_path / "samples")


@pytest.mark.parametrize("change", ["sample_file", "tensor_sha256", "valid_fraction"])
def test_modified_cache_is_rejected(tmp_path: Path, change: str) -> None:
    plan, _ = a14.freeze_balanced_samples([FakeStaticDataset()] * 3, ["mis", "sc465", "sc635"],
                                         6, 0.25, 3, tmp_path / "samples")
    row = deepcopy(plan[0])
    if change == "sample_file":
        path = tmp_path / row["sample_file"]
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif change == "tensor_sha256":
        row["tensor_sha256"]["inputs"] = "0" * 64
    else:
        row["valid_fraction"] = 0.5
    with pytest.raises(ValueError, match="SHA mismatch|coverage changed"):
        a14.load_frozen_sample(tmp_path, row, 0.25)


def test_sample_validation_rejects_nonfinite_and_nonbinary() -> None:
    inputs, targets, masks = FakeStaticDataset()[0]
    inputs[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        a14.validate_sample((inputs, targets, masks))
    inputs[0, 0, 0, 0] = 0.0
    masks[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        a14.validate_sample((inputs, targets, masks))


@pytest.mark.parametrize("key,value", [("steps", 7), ("precision", "float32"), ("ema_decay", 1.0),
                                      ("learning_rate", float("nan")), ("expected_gpu_uuid", "wrong"),
                                      ("gradient_weight", 0.2)])
def test_invalid_frozen_protocol_is_rejected(config: dict, key: str, value: object) -> None:
    a14.validate_training_config(config)
    config[key] = value
    with pytest.raises(ValueError):
        a14.validate_training_config(config)


def test_heldout_gain_cannot_replace_static_training_source(config: dict) -> None:
    config["static_sources"][0]["source"]["name"] = "SC465SL_2000x"
    with pytest.raises(ValueError, match="gain/frame identity"):
        a14.validate_training_config(config)


def test_prepared_plan_rejects_target_input_overlap(tmp_path: Path, config: dict) -> None:
    plan, _ = a14.freeze_balanced_samples([FakeStaticDataset()] * 3, ["mis", "sc465", "sc635"],
                                         12, 0.25, 3, tmp_path / "samples")
    plan[1]["target_reference_parity"] = plan[1]["input_parity"]
    with pytest.raises(ValueError, match="parities overlap"):
        a14.validate_prepared_audit(make_audit(config, plan), config)


@pytest.mark.parametrize("use_override", [False, True])
def test_oom_stops_once_and_records_status(tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch, use_override: bool) -> None:
    config = deepcopy(config)
    config.update({"prepared_dir": str(tmp_path / "prepared"), "output_dir": str(tmp_path / "training"),
                   "a3_checkpoint": str(tmp_path / "initial.pth")})
    Path(config["a3_checkpoint"]).write_bytes(b"initial")
    config["a3_checkpoint_sha256"] = a14.sha256_file(Path(config["a3_checkpoint"]))
    frozen_config = deepcopy(config)
    runtime_output = tmp_path / "training_attempt02" if use_override else Path(config["output_dir"])
    prepared = Path(config["prepared_dir"])
    prepared.mkdir()
    (prepared / "audit.json").write_text(json.dumps({"plan": []}), encoding="utf-8")
    monkeypatch.setattr(a14, "validate_training_config", lambda _: None)
    monkeypatch.setattr(a14, "validate_prepared_audit", lambda *_: None)
    monkeypatch.setattr(a14, "set_seed", lambda _: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", a14.EXPECTED_GPU_UUID)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=a14.EXPECTED_GPU_UUID.removeprefix("GPU-"), name="RTX 3090"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    calls = []

    def fail_allocation() -> object:
        calls.append("allocation")
        raise torch.cuda.OutOfMemoryError("deliberate test allocation failure")

    monkeypatch.setattr(a14, "build_rvideformer", fail_allocation)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        a14.train_static_model(config, output_dir=runtime_output if use_override else None)
    status = json.loads((runtime_output / "status.json").read_text())
    assert calls == ["allocation"]
    assert status["status"] == "stopped_oom"
    assert status["completed_steps"] == 0
    assert status["gpu_uuid"] == a14.EXPECTED_GPU_UUID
    assert status["runtime_output_dir"] == str(runtime_output)
    assert config == frozen_config == status["config"]
    if use_override:
        assert not Path(config["output_dir"]).exists()
    assert "no retry" in status["oom_policy"]


@pytest.mark.parametrize("location", ["outside", "prepared", "prepared_child", "prepared_parent"])
def test_runtime_output_override_rejects_unsafe_locations(
    tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch, location: str,
) -> None:
    monkeypatch.setattr(a14, "REPO_ROOT", tmp_path)
    config["prepared_dir"] = str(tmp_path / "artifacts/a14_run/prepared")
    config["output_dir"] = str(tmp_path / "artifacts/a14_run/training")
    prepared = Path(config["prepared_dir"])
    paths = {"outside": tmp_path / "elsewhere", "prepared": prepared,
             "prepared_child": prepared / "attempt02", "prepared_parent": prepared.parent}
    frozen_config = deepcopy(config)
    with pytest.raises(ValueError, match="A14 writes only|distinct non-nested"):
        a14.train_static_model(config, output_dir=paths[location])
    assert config == frozen_config


def test_runtime_override_validates_original_frozen_config(
    tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a14, "REPO_ROOT", tmp_path)
    config["prepared_dir"] = str(tmp_path / "artifacts/a14_run/prepared")
    config["output_dir"] = str(tmp_path / "artifacts/a14_run/training")
    prepared = Path(config["prepared_dir"])
    plan, _ = a14.freeze_balanced_samples([FakeStaticDataset()] * 3, ["mis", "sc465", "sc635"],
                                         config["steps"], 0.25, 3, prepared / "samples")
    audit = make_audit(config, plan)
    audit_path = prepared / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    original_bytes = audit_path.read_bytes()
    frozen_config = deepcopy(config)
    monkeypatch.setattr(a14, "sha256_file", lambda _: config["a3_checkpoint_sha256"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", a14.EXPECTED_GPU_UUID)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime_output = prepared.parent / "training_attempt02"
    with pytest.raises(RuntimeError, match="exactly one isolated CUDA GPU"):
        a14.train_static_model(config, output_dir=runtime_output)
    assert config == frozen_config
    assert audit_path.read_bytes() == original_bytes
    assert not runtime_output.exists()

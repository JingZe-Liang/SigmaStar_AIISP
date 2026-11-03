"""Paired ancestry, data integrity, and optimizer/RNG restoration tests."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from raw_fusion.finetune import a12_paired_train as paired
from raw_fusion.finetune.a3 import A3Config
from raw_fusion.finetune.a12_sampling import A12SamplePlan
from raw_fusion.finetune.a12_train import capture_rng_state, sha256_file


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> paired.A12PairedTrainConfig:
    monkeypatch.setattr(paired, "REPO_ROOT", tmp_path)
    return paired.A12PairedTrainConfig(
        "data", "supervision", "noise", str(tmp_path / "artifacts/a12_shared"), "a3.pth",
        device="cuda:0", audit_path=str(tmp_path / "audit.json"), expected_audit_sha256="audit-hash",
    )


def resume_payload(config: paired.A12PairedTrainConfig, step: int = 512) -> dict:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return {
        "schema": paired.CHECKPOINT_SCHEMA, "step": step, "next_sample_index": step,
        "config": asdict(config), "source_checkpoint_sha256": config.expected_checkpoint_sha256,
        "model": model.state_dict(), "ema": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(), "runtime": {"device": "cuda:0"},
        "fork_parent_checkpoint_sha256": None if config.phase == "shared" else "shared-parent",
    }


def test_only_completed_shared_experiment_can_fork(config: paired.A12PairedTrainConfig) -> None:
    payload = resume_payload(config)
    for phase in ("fixed", "refresh"):
        child = replace(config, phase=phase, steps=1024, output_dir=config.output_dir + phase,
                        resume_checkpoint="shared512.pth", parent_checkpoint_sha256="parent")
        assert paired.validate_paired_resume_payload(payload, child) == 512
        changed = deepcopy(payload)
        changed["step"] = changed["next_sample_index"] = 128
        with pytest.raises(ValueError, match="shared step-512"):
            paired.validate_paired_resume_payload(changed, child)
        changed = deepcopy(payload)
        changed["config"]["purpose"] = "resume_verification"
        with pytest.raises(ValueError, match="shared step-512"):
            paired.validate_paired_resume_payload(changed, child)
        with pytest.raises(ValueError, match="configuration"):
            paired.validate_paired_resume_payload(payload, replace(child, seed=3))
        with pytest.raises(ValueError, match="configuration"):
            paired.validate_paired_resume_payload(payload, replace(child, expected_audit_sha256="changed"))


def test_branch_switch_and_changed_runtime_configuration_are_rejected(config: paired.A12PairedTrainConfig) -> None:
    fixed = replace(config, phase="fixed", steps=1024, resume_checkpoint="shared.pth", parent_checkpoint_sha256="shared")
    payload = resume_payload(fixed, 768)
    continuation = replace(fixed, output_dir=config.output_dir + "continue", resume_checkpoint="fixed768.pth",
                           parent_checkpoint_sha256="fixed768")
    assert paired.validate_paired_resume_payload(payload, continuation) == 768
    with pytest.raises(ValueError, match="only a shared"):
        paired.validate_paired_resume_payload(payload, replace(continuation, phase="refresh"))
    with pytest.raises(ValueError, match="configuration"):
        paired.validate_paired_resume_payload(payload, replace(continuation, device="cuda:1"))
    payload["fork_parent_checkpoint_sha256"] = None
    with pytest.raises(ValueError, match="original shared"):
        paired.validate_paired_resume_payload(payload, continuation)


def test_config_is_bounded_and_cannot_overwrite(config: paired.A12PairedTrainConfig) -> None:
    assert paired.validate_paired_config(config) == Path(config.output_dir)
    for changes in ({"steps": 1024}, {"phase": "fixed", "steps": 1024}, {"purpose": "resume_verification", "steps": 512},
                    {"learning_rate": 2e-5}, {"samples_per_epoch": 256}, {"audit_path": None},
                    {"sample_overrides_path": "overrides.json"}, {"expected_sample_overrides_sha256": "hash"}):
        with pytest.raises(ValueError):
            paired.validate_paired_config(replace(config, **changes))
    verification = replace(config, purpose="resume_verification", steps=2)
    assert paired.validate_paired_config(verification) == Path(config.output_dir)
    output = Path(config.output_dir)
    output.mkdir(parents=True)
    (output / "train.jsonl").write_text("existing\n")
    with pytest.raises(FileExistsError):
        paired.validate_paired_config(config)


class TinyDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.9))
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs) * self.scale


class TinyPairedDataset:
    def __init__(self, config: A3Config, sampling_mode: str, samples_per_epoch: int = 512,
                 plan_overrides_path: Path | None = None) -> None:
        self.config = config
        self.mode = sampling_mode
        self.samples = config.steps

    def __len__(self) -> int:
        return self.samples

    def sample_plan(self, index: int) -> A12SamplePlan:
        position = index % 512
        source = "real" if position % 2 == 0 else "synthetic"
        return A12SamplePlan(index, index // 512, position, source, 0, "800", 0, 0,
                             0 if source == "real" else None, position, position if source == "synthetic" else None)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.full((8, 4, 64, 64), 0.2 + (index % 2) * 0.1)
        targets = torch.full_like(inputs, 0.25)
        mask = torch.ones((8, 64, 64))
        return inputs, targets, mask


@pytest.fixture
def executable_config(config: paired.A12PairedTrainConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> paired.A12PairedTrainConfig:
    torch.set_num_threads(1)
    monkeypatch.setattr(paired, "build_rvideformer", TinyDenoiser)
    monkeypatch.setattr(paired, "A12PairedSamplingDataset", TinyPairedDataset)
    checkpoint = tmp_path / "a3.pth"
    torch.save({"step": 5000, "ema": TinyDenoiser().state_dict()}, checkpoint)
    source_sha = sha256_file(checkpoint)
    monkeypatch.setattr(paired, "A3_STEP5000_SHA256", source_sha)
    config = replace(config, checkpoint=str(checkpoint), expected_checkpoint_sha256=source_sha,
                     purpose="resume_verification", steps=2, device="cpu")
    dataset = TinyPairedDataset(replace(config, steps=1024), "refresh")
    hashes = {index: [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in dataset[index]] for index in (0, 1)}
    source_schedule = [{"position": index, "source": dataset.sample_plan(index).source,
                        "record_index": 0, "gain": "800"} for index in range(512)]
    manifest = {"source_schedule": source_schedule, "modes": {
        mode: [{"plan": asdict(dataset.sample_plan(index)), "tensor_sha256": hashes[index % 2]} for index in range(1024)]
        for mode in ("fixed", "refresh")
    }}
    manifest_path = tmp_path / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    audit = {
        "schema": paired.AUDIT_SCHEMA, "status": "passed", "samples_per_epoch": 512, "cycles": 2,
        "sample_overrides_path": None, "expected_sample_overrides_sha256": None,
        "config": {key: value for key, value in asdict(config).items() if key in A3Config.__dataclass_fields__},
        "checks": {key: True for key in (
            "first_cycle_matches_a12_plans", "identical_repeating_source_record_gain", "exact_source_ratio_every_cycle",
            "first_cycle_tensor_identity", "fixed_cycle_tensor_replay", "refresh_changes_real_inputs",
            "refresh_changes_synthetic_inputs", "valid_tensor_contents", "no_empty_supervision",
        )},
        "coverage": {mode: {"samples_read": 1024, "zero_valid_sample_indices": []} for mode in ("fixed", "refresh")},
        "metadata_sha256": {str(checkpoint): source_sha}, "raw_files": [],
        "source_schedule_sha256": hashlib.sha256(json.dumps(source_schedule, sort_keys=True).encode()).hexdigest(),
        "sample_manifest": str(manifest_path), "sample_manifest_sha256": sha256_file(manifest_path),
    }
    audit_path = Path(config.audit_path)
    audit_path.write_text(json.dumps(audit))
    return replace(config, expected_audit_sha256=sha256_file(audit_path))


def test_real_training_loop_restores_optimizer_ema_and_dropout_rng(executable_config: paired.A12PairedTrainConfig) -> None:
    full = executable_config
    paired._execute_training(full, torch.device("cpu"), paired.prepare_paired(full))
    prefix = replace(full, steps=1, output_dir=full.output_dir + "_prefix")
    paired._execute_training(prefix, torch.device("cpu"), paired.prepare_paired(prefix))
    prefix_checkpoint = Path(prefix.output_dir) / "checkpoint_step_000001.pth"
    resumed = replace(full, output_dir=full.output_dir + "_resumed", resume_checkpoint=str(prefix_checkpoint),
                      parent_checkpoint_sha256=sha256_file(prefix_checkpoint))
    paired._execute_training(resumed, torch.device("cpu"), paired.prepare_paired(resumed))
    full_payload = torch.load(Path(full.output_dir) / "checkpoint_step_000002.pth", weights_only=True)
    resumed_payload = torch.load(Path(resumed.output_dir) / "checkpoint_step_000002.pth", weights_only=True)
    for key in ("model", "ema", "optimizer", "rng"):
        assert paired.state_digest(full_payload[key]) == paired.state_digest(resumed_payload[key])
    full_rows = [json.loads(line) for line in (Path(full.output_dir) / "train.jsonl").read_text().splitlines()]
    resumed_rows = [json.loads(line) for line in (Path(resumed.output_dir) / "train.jsonl").read_text().splitlines()]
    assert full_rows[1] == resumed_rows[0]
    with pytest.raises(FileExistsError):
        paired.save_paired_checkpoint(prefix_checkpoint, TinyDenoiser(), TinyDenoiser(),
                                     torch.optim.Adam(TinyDenoiser().parameters()), 1, prefix, {}, None)


def test_audit_tampering_and_changed_samples_stop_before_update(executable_config: paired.A12PairedTrainConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    config = executable_config
    with pytest.raises(ValueError, match="audit configuration"):
        paired.load_paired_audit(replace(config, seed=33))
    with pytest.raises(ValueError, match="audit SHA"):
        paired.load_paired_audit(replace(config, expected_audit_sha256="incorrect"))
    preparation = paired.prepare_paired(config)
    original = TinyPairedDataset.__getitem__

    def changed(self: TinyPairedDataset, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, target, mask = original(self, index)
        inputs[0, 0, 0, 0] += 0.1
        return inputs, target, mask

    monkeypatch.setattr(TinyPairedDataset, "__getitem__", changed)
    with pytest.raises(ValueError, match="differs from the CPU audit"):
        paired._execute_training(config, torch.device("cpu"), preparation)
    assert (Path(config.output_dir) / "train.jsonl").read_text() == ""
    assert not list(Path(config.output_dir).glob("checkpoint*"))


def test_parent_hash_and_runtime_are_checked(executable_config: paired.A12PairedTrainConfig) -> None:
    prefix = replace(executable_config, steps=1)
    paired._execute_training(prefix, torch.device("cpu"), paired.prepare_paired(prefix))
    checkpoint = Path(prefix.output_dir) / "checkpoint_step_000001.pth"
    child = replace(executable_config, output_dir=prefix.output_dir + "_resumed", resume_checkpoint=str(checkpoint),
                    parent_checkpoint_sha256="wrong")
    with pytest.raises(ValueError, match="parent checkpoint SHA"):
        paired.prepare_paired(child)
    payload = torch.load(checkpoint, weights_only=True)
    payload["runtime"]["device_name"] = "different hardware"
    changed_checkpoint = Path(prefix.output_dir) / "runtime_changed.pth"
    torch.save(payload, changed_checkpoint)
    child = replace(child, resume_checkpoint=str(changed_checkpoint), parent_checkpoint_sha256=sha256_file(changed_checkpoint))
    with pytest.raises(ValueError, match="runtime/device"):
        paired._execute_training(child, torch.device("cpu"), paired.prepare_paired(child))


def test_override_identity_is_pinned_to_audit_and_checkpoint(
    executable_config: paired.A12PairedTrainConfig, tmp_path: Path,
) -> None:
    config = executable_config
    override_path = tmp_path / "empty_plan_overrides.json"
    override_path.write_text('{"schema":"test-only-identity"}')
    pinned = replace(config, sample_overrides_path=str(override_path),
                     expected_sample_overrides_sha256=sha256_file(override_path))
    with pytest.raises(ValueError, match="override identity mismatch"):
        paired.load_paired_audit(pinned)
    audit_path = Path(config.audit_path)
    audit = json.loads(audit_path.read_text())
    audit["sample_overrides_path"] = pinned.sample_overrides_path
    audit["expected_sample_overrides_sha256"] = pinned.expected_sample_overrides_sha256
    audit_path.write_text(json.dumps(audit))
    pinned = replace(pinned, expected_audit_sha256=sha256_file(audit_path))
    paired.load_paired_audit(pinned)
    shared = replace(pinned, purpose="experiment", steps=512)
    payload = resume_payload(shared)
    child = replace(shared, phase="refresh", steps=1024, resume_checkpoint="shared512.pth",
                    parent_checkpoint_sha256="shared512", output_dir=shared.output_dir + "refresh")
    assert paired.validate_paired_resume_payload(payload, child) == 512
    with pytest.raises(ValueError, match="configuration"):
        paired.validate_paired_resume_payload(payload, replace(child, expected_sample_overrides_sha256="changed"))
    fixed = replace(child, phase="fixed")
    fixed_payload = resume_payload(fixed, step=768)
    with pytest.raises(ValueError, match="configuration"):
        paired.validate_paired_resume_payload(fixed_payload, replace(fixed, sample_overrides_path="other.json"))
    override_path.write_text('{"schema":"modified"}')
    with pytest.raises(ValueError, match="overrides SHA-256"):
        paired.load_paired_audit(pinned)

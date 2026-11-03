"""Check sample refresh without changing source identity or supervision rules."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from raw_fusion.finetune import a12_sampling
from raw_fusion.finetune.a3 import A3Config, _RealRecord, _SyntheticRecord
from raw_fusion.finetune.a12_paired_sampling import A12PairedSamplingDataset, paired_sampling_config_identity
from raw_fusion.finetune.a12_sampling import A12SamplingDataset, A12StepSampler


@pytest.fixture
def paired_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> A3Config:
    y, x = np.indices((540, 960), dtype=np.float32)
    clean = np.broadcast_to(500.0 + y + x * 0.5, (4, 540, 960)).copy()
    valid = np.ones((540, 960), dtype=bool)
    valid[:, ::7] = False
    black = np.full(4, 256.0, dtype=np.float32)
    fpn_path = tmp_path / "fpn.npy"
    np.save(fpn_path, np.zeros_like(clean))
    monkeypatch.setattr(a12_sampling, "_load_reference", lambda _: (clean, valid, black))
    shifts = ((-11, 13), (17, -19), (-23, -29))
    records = [
        _RealRecord(tmp_path / f"real_{i}.json", f"gain{i}", 80 + i * 7,
                    shift_x, shift_y, np.ones(4), np.zeros(4), black,
                    tmp_path / f"motion_{i}.npy")
        for i, (shift_x, shift_y) in enumerate(shifts)
    ]
    monkeypatch.setattr(a12_sampling, "_real_records", lambda *_: records)
    monkeypatch.setattr(a12_sampling, "_synthetic_records", lambda _: [
        _SyntheticRecord(tmp_path / f"unused_{i}.npy", str(i)) for i in range(1, 6)
    ])
    noise_model = {
        "gain_factors": {str(i): float(i) for i in range(1, 6)},
        "empirical_shot_coefficients": {
            c: {str(i): 0.1 for i in range(1, 6)} for c in ("R", "Gr", "Gb", "B")
        },
        "read_variance_dn2": {c: {str(i): 4.0 for i in range(1, 6)} for c in ("R", "Gr", "Gb", "B")},
        "channel_correlation": {"aggregate": np.eye(4).tolist()},
    }

    def load_json(path: Path) -> dict:
        if path.name == "noise_model.json":
            return noise_model
        else:
            assert path.name == "a1_reference_manifest.json"
            return {"fpn_paths": {"offset_dn": str(fpn_path)}}

    monkeypatch.setattr(a12_sampling, "_load_json", load_json)
    return A3Config("unused", str(tmp_path), str(tmp_path), str(tmp_path), "unused", steps=128, device="cpu")


@pytest.mark.parametrize("mode", ["fixed", "refresh"])
def test_first_cycle_exactly_preserves_a12(paired_config: A3Config, mode: str) -> None:
    original = A12SamplingDataset(paired_config, samples_per_epoch=32)
    paired = A12PairedSamplingDataset(paired_config, mode, samples_per_epoch=32)
    assert [paired.sample_plan(i) for i in range(32)] == [original.sample_plan(i) for i in range(32)]
    assert all(torch.equal(a, b) for a, b in zip(paired[1], original[1], strict=True))


@pytest.mark.parametrize("mode", ["fixed", "refresh"])
def test_record_schedule_does_not_rotate_between_cycles(paired_config: A3Config, mode: str) -> None:
    paired = A12PairedSamplingDataset(paired_config, mode, samples_per_epoch=32)
    original = A12SamplingDataset(paired_config, samples_per_epoch=32)
    first = [paired.sample_plan(i) for i in range(32)]
    for epoch in (1, 2, 3):
        for position, old in enumerate(first):
            new = paired.sample_plan(epoch * 32 + position)
            assert (new.source, new.record_index, new.gain) == (old.source, old.record_index, old.gain)
            assert new.sample_index == epoch * 32 + position
            assert new.epoch == epoch and new.epoch_index == position
    assert original.sample_plan(32).record_index != paired.sample_plan(32).record_index
    assert original.sample_plan(33).record_index != paired.sample_plan(33).record_index


def test_fixed_replays_tensors_and_refresh_changes_samples(
    paired_config: A3Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def real_sample(
        self: A12SamplingDataset, record: _RealRecord, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = rng.random((8, 4, 64, 64), dtype=np.float32)
        return inputs, np.broadcast_to(inputs[0], inputs.shape).copy(), np.ones((8, 64, 64), dtype=np.float32)

    monkeypatch.setattr(A12SamplingDataset, "_real_sample", real_sample)
    fixed = A12PairedSamplingDataset(paired_config, "fixed", samples_per_epoch=32)
    refresh = A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32)
    for position in (0, 1, 4, 5):
        assert all(torch.equal(a, b) for a, b in zip(fixed[position], fixed[position + 32], strict=True))
        assert all(torch.equal(a, b) for a, b in zip(fixed[position], refresh[position], strict=True))
        assert not torch.equal(refresh[position][0], refresh[position + 32][0])
        assert all(torch.equal(a, b) for a, b in zip(refresh[position + 32], refresh[position + 32], strict=True))


@pytest.mark.parametrize("mode", ["fixed", "refresh"])
def test_real_refreshed_plan_matches_actual_pixels_and_motion_mask(
    paired_config: A3Config, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from raw_fusion.finetune.adapters import SourceAdapter

    dataset = A12PairedSamplingDataset(paired_config, mode, samples_per_epoch=32)
    y, x = np.indices((1080, 1920), dtype=np.uint16)
    base = 500 + y // 2 + x // 2

    class Reader:
        def read_frame(self, index: int) -> np.ndarray:
            return base + index

    adapter = SourceAdapter("04", "shdarkroom", "noisy", 0, "RGGB", 256, "single_file")
    monkeypatch.setattr(dataset, "_reader", lambda _: (Reader(), adapter))
    for index in (32, 34, 36):
        plan = dataset.sample_plan(index)
        record = dataset.real[plan.record_index]
        moving_y, moving_x = plan.crop_y + record.shift_y, plan.crop_x + record.shift_x
        assert 0 <= plan.crop_y <= 540 - 64 and 0 <= moving_y <= 540 - 64
        assert 0 <= plan.crop_x <= 960 - 64 and 0 <= moving_x <= 960 - 64
        motion = np.zeros((1, 1, 1080, 1920), dtype=np.float32)
        motion[0, 0, (moving_y + 3) * 2, (moving_x + 5) * 2] = 1
        dataset._motion_masks[record.motion_mask_path] = np.broadcast_to(motion, (record.frame_count, 1, 1080, 1920))
        inputs, targets, masks = dataset[index]
        expected_dn = 500 + moving_y + moving_x + plan.frame_start
        assert inputs[0, 0, 0, 0] == pytest.approx((expected_dn - 256) / (4095 - 256))
        assert inputs[-1, 0, 0, 0] == pytest.approx((expected_dn + 7 - 256) / (4095 - 256))
        assert torch.all(masks[:, 3, 5] == 0)
        expected_target = (500 + plan.crop_y + plan.crop_x * 0.5 - 256) / (4095 - 256)
        assert targets[0, 0, 0, 0] == pytest.approx(expected_target)
        assert torch.all(targets == targets[0])


def test_synthetic_refresh_can_change_noise_without_changing_target(
    paired_config: A3Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32)
    first = dataset.sample_plan(1)
    second = replace(dataset.sample_plan(33), crop_y=first.crop_y, crop_x=first.crop_x)
    assert first.gain == second.gain and first.record_index == second.record_index
    assert first.noise_seed != second.noise_seed
    monkeypatch.setattr(dataset, "sample_plan", lambda index: first if index == 1 else second)
    initial, refreshed = dataset[1], dataset[33]
    assert not torch.equal(initial[0], refreshed[0])
    assert torch.equal(initial[1], refreshed[1])
    assert torch.equal(initial[2], refreshed[2])


@pytest.mark.parametrize("mode", ["fixed", "refresh"])
def test_resume_and_extended_length_preserve_sample_identity(paired_config: A3Config, mode: str) -> None:
    initial = A12PairedSamplingDataset(replace(paired_config, steps=64), mode, samples_per_epoch=32)
    extended = A12PairedSamplingDataset(paired_config, mode, samples_per_epoch=32)
    assert list(A12StepSampler(initial, start_step=32)) == list(range(32, 64))
    assert [initial.sample_plan(i) for i in range(64)] == [extended.sample_plan(i) for i in range(64)]
    assert all(torch.equal(a, b) for a, b in zip(initial[33], extended[33], strict=True))


def test_invalid_mode_and_index_are_rejected(paired_config: A3Config) -> None:
    with pytest.raises(ValueError, match="sampling_mode"):
        A12PairedSamplingDataset(paired_config, "invalid")
    dataset = A12PairedSamplingDataset(paired_config, "fixed", samples_per_epoch=32)
    with pytest.raises(IndexError):
        dataset.sample_plan(-1)
    with pytest.raises(IndexError):
        dataset.sample_plan(len(dataset))
    with pytest.raises(TypeError):
        dataset.sample_plan(1.5)


@pytest.fixture
def overrides_manifest(paired_config: A3Config, tmp_path: Path) -> tuple[Path, dict]:
    dataset = A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32)
    original = dataset.sample_plan(50)
    record = dataset.real[original.record_index]
    seed = int(np.random.SeedSequence(
        [paired_config.seed, original.epoch, original.epoch_index, 12, 2, 0],
    ).generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(seed)
    y = int(rng.integers(max(0, -record.shift_y), min(540 - 64, 540 - 64 - record.shift_y) + 1))
    x = int(rng.integers(max(0, -record.shift_x), min(960 - 64, 960 - 64 - record.shift_x) + 1))
    start = int(rng.integers(0, record.frame_count - 8 + 1))
    replacement = replace(original, crop_y=y, crop_x=x, frame_start=start, geometry_seed=seed)
    tensor_hashes = ["0" * 64, "1" * 64, "2" * 64]
    records = [{} for _ in range(51)]
    records[50] = {"plan": asdict(original), "valid_fraction": 0.0, "tensor_sha256": tensor_hashes}
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps({"modes": {"refresh": records}}), encoding="utf-8")
    samples_sha = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({
        "sample_manifest_sha256": samples_sha,
        "coverage": {"fixed": {"zero_valid_sample_indices": []}, "refresh": {"zero_valid_sample_indices": [50]}},
    }), encoding="utf-8")
    manifest = {
        "schema": "a12_empty_plan_overrides_v1", "status": "resolved",
        "sampling_config": paired_sampling_config_identity(paired_config), "samples_per_epoch": 32,
        "source_audit": str(audit_path), "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_sample_manifest": str(samples_path), "source_sample_manifest_sha256": samples_sha,
        "overrides": [{
            "sampling_mode": "refresh", "sample_index": 50, "original_plan": asdict(original),
            "original_valid_fraction": 0.0, "original_tensor_sha256": tensor_hashes,
            "replacement_plan": asdict(replacement), "replacement_valid_fraction": 0.5,
            "replacement_tensor_sha256": tensor_hashes, "accepted_attempt": 0,
        }],
    }
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_override_only_changes_the_selected_refresh_plan(
    paired_config: A3Config, overrides_manifest: tuple[Path, dict],
) -> None:
    path, manifest = overrides_manifest
    plain = A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32)
    refresh = A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32, plan_overrides_path=path)
    fixed = A12PairedSamplingDataset(paired_config, "fixed", samples_per_epoch=32, plan_overrides_path=path)
    assert asdict(refresh.sample_plan(50)) == manifest["overrides"][0]["replacement_plan"]
    assert refresh.sample_plan(50) != plain.sample_plan(50)
    assert all(refresh.sample_plan(i) == plain.sample_plan(i) for i in range(len(plain)) if i != 50)
    assert fixed.sample_plan(50).geometry_seed == fixed.sample_plan(18).geometry_seed


@pytest.mark.parametrize("steps", [1, 2, 32])
def test_full_manifest_loads_for_short_shared_or_recovery_runs(
    paired_config: A3Config, overrides_manifest: tuple[Path, dict], steps: int,
) -> None:
    path, _ = overrides_manifest
    dataset = A12PairedSamplingDataset(replace(paired_config, steps=steps), "refresh", samples_per_epoch=32,
                                       plan_overrides_path=path)
    assert len(dataset) == steps
    assert dataset.sample_plan(0).sample_index == 0
    with pytest.raises(IndexError):
        dataset.sample_plan(50)


@pytest.mark.parametrize("change", ["gain", "geometry", "seed", "nonempty", "fixed", "position"])
def test_override_rejects_identity_geometry_and_nonempty_changes(
    paired_config: A3Config, overrides_manifest: tuple[Path, dict], change: str,
) -> None:
    path, manifest = overrides_manifest
    entry = manifest["overrides"][0]
    if change == "gain":
        entry["replacement_plan"]["gain"] = "other"
    elif change == "geometry":
        entry["replacement_plan"]["crop_y"] += 1
    elif change == "seed":
        entry["replacement_plan"]["geometry_seed"] += 1
    elif change == "nonempty":
        entry["original_valid_fraction"] = 0.1
    elif change == "fixed":
        entry["sampling_mode"] = "fixed"
    else:
        entry["sample_index"] = 0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32, plan_overrides_path=path)


def test_override_rejects_changed_audit_source(
    paired_config: A3Config, overrides_manifest: tuple[Path, dict],
) -> None:
    path, manifest = overrides_manifest
    audit_path = Path(manifest["source_audit"])
    audit_path.write_text(audit_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        A12PairedSamplingDataset(paired_config, "refresh", samples_per_epoch=32, plan_overrides_path=path)

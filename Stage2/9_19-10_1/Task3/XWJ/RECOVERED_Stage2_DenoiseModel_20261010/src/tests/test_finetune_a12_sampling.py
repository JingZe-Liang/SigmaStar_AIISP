"""A12 sample diversity, restart reproducibility, and supervision invariants."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from raw_fusion.finetune import a12_sampling
from raw_fusion.finetune.a3 import A3Config, _RealRecord, _SyntheticRecord
from raw_fusion.finetune.a12_sampling import A12SamplingDataset, A12StepSampler


@pytest.fixture
def sampling_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> A3Config:
    y, x = np.indices((540, 960), dtype=np.float32)
    clean = np.broadcast_to(500.0 + y + x * 0.5, (4, 540, 960)).copy()
    valid = np.ones((540, 960), dtype=bool)
    valid[:, ::7] = False
    black = np.full(4, 256.0, dtype=np.float32)
    fpn_path = tmp_path / "fpn.npy"
    np.save(fpn_path, np.zeros_like(clean))
    monkeypatch.setattr(a12_sampling, "_load_reference", lambda _: (clean, valid, black))
    monkeypatch.setattr(a12_sampling, "_real_records", lambda *_: [
        _RealRecord(tmp_path / f"real_{i}.json", f"gain{i}", 200, 0, 0,
                    np.ones(4), np.zeros(4), black, tmp_path / "motion.npy")
        for i in range(4)
    ])
    monkeypatch.setattr(a12_sampling, "_synthetic_records", lambda _: [
        _SyntheticRecord(tmp_path / f"unused_{i}.npy", str(i)) for i in (1, 2, 3)
    ])
    model = {
        "gain_factors": {str(i): float(i) for i in (1, 2, 3)},
        "empirical_shot_coefficients": {c: {str(i): 0.1 for i in (1, 2, 3)} for c in ("R", "Gr", "Gb", "B")},
        "read_variance_dn2": {c: {str(i): 4.0 for i in (1, 2, 3)} for c in ("R", "Gr", "Gb", "B")},
        "channel_correlation": {"aggregate": np.eye(4).tolist()},
    }

    def load_json(path: Path) -> dict:
        if path.name == "noise_model.json":
            return model
        else:
            assert path.name == "a1_reference_manifest.json"
            return {"fpn_paths": {"offset_dn": str(fpn_path)}}

    monkeypatch.setattr(a12_sampling, "_load_json", load_json)
    return A3Config("unused", str(tmp_path), str(tmp_path), str(tmp_path), "unused", steps=128, device="cpu")


@pytest.mark.parametrize("fraction,period", [(0.5, 32), (0.3, 20), (0.7, 20)])
def test_epoch_ratio_and_interleaving(sampling_config: A3Config, fraction: float, period: int) -> None:
    dataset = A12SamplingDataset(replace(sampling_config, real_fraction=fraction), samples_per_epoch=period)
    for epoch in range(3):
        sources = [dataset.sample_plan(epoch * period + i).source for i in range(period)]
        assert Counter(sources)["real"] == round(period * fraction)
        assert sources != sorted(sources)
        if fraction == 0.5:
            assert sources == ["real", "synthetic"] * (period // 2)
        else:
            assert all(len(set(sources[start : start + 4])) == 2 for start in range(period - 3))


def test_epochs_change_crop_time_and_noise(sampling_config: A3Config) -> None:
    dataset = A12SamplingDataset(sampling_config, samples_per_epoch=32)
    first = [dataset.sample_plan(i) for i in range(32)]
    second = [dataset.sample_plan(i + 32) for i in range(32)]
    assert any((a.crop_y, a.crop_x) != (b.crop_y, b.crop_x) for a, b in zip(first, second))
    assert any(a.frame_start != b.frame_start for a, b in zip(first[::2], second[::2]))
    assert all(a.noise_seed != b.noise_seed for a, b in zip(first[1::2], second[1::2]))
    assert not hasattr(dataset, "_synthetic_temporal")


def test_same_patch_gets_fresh_noise_and_same_supervision(
    sampling_config: A3Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = A12SamplingDataset(sampling_config, samples_per_epoch=32)
    first_plan = dataset.sample_plan(1)
    next_plan = replace(dataset.sample_plan(33), crop_y=first_plan.crop_y,
                        crop_x=first_plan.crop_x, gain=first_plan.gain, record_index=first_plan.record_index)
    monkeypatch.setattr(dataset, "sample_plan", lambda index: first_plan if index == 1 else next_plan)
    first = dataset[1]
    second = dataset[33]
    repeated = dataset[1]
    assert not torch.equal(first[0], second[0])
    assert not torch.equal(first[0][0], first[0][1])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))
    assert first[0].shape == (8, 4, 64, 64)
    assert first[2].shape == (8, 64, 64)
    assert first[2].min() == 0 and first[2].max() == 1
    assert all(torch.equal(first[1][0], target) for target in first[1])


def test_saturation_mask_excludes_pixel_if_any_frame_or_channel_clips(
    sampling_config: A3Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = A12SamplingDataset(sampling_config)
    calls = 0

    def noisy_frame(clean: np.ndarray, *_: object) -> np.ndarray:
        nonlocal calls
        result = clean.copy()
        if calls == 3:
            result[2, 4, 5] = 4095
        else:
            result[2, 4, 5] = 500
        calls += 1
        return result

    monkeypatch.setattr(a12_sampling, "generate_noisy_frame", noisy_frame)
    _, _, mask = dataset[1]
    assert torch.all(mask[:, 4, 5] == 0)


def test_resume_loader_matches_uninterrupted_samples(
    sampling_config: A3Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(sampling_config, steps=9, batch_size=2)

    def real_sample(self: A12SamplingDataset, record: _RealRecord, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = rng.random((8, 4, 64, 64), dtype=np.float32)
        return values, values.copy(), np.ones((8, 64, 64), dtype=np.float32)

    monkeypatch.setattr(A12SamplingDataset, "_real_sample", real_sample)
    dataset = A12SamplingDataset(config, samples_per_epoch=8)
    uninterrupted = list(DataLoader(dataset, batch_size=2, sampler=A12StepSampler(dataset)))
    restarted = A12SamplingDataset(config, samples_per_epoch=8)
    resumed = list(DataLoader(restarted, batch_size=2, sampler=A12StepSampler(restarted, start_step=5)))
    assert len(resumed) == 4
    assert all(torch.equal(a, b) for old, new in zip(uninterrupted[5:], resumed) for a, b in zip(old, new))
    assert list(A12StepSampler(restarted, start_step=9)) == []
    extended = A12SamplingDataset(replace(config, steps=12), samples_per_epoch=8)
    assert all(dataset.sample_plan(i) == extended.sample_plan(i) for i in range(len(dataset)))


def test_access_order_and_external_rng_do_not_change_samples(sampling_config: A3Config) -> None:
    dataset = A12SamplingDataset(sampling_config)
    first = dataset[1]
    _ = dataset[3]
    torch.manual_seed(9876)
    np.random.seed(1234)
    second_dataset = A12SamplingDataset(sampling_config)
    repeated = second_dataset[1]
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))


def test_real_plan_matches_pixels_and_masks(
    sampling_config: A3Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from raw_fusion.finetune.adapters import SourceAdapter

    dataset = A12SamplingDataset(sampling_config)
    plan = dataset.sample_plan(0)
    y, x = np.indices((1080, 1920), dtype=np.uint16)
    base = 500 + y // 2 + x // 2

    class Reader:
        def read_frame(self, index: int) -> np.ndarray:
            return base + index

    adapter = SourceAdapter("04", "shdarkroom", "noisy", 0, "RGGB", 256, "single_file")
    monkeypatch.setattr(dataset, "_reader", lambda _: (Reader(), adapter))
    motion = np.zeros((1, 1, 1080, 1920), dtype=np.float32)
    motion[0, 0, (plan.crop_y + 3) * 2, (plan.crop_x + 5) * 2] = 1
    dataset._motion_masks[dataset.real[plan.record_index].motion_mask_path] = np.broadcast_to(motion, (200, 1, 1080, 1920))
    inputs, targets, masks = dataset[0]
    expected_dn = 500 + plan.crop_y + plan.crop_x + plan.frame_start
    assert inputs[0, 0, 0, 0] == pytest.approx((expected_dn - 256) / (4095 - 256))
    assert inputs[-1, 0, 0, 0] == pytest.approx((expected_dn + 7 - 256) / (4095 - 256))
    assert torch.all(masks[:, 3, 5] == 0)
    assert torch.all(targets == targets[0])


def test_invalid_indices_and_empty_source_fraction_fail(sampling_config: A3Config) -> None:
    dataset = A12SamplingDataset(sampling_config)
    with pytest.raises(IndexError):
        dataset.sample_plan(len(dataset))
    with pytest.raises(IndexError):
        dataset.sample_plan(-1)
    with pytest.raises(ValueError):
        A12StepSampler(dataset, start_step=-1)
    with pytest.raises(ValueError):
        A12SamplingDataset(replace(sampling_config, real_fraction=0.01), samples_per_epoch=2)

"""A8 static sensor adaptation data-contract tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from raw_fusion.finetune.a8_adaptation import (
    A8AdaptationConfig,
    StaticSplitHalfDataset,
    configure_output_head_training,
    config_mapping,
    update_selected_ema,
)
from raw_fusion.finetune.a3 import build_rvideformer
from raw_fusion.finetune.cross_sensor import NativeSequenceSpec


def _build_fixture(tmp_path: Path) -> A8AdaptationConfig:
    dataset_root = tmp_path / "dataset"
    source_path = dataset_root / "sensor.raw"
    dataset_root.mkdir()
    frames = np.empty((32, 128, 128), dtype="<u2")
    base = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128) % 1024 + 500
    for index in range(len(frames)):
        frames[index] = (base + index % 2) << 4
    frames.tofile(source_path)

    source = NativeSequenceSpec(
        name="sensor_static",
        sensor="sensor",
        path=source_path,
        width=128,
        height=128,
        frames_used=32,
        bit_shift=4,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="motion_aware_split_half",
        clip_starts=(0,),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    packed_shape = (4, 64, 64)
    even = np.stack(
        [np.full((64, 64), 600.0 + channel) for channel in range(4)]
    ).astype(np.float32)
    odd = even + 20.0
    np.save(reference_dir / "full_mean_dn.npy", (even + odd) / 2.0)
    np.save(reference_dir / "even_mean_dn.npy", even)
    np.save(reference_dir / "odd_mean_dn.npy", odd)
    np.save(reference_dir / "even_confidence.npy", np.ones(packed_shape[1:], np.float32))
    np.save(reference_dir / "odd_confidence.npy", np.ones(packed_shape[1:], np.float32))
    np.save(reference_dir / "valid_region.npy", np.ones(packed_shape[1:], bool))
    np.save(reference_dir / "motion_threshold_per_frame.npy", np.full(32, 4095.0))
    (reference_dir / "report.json").write_text(
        json.dumps(
            {
                "black_levels": {"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0},
                "provenance": {
                    "source_path": str(source_path.resolve()),
                    "source_sha256": "source-hash",
                },
            }
        ),
        encoding="utf-8",
    )
    return A8AdaptationConfig(
        dataset_root=str(dataset_root),
        source=source,
        reference_dir=str(reference_dir),
        a3_checkpoint=str(tmp_path / "a3.pth"),
        a3_checkpoint_sha256="a3-hash",
        source_sha256="source-hash",
        replay_supervision_dir=str(tmp_path / "supervision"),
        replay_noise_dir=str(tmp_path / "noise"),
        output_dir=str(tmp_path / "output"),
        min_valid_fraction=1.0,
        device="cpu",
    )


def test_split_half_targets_use_opposite_frame_parity(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    dataset = StaticSplitHalfDataset(config, samples=2)
    _, even_input_target, even_mask = dataset[0]
    _, odd_input_target, odd_mask = dataset[1]
    denominator = 4095.0 - 256.0
    assert np.isclose(float(even_input_target[0, 0, 0, 0]), (620.0 - 256.0) / denominator)
    assert np.isclose(float(odd_input_target[0, 0, 0, 0]), (600.0 - 256.0) / denominator)
    assert bool(even_mask.all())
    assert bool(odd_mask.all())


def test_config_rejects_source_outside_authoritative_dataset(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    outside = tmp_path / "outside.raw"
    outside.write_bytes(b"raw")
    source = NativeSequenceSpec(
        name=config.source.name,
        sensor=config.source.sensor,
        path=outside,
        width=config.source.width,
        height=config.source.height,
        frames_used=config.source.frames_used,
        bit_shift=config.source.bit_shift,
        cfa=config.source.cfa,
        nominal_black_dn=config.source.nominal_black_dn,
        reference_mode=config.source.reference_mode,
        clip_starts=config.source.clip_starts,
        fixed_isp_wb=config.source.fixed_isp_wb,
        fixed_isp_exposure=config.source.fixed_isp_exposure,
    )
    invalid = A8AdaptationConfig(
        **{
            **config_mapping(config),
            "source": source,
        }
    )
    with pytest.raises(ValueError, match="authoritative dataset root"):
        invalid.validate()


def test_config_mapping_serializes_source_path(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    mapping = config_mapping(config)
    source = mapping["source"]
    assert isinstance(source, dict)
    assert source["name"] == "sensor_static"
    assert source["path"] == str(config.source.path)
    json.dumps(mapping)


def test_config_rejects_unknown_precision(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    invalid = A8AdaptationConfig(
        **{
            **config_mapping(config),
            "source": config.source,
            "precision": "float16",
        }
    )
    with pytest.raises(ValueError, match="precision"):
        invalid.validate()


def test_output_head_scope_freezes_rvideformer_trunk() -> None:
    model = build_rvideformer()
    parameters, report = configure_output_head_training(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert parameters
    assert report["scope"] == "output_head"
    assert trainable_names
    assert all(
        name.startswith(("norm.", "conv_after_body.", "conv_last."))
        for name in trainable_names
    )


def test_selected_ema_keeps_frozen_parameters_bit_exact() -> None:
    model = build_rvideformer()
    ema = build_rvideformer()
    ema.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        model.norm.weight.add_(1.0)
        model.conv_first.weight.add_(1.0)
    frozen_before = ema.conv_first.weight.detach().clone()
    update_selected_ema(ema, model, {"norm.weight"}, decay=0.5)
    assert torch.equal(ema.conv_first.weight, frozen_before)
    assert torch.allclose(
        ema.norm.weight,
        model.norm.weight - 0.5,
    )

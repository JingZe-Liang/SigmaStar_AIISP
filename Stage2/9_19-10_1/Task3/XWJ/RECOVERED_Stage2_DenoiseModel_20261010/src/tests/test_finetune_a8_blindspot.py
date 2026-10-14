"""A8 blind-spot sensor adaptation data-contract tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from raw_fusion.finetune.a8_blindspot import (
    A8BlindSpotConfig,
    BlindSpotSingleFrameDataset,
    blindspot_config_mapping,
    blindspot_loss,
)
from raw_fusion.finetune.cross_sensor import NativeSequenceSpec


def _build_fixture(tmp_path: Path) -> A8BlindSpotConfig:
    dataset_root = tmp_path / "dataset"
    source_path = dataset_root / "sensor.raw"
    dataset_root.mkdir()
    frames = np.empty((20, 128, 128), dtype="<u2")
    base = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128) % 512 + 500
    for index in range(len(frames)):
        frames[index] = base + index * 4
    frames.tofile(source_path)
    source = NativeSequenceSpec(
        name="sensor_dynamic",
        sensor="sensor",
        path=source_path,
        width=128,
        height=128,
        frames_used=20,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="stress_only",
        clip_starts=(0, 12),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "report.json").write_text(
        json.dumps(
            {
                "black_levels": {
                    "R": 256.0,
                    "Gr": 256.0,
                    "Gb": 256.0,
                    "B": 256.0,
                },
                "provenance": {
                    "source_path": str(source_path.resolve()),
                    "source_sha256": "source-hash",
                },
            }
        ),
        encoding="utf-8",
    )
    return A8BlindSpotConfig(
        dataset_root=str(dataset_root),
        source=source,
        profile_dir=str(profile_dir),
        a3_checkpoint=str(tmp_path / "a3.pth"),
        a3_checkpoint_sha256="a3-hash",
        source_sha256="source-hash",
        replay_supervision_dir=str(tmp_path / "supervision"),
        replay_noise_dir=str(tmp_path / "noise"),
        output_dir=str(tmp_path / "output"),
        train_frame_start=0,
        train_frame_stop=12,
        validation_frame_start=12,
        validation_frame_stop=20,
        blindspot_fraction=0.05,
        blindspot_radius=2,
        device="cpu",
    )


def test_blindspot_dataset_hides_same_locations_in_every_frame(
    tmp_path: Path,
) -> None:
    dataset = BlindSpotSingleFrameDataset(_build_fixture(tmp_path), samples=1)
    inputs, targets, masks = dataset[0]
    assert inputs.shape == (8, 4, 64, 64)
    assert targets.shape == (8, 4, 64, 64)
    assert masks.shape == (8, 64, 64)
    assert torch.equal(inputs[0], inputs[-1])
    assert torch.equal(targets[0], targets[-1])
    assert torch.equal(masks[0], masks[-1])
    expanded_mask = masks[0].bool().unsqueeze(0).expand_as(inputs[0])
    assert bool(torch.any(inputs[0][expanded_mask] != targets[0][expanded_mask]))
    assert torch.equal(inputs[0][~expanded_mask], targets[0][~expanded_mask])
    assert 0.03 <= float(masks[0].mean()) <= 0.07


def test_blindspot_config_rejects_overlapping_frame_ranges(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    mapping = blindspot_config_mapping(config)
    mapping["source"] = config.source
    mapping["validation_frame_start"] = 11
    invalid = A8BlindSpotConfig(**mapping)
    with pytest.raises(ValueError, match="disjoint"):
        invalid.validate()


def test_blindspot_config_rejects_static_reference_source(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    source = NativeSequenceSpec(
        name=config.source.name,
        sensor=config.source.sensor,
        path=config.source.path,
        width=config.source.width,
        height=config.source.height,
        frames_used=config.source.frames_used,
        bit_shift=config.source.bit_shift,
        cfa=config.source.cfa,
        nominal_black_dn=config.source.nominal_black_dn,
        reference_mode="motion_aware_split_half",
        clip_starts=(0,),
        fixed_isp_wb=config.source.fixed_isp_wb,
        fixed_isp_exposure=config.source.fixed_isp_exposure,
    )
    mapping = blindspot_config_mapping(config)
    mapping["source"] = source
    invalid = A8BlindSpotConfig(**mapping)
    with pytest.raises(ValueError, match="stress-only"):
        invalid.validate()


def test_blindspot_loss_ignores_unmasked_errors() -> None:
    prediction = torch.zeros((1, 8, 4, 4, 4))
    target = torch.zeros_like(prediction)
    mask = torch.zeros((1, 8, 1, 4, 4))
    prediction[..., 0, 0] = 1.0
    loss, _ = blindspot_loss(prediction, target, mask)
    assert float(loss) == 0.0
    mask[..., 1, 1] = 1.0
    prediction[..., 1, 1] = 0.5
    loss, _ = blindspot_loss(prediction, target, mask)
    assert 0.49 < float(loss) < 0.51


def test_blindspot_config_mapping_is_json_serializable(tmp_path: Path) -> None:
    mapping = blindspot_config_mapping(_build_fixture(tmp_path))
    source = mapping["source"]
    assert isinstance(source, dict)
    assert source["name"] == "sensor_dynamic"
    json.dumps(mapping)

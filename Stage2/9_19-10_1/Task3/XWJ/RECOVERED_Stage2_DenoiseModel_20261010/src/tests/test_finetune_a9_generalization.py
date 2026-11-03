"""A9 unified multi-sensor training and noise-profile tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from raw_fusion.finetune.a3 import build_rvideformer
from raw_fusion.finetune.a9_generalization import (
    A9CleanReference,
    A9GeneralizationConfig,
    A9StaticSource,
    A9SyntheticSource,
    NoiseObservation,
    configure_input_output_head_training,
    fit_variance_models,
    generate_profile_noisy_frame,
)
from raw_fusion.finetune.cross_sensor import NativeSequenceSpec


def _source(tmp_path: Path) -> NativeSequenceSpec:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    path = dataset_root / "sensor.raw"
    path.write_bytes(b"raw")
    return NativeSequenceSpec(
        name="sensor_static",
        sensor="sensor",
        path=path,
        width=128,
        height=128,
        frames_used=32,
        bit_shift=0,
        cfa="RGGB",
        nominal_black_dn=256.0,
        reference_mode="motion_aware_split_half",
        clip_starts=(0,),
        fixed_isp_wb=(1.0, 1.0, 1.0, 1.0),
        fixed_isp_exposure=1.0,
    )


def _config(tmp_path: Path) -> A9GeneralizationConfig:
    source = _source(tmp_path)
    return A9GeneralizationConfig(
        dataset_root=str(tmp_path / "dataset"),
        a3_checkpoint=str(tmp_path / "a3.pth"),
        a3_checkpoint_sha256="a3",
        replay_supervision_dir=str(tmp_path / "supervision"),
        replay_noise_dir=str(tmp_path / "noise"),
        replay_weight=0.4,
        static_sources=(
            A9StaticSource(
                name="static",
                source=source,
                reference_dir=str(tmp_path / "reference"),
                source_sha256="source",
                weight=0.4,
            ),
        ),
        synthetic_sources=(
            A9SyntheticSource(
                name="synthetic",
                sensor="other",
                profile_path=str(tmp_path / "profile.json"),
                profile_sha256="profile",
                weight=0.2,
            ),
        ),
        clean_references=(
            A9CleanReference(
                name="clean",
                reference_path=str(tmp_path / "clean.npy"),
                reference_sha256="clean",
                valid_region_path=str(tmp_path / "valid.npy"),
                valid_region_sha256="valid",
                black_levels=(256.0, 256.0, 256.0, 256.0),
            ),
        ),
        output_dir=str(tmp_path / "output"),
        device="cpu",
    )


def test_a9_config_requires_weights_to_sum_to_one(tmp_path: Path) -> None:
    config = _config(tmp_path)
    invalid = replace(config, replay_weight=0.5)
    with pytest.raises(ValueError, match="sum to 1.0"):
        invalid.validate()


def test_robust_variance_fit_limits_one_large_outlier() -> None:
    observations = []
    for channel in ("R", "Gr", "Gb", "B"):
        for index, signal in enumerate((0.0, 50.0, 200.0, 800.0, 1600.0, 3000.0)):
            variance = 9.0 + 0.04 * signal
            if index == 4:
                variance *= 9.0
            observations.append(
                NoiseObservation(
                    channel=channel,
                    signal_dn=signal,
                    sigma_dn=float(np.sqrt(variance)),
                    pixels=4096,
                    pair_first_frame=index * 2,
                )
            )
    models = fit_variance_models(observations)
    for model in models.values():
        assert np.isclose(model["shot_variance_per_signal_dn"], 0.04, atol=0.02)
        assert np.isclose(model["read_variance_dn2"], 9.0, atol=8.0)


def test_profile_noise_matches_requested_variance() -> None:
    profile = {
        "authenticity": {"passed": True},
        "black_levels": {"R": 256.0, "Gr": 256.0, "Gb": 256.0, "B": 256.0},
        "variance_model": {
            name: {
                "shot_variance_per_signal_dn": 0.0,
                "read_variance_dn2": 100.0,
            }
            for name in ("R", "Gr", "Gb", "B")
        },
    }
    clean = np.full((4, 256, 256), 0.25, dtype=np.float32)
    noisy = generate_profile_noisy_frame(clean, profile, np.random.default_rng(3))
    expected = 256.0 + 0.25 * (4095.0 - 256.0)
    assert np.isclose(float(noisy.mean()), expected, atol=0.2)
    assert np.isclose(float(noisy.std()), 10.0, atol=0.2)


def test_input_output_scope_keeps_transformer_trunk_frozen() -> None:
    model = build_rvideformer()
    parameters, report = configure_input_output_head_training(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert parameters
    assert report["scope"] == "input_output_heads"
    assert any(name.startswith("conv_first.") for name in trainable_names)
    assert all(
        name.startswith(("conv_first.", "norm.", "conv_after_body.", "conv_last."))
        for name in trainable_names
    )

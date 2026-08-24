import json
from dataclasses import replace
from pathlib import Path

import pytest

from raw_fusion.config import (
    DatasetConfig,
    ExperimentConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
    RawLayout,
    SequenceConfig,
    SplitConfig,
    TrainConfig,
    config_fingerprint,
    load_dataset_config,
    load_experiment_config,
)


def write_minimal_dataset_json(path: Path) -> Path:
    dataset_path = path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "layout": {
                    "width": 1920,
                    "height": 1080,
                    "frame_count": 10,
                    "dtype": "uint16",
                    "white_level": 1023,
                    "noisy_black_level": 64,
                    "candidate_black_level": 300,
                    "target_black_level": 64,
                    "noisy_shift": 0,
                    "cfa_pattern": "RGGB",
                    "pseudo_gt_pattern": "{frame:04d}.raw",
                },
                "sequences": {
                    "train": {
                        "name": "train",
                        "noisy_stream": "noisy.raw",
                        "denoised_stream": "denoised.raw",
                        "fused_stream": "fused.raw",
                        "pseudo_gt_dir": "pseudo_gt",
                        "white_balance": [2.0, 1.0, 1.5],
                        "isp_gain": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_path


def write_minimal_experiment_json(
    path: Path, dataset: str, extra: dict[str, object] | None = None
) -> Path:
    experiment_path = path / "experiment.json"
    document: dict[str, object] = {
        "dataset": dataset,
        "split": {
            "train_sequence": "train",
            "train_frames": [0, 7],
            "validation_sequence": "train",
            "validation_frames": [8, 8],
            "test_sequence": "train",
            "test_frames": [9, 9],
        },
        "model": {
            "channels": [24, 48, 72],
            "residual_scale": 0.1,
            "use_temporal": True,
        },
        "loss": {
            "gradient_weight": 1.0,
            "gate_weight": 1.0,
            "residual_weight": 1.0,
            "range_weight": 1.0,
            "charbonnier_epsilon": 0.001,
            "gate_temperature": 1.0,
            "gate_margin": 0.1,
            "saturation_margin_dn": 4,
        },
        "train": {
            "patch_size_packed": 128,
            "samples_per_epoch": 100,
            "batch_size": 4,
            "epochs": 10,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "num_workers": 0,
            "seed": 1,
            "amp": True,
            "device": "cpu",
        },
        "inference": {"tile_size_packed": 128, "overlap_packed": 16},
    }
    if extra:
        document.update(extra)
    experiment_path.write_text(json.dumps(document), encoding="utf-8")
    return experiment_path


def test_load_config_resolves_dataset_relative_to_experiment(tmp_path: Path) -> None:
    dataset_path = write_minimal_dataset_json(tmp_path)
    experiment_path = write_minimal_experiment_json(tmp_path, dataset_path.name)

    config = load_experiment_config(experiment_path)

    assert config.dataset_path == dataset_path.resolve()
    assert config.model.channels == (24, 48, 72)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = write_minimal_experiment_json(tmp_path, "dataset.json", extra={"typo": 1})

    with pytest.raises(ValueError, match="未知字段.*typo"):
        load_experiment_config(path)


def test_fingerprint_changes_when_normalization_changes(tmp_path: Path) -> None:
    dataset_a = make_dataset_config(candidate_black=300)
    dataset_b = make_dataset_config(candidate_black=301)
    experiment = make_experiment_config()

    assert config_fingerprint(dataset_a, experiment) != config_fingerprint(dataset_b, experiment)


def test_dataset_layout_rejects_non_rggb_cfa_pattern(tmp_path: Path) -> None:
    dataset_path = write_minimal_dataset_json(tmp_path)
    document = json.loads(dataset_path.read_text(encoding="utf-8"))
    document["layout"]["cfa_pattern"] = "BGGR"
    dataset_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="cfa_pattern.*RGGB"):
        load_dataset_config(dataset_path)


def test_fingerprint_changes_when_cfa_pattern_changes() -> None:
    dataset = make_dataset_config(candidate_black=300)
    changed_cfa = replace(dataset, layout=replace(dataset.layout, cfa_pattern="BGGR"))
    experiment = make_experiment_config()

    assert config_fingerprint(dataset, experiment) != config_fingerprint(changed_cfa, experiment)


def make_dataset_config(candidate_black: int) -> DatasetConfig:
    return DatasetConfig(
        layout=RawLayout(
            width=1920,
            height=1080,
            frame_count=10,
            dtype="uint16",
            white_level=1023,
            noisy_black_level=64,
            candidate_black_level=candidate_black,
            target_black_level=64,
            noisy_shift=0,
            cfa_pattern="RGGB",
            pseudo_gt_pattern="{frame:04d}.raw",
        ),
        sequences={
            "train": SequenceConfig(
                name="train",
                noisy_stream=Path("/data/noisy.raw"),
                denoised_stream=Path("/data/denoised.raw"),
                fused_stream=Path("/data/fused.raw"),
                pseudo_gt_dir=Path("/data/pseudo_gt"),
                white_balance=(2.0, 1.0, 1.5),
                isp_gain=1.0,
            )
        },
    )


def make_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset_path=Path("/data/dataset.json"),
        split=SplitConfig("train", (0, 7), "train", (8, 8), "train", (9, 9)),
        model=ModelConfig((24, 48, 72), 0.1, True),
        loss=LossConfig(1.0, 1.0, 1.0, 1.0, 0.001, 1.0, 0.1, 4),
        train=TrainConfig(128, 100, 4, 10, 0.0001, 0.0, 0, 1, True, "cpu"),
        inference=InferenceConfig(128, 16),
    )

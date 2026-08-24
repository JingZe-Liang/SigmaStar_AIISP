from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from raw_fusion.checkpoint import save_checkpoint_atomic
from raw_fusion.config import config_fingerprint, load_dataset_config, load_experiment_config
from raw_fusion.infer import (
    AtomicRawStreamWriter,
    _validate_tile_output,
    infer_sequence,
    infer_tiled,
    tile_starts,
)
from raw_fusion.model import FusionOutput


class AverageModel(nn.Module):
    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        prediction = 0.25 * (prev_noisy + curr_noisy + denoised + fused)
        return FusionOutput(
            prediction=prediction,
            base=prediction,
            gate=torch.full_like(prediction[:, :1], 0.25),
            correction=torch.full_like(prediction, 0.125),
        )


class TinyFusionNet(nn.Module):
    """Checkpoint-compatible test double for an 8x8 sensor-frame integration test."""

    def __init__(self, _model_config: object) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        del denoised, fused
        prediction = 0.25 * prev_noisy + 0.75 * curr_noisy + self.offset
        return FusionOutput(
            prediction=prediction,
            base=prediction,
            gate=torch.full_like(prediction[:, :1], 0.5),
            correction=torch.zeros_like(prediction),
        )


def make_inputs(*, batch: int, height: int, width: int) -> dict[str, torch.Tensor]:
    values = torch.arange(batch * 4 * height * width, dtype=torch.float32).reshape(
        batch, 4, height, width
    )
    return {
        "prev_noisy": values / 10_000.0,
        "curr_noisy": values / 8_000.0,
        "denoised": values / 6_000.0,
        "fused": values / 4_000.0,
    }


def test_tile_starts_include_last_boundary() -> None:
    assert tile_starts(53, tile=16, overlap=4) == (0, 12, 24, 36, 37)


@pytest.mark.parametrize(
    ("length", "tile", "overlap"),
    ((0, 1, 0), (5, 0, 0), (5, 6, 0), (5, 4, -1), (5, 4, 4)),
)
def test_tile_starts_rejects_invalid_geometry(length: int, tile: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        tile_starts(length, tile=tile, overlap=overlap)


def test_tiled_identity_matches_direct_on_irregular_shape() -> None:
    model = AverageModel()
    inputs = make_inputs(batch=1, height=37, width=53)
    direct = model(**inputs)

    tiled = infer_tiled(model, inputs, tile_size=16, overlap=4)

    torch.testing.assert_close(tiled.prediction, direct.prediction, rtol=0, atol=1e-6)
    torch.testing.assert_close(tiled.gate, direct.gate, rtol=0, atol=1e-6)
    torch.testing.assert_close(tiled.correction, direct.correction, rtol=0, atol=1e-6)
    assert tiled.prediction.dtype == torch.float32
    assert torch.all(tiled.weight_sum > 0)


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_infer_tiled_rejects_nonfinite_inputs(nonfinite: float) -> None:
    inputs = make_inputs(batch=1, height=16, width=16)
    inputs["fused"][0, 0, 0, 0] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        infer_tiled(AverageModel(), inputs, tile_size=16, overlap=4)


class NonfiniteOutputModel(AverageModel):
    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        output = super().forward(prev_noisy, curr_noisy, denoised, fused)
        return FusionOutput(
            prediction=torch.full_like(output.prediction, float("nan")),
            base=output.base,
            gate=torch.full_like(output.gate, float("inf")),
            correction=output.correction,
        )


def test_infer_tiled_rejects_nonfinite_model_outputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        infer_tiled(
            NonfiniteOutputModel(),
            make_inputs(batch=1, height=16, width=16),
            tile_size=16,
            overlap=4,
        )


def test_validate_tile_output_rejects_wrong_device_before_accumulation() -> None:
    output = SimpleNamespace(
        prediction=torch.empty((1, 4, 2, 2), device="meta"),
        gate=torch.empty((1, 1, 2, 2), device="meta"),
        correction=torch.empty((1, 4, 2, 2), device="meta"),
    )

    with pytest.raises(ValueError, match="device"):
        _validate_tile_output(output, batch=1, tile=2, expected_device=torch.device("cpu"))


def test_infer_tiled_restores_training_mode_after_success() -> None:
    model = AverageModel().train()

    infer_tiled(model, make_inputs(batch=1, height=16, width=16), tile_size=16, overlap=4)

    assert model.training


class FailingModel(AverageModel):
    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        del prev_noisy, curr_noisy, denoised, fused
        raise RuntimeError("intentional model failure")


def test_infer_tiled_restores_training_mode_after_model_failure() -> None:
    model = FailingModel().train()

    with pytest.raises(RuntimeError, match="intentional"):
        infer_tiled(model, make_inputs(batch=1, height=16, width=16), tile_size=16, overlap=4)

    assert model.training


def test_inference_quantization_preserves_expected_byte_count(tmp_path: Path) -> None:
    writer = AtomicRawStreamWriter(tmp_path / "result.raw", width=4, height=2, frame_count=2)
    writer.write(np.full((2, 4), 252, dtype="<u2"))
    writer.write(np.full((2, 4), 4095, dtype="<u2"))
    writer.close()

    assert (tmp_path / "result.raw").stat().st_size == 32


def test_writer_rejects_fewer_frames_without_publishing_output(tmp_path: Path) -> None:
    destination = tmp_path / "partial.raw"
    writer = AtomicRawStreamWriter(destination, width=2, height=2, frame_count=2)
    writer.write(np.full((2, 2), 252, dtype="<u2"))

    with pytest.raises(RuntimeError, match="frame"):
        writer.close()

    assert not destination.exists()


def test_writer_rejects_extra_or_malformed_frame_without_publishing_output(tmp_path: Path) -> None:
    malformed_destination = tmp_path / "malformed.raw"
    malformed_writer = AtomicRawStreamWriter(malformed_destination, width=2, height=2, frame_count=1)

    with pytest.raises(ValueError, match="shape"):
        malformed_writer.write(np.zeros((1, 2), dtype="<u2"))
    with pytest.raises(RuntimeError, match="closed"):
        malformed_writer.write(np.zeros((2, 2), dtype="<u2"))
    assert not malformed_destination.exists()

    destination = tmp_path / "extra.raw"
    writer = AtomicRawStreamWriter(destination, width=2, height=2, frame_count=1)
    writer.write(np.zeros((2, 2), dtype="<u2"))
    with pytest.raises(RuntimeError, match="more"):
        writer.write(np.zeros((2, 2), dtype="<u2"))
    writer.abort()

    assert not destination.exists()


def _write_inference_configs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.json"
    experiment_path = tmp_path / "experiment.json"
    dataset = {
        "layout": {
            "width": 8,
            "height": 8,
            "frame_count": 2,
            "dtype": "<u2",
            "white_level": 4095,
            "noisy_black_level": 252,
            "candidate_black_level": 300,
            "target_black_level": 252,
            "noisy_shift": 4,
            "cfa_pattern": "RGGB",
            "pseudo_gt_pattern": "unused_{index:04d}.raw",
        },
        "sequences": {
            "synthetic": {
                "name": "synthetic",
                "noisy_stream": "noisy.raw",
                "denoised_stream": "denoised.raw",
                "fused_stream": "fused.raw",
                "pseudo_gt_dir": "targets",
                "white_balance": [1.0, 1.0, 1.0],
                "isp_gain": 1.0,
            }
        },
    }
    experiment = {
        "dataset": "dataset.json",
        "split": {
            "train_sequence": "synthetic",
            "train_frames": [0, 1],
            "validation_sequence": "synthetic",
            "validation_frames": [0, 1],
            "test_sequence": "synthetic",
            "test_frames": [0, 1],
        },
        "model": {"channels": [4, 4, 4], "residual_scale": 0.03, "use_temporal": True},
        "loss": {
            "gradient_weight": 0.0,
            "gate_weight": 0.0,
            "residual_weight": 0.0,
            "range_weight": 0.0,
            "charbonnier_epsilon": 0.001,
            "gate_temperature": 0.02,
            "gate_margin": 0.005,
            "saturation_margin_dn": 4,
        },
        "train": {
            "patch_size_packed": 4,
            "samples_per_epoch": 1,
            "batch_size": 1,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
            "seed": 7,
            "amp": False,
            "device": "cpu",
        },
        "inference": {"tile_size_packed": 4, "overlap_packed": 1},
    }
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    return dataset_path, experiment_path


def test_infer_sequence_writes_two_quantized_synthetic_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import raw_fusion.infer as infer_module

    dataset_path, experiment_path = _write_inference_configs(tmp_path)
    noisy = np.array(
        [
            np.full((8, 8), 252 << 4, dtype="<u2"),
            np.full((8, 8), 4095 << 4, dtype="<u2"),
        ]
    )
    denoised = np.full((2, 8, 8), 300, dtype="<u2")
    fused = np.full((2, 8, 8), 4095, dtype="<u2")
    (tmp_path / "noisy.raw").write_bytes(noisy.tobytes())
    (tmp_path / "denoised.raw").write_bytes(denoised.tobytes())
    (tmp_path / "fused.raw").write_bytes(fused.tobytes())
    dataset = load_dataset_config(dataset_path)
    experiment = load_experiment_config(experiment_path)
    checkpoint = tmp_path / "tiny.pt"
    tiny_model = TinyFusionNet(experiment.model)
    save_checkpoint_atomic(
        checkpoint,
        {
            "schema_version": 1,
            "model": tiny_model.state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": None,
            "epoch": 0,
            "global_step": 0,
            "config_fingerprint": config_fingerprint(dataset, experiment),
            "experiment_config": {},
            "seed": 7,
        },
    )
    monkeypatch.setattr(infer_module, "CausalRawFusionNet", TinyFusionNet)

    manifest = infer_sequence(
        experiment_path,
        checkpoint,
        "synthetic",
        (0, 1),
        tmp_path / "output",
        "cpu",
    )

    output = np.fromfile(manifest.output_raw_path, dtype="<u2")
    np.testing.assert_array_equal(
        output,
        np.concatenate(
            (np.full(64, 252, dtype="<u2"), np.full(64, 3134, dtype="<u2"))
        ),
    )
    assert manifest.output_raw_bytes == 256
    assert manifest.minimum == 252
    assert manifest.maximum == 3134
    saved = json.loads((tmp_path / "output" / "manifest.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert saved["config_fingerprint"] == config_fingerprint(dataset, experiment)
    assert saved["sequence"] == "synthetic"
    assert saved["frame_range"] == [0, 1]
    assert saved["frame_count"] == 2
    assert saved["output_raw_bytes"] == 256
    assert saved["minimum"] == 252
    assert saved["maximum"] == 3134


def test_infer_sequence_removes_stale_manifest_before_checkpoint_failure(tmp_path: Path) -> None:
    _, experiment_path = _write_inference_configs(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stale_manifest = output_dir / "manifest.json"
    stale_manifest.write_text('{"stale": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="检查点"):
        infer_sequence(
            experiment_path,
            tmp_path / "missing.pt",
            "synthetic",
            (0, 1),
            output_dir,
            "cpu",
        )

    assert not stale_manifest.exists()


def test_infer_sequence_hashes_and_loads_same_checkpoint_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import raw_fusion.infer as infer_module

    dataset_path, experiment_path = _write_inference_configs(tmp_path)
    noisy = np.array(
        [
            np.full((8, 8), 252 << 4, dtype="<u2"),
            np.full((8, 8), 4095 << 4, dtype="<u2"),
        ]
    )
    (tmp_path / "noisy.raw").write_bytes(noisy.tobytes())
    (tmp_path / "denoised.raw").write_bytes(np.full((2, 8, 8), 300, dtype="<u2").tobytes())
    (tmp_path / "fused.raw").write_bytes(np.full((2, 8, 8), 4095, dtype="<u2").tobytes())
    dataset = load_dataset_config(dataset_path)
    experiment = load_experiment_config(experiment_path)
    checkpoint = tmp_path / "tiny.pt"
    original_model = TinyFusionNet(experiment.model)
    replacement_model = TinyFusionNet(experiment.model)
    with torch.no_grad():
        replacement_model.offset.fill_(0.2)
    fingerprint = config_fingerprint(dataset, experiment)
    original_state = {
        "schema_version": 1,
        "model": original_model.state_dict(),
        "optimizer": {"state": {}, "param_groups": []},
        "scaler": None,
        "epoch": 0,
        "global_step": 0,
        "config_fingerprint": fingerprint,
        "experiment_config": {},
        "seed": 7,
    }
    replacement_state = {**original_state, "model": replacement_model.state_dict()}
    save_checkpoint_atomic(checkpoint, original_state)
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    real_load = infer_module.load_checkpoint_strict

    def load_then_replace(path: Path, **kwargs: object) -> object:
        restored = real_load(path, **kwargs)
        save_checkpoint_atomic(checkpoint, replacement_state)
        return restored

    monkeypatch.setattr(infer_module, "CausalRawFusionNet", TinyFusionNet)
    monkeypatch.setattr(infer_module, "load_checkpoint_strict", load_then_replace)

    manifest = infer_sequence(
        experiment_path,
        checkpoint,
        "synthetic",
        (0, 1),
        tmp_path / "output",
        "cpu",
    )

    assert manifest.checkpoint_sha256 == original_hash
    assert manifest.checkpoint_sha256 != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = json.loads((tmp_path / "output" / "manifest.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_sha256"] == original_hash
    np.testing.assert_array_equal(
        np.fromfile(manifest.output_raw_path, dtype="<u2"),
        np.concatenate(
            (np.full(64, 252, dtype="<u2"), np.full(64, 3134, dtype="<u2"))
        ),
    )
    assert not list((tmp_path / "output").glob(".checkpoint-snapshot.*"))

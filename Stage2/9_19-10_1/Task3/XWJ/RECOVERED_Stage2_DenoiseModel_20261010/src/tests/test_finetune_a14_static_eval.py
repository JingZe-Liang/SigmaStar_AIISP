"""Frozen split-half evaluation, structural fidelity and failure semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from raw_fusion.finetune import a12_paired_eval, a14_static_eval
from raw_fusion.finetune.a12_train import sha256_file
from raw_fusion.finetune.a14_static_eval import (
    evaluate_static_checkpoint, prepare_static_cases, score_static_prediction,
    select_reference_rois, summarize_cases,
)


def test_high_frequency_energy_does_not_replace_target_fidelity() -> None:
    checker = (np.indices((12, 12)).sum(axis=0) % 2).astype(np.float32) * 20
    target = np.broadcast_to(500 + checker, (8, 4, 12, 12)).copy()
    wrong_phase = np.broadcast_to(520 - checker, target.shape).copy()
    valid = np.ones((8, 12, 12), dtype=bool)
    exact = score_static_prediction(target, target, valid, valid, valid)
    wrong = score_static_prediction(wrong_phase, target, valid, valid, valid)
    assert exact["all"]["rmse_dn"] == 0
    assert exact["all"]["highpass_rmse_dn"] == 0
    assert wrong["all"]["highpass_rmse_dn"] > 10
    assert wrong["all"]["gradient_rmse_dn"] == 40
    assert wrong["flat"]["residual_sigma_dn"] == 20
    assert wrong["all"]["temporal_mad_dn"] == 0


def test_invalid_neighbors_do_not_pollute_structural_errors() -> None:
    target = np.full((8, 4, 12, 12), 500, dtype=np.float32)
    prediction = target.copy()
    prediction[..., 5, 5] = 3000
    valid = np.ones((8, 12, 12), dtype=bool)
    valid[:, 5, 5] = False
    result = score_static_prediction(prediction, target, valid, valid, valid)
    assert result["all"]["rmse_dn"] == 0
    assert result["all"]["highpass_rmse_dn"] == 0
    assert result["all"]["gradient_rmse_dn"] == 0


def test_grid_filter_preserves_reference_only_ranking() -> None:
    reference = np.full((4, 128, 128), 500, dtype=np.float32)
    valid = np.ones((128, 128), dtype=bool)
    allowed = {(0, 0), (64, 0), (64, 64)}
    rois = select_reference_rois(reference, valid, count=3, grid_size=3, eligible_rois=allowed)
    assert {(roi["y"], roi["x"]) for roi in rois} == allowed
    assert all(roi["size"] == 64 for roi in rois)
    with pytest.raises(ValueError, match="too few"):
        select_reference_rois(reference, valid, count=4, grid_size=3, eligible_rois=allowed)


@pytest.fixture
def prepared_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    monkeypatch.setattr(a14_static_eval, "REPO_ROOT", tmp_path)
    raw_path = tmp_path / "static.raw"
    raw = np.full((16, 128, 128), 400, dtype="<u2")
    raw[1::2] = 600
    raw.tofile(raw_path)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    arrays = {"full_mean_dn.npy": np.full((4, 64, 64), 500, dtype=np.float32),
              "even_mean_dn.npy": np.full((4, 64, 64), 400, dtype=np.float32),
              "odd_mean_dn.npy": np.full((4, 64, 64), 600, dtype=np.float32),
              "valid_region.npy": np.ones((64, 64), dtype=bool),
              "even_confidence.npy": np.ones((64, 64), dtype=np.float32),
              "odd_confidence.npy": np.ones((64, 64), dtype=np.float32),
              "motion_threshold_per_frame.npy": np.full(16, 1000, dtype=np.float32)}
    arrays["odd_confidence.npy"][0] = 0
    outputs = {}
    for name, array in arrays.items():
        path = reference_dir / name
        np.save(path, array)
        outputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    source_hash = sha256_file(raw_path)
    report = {"frames_used": 16, "black_levels": {name: 256 for name in ("R", "Gr", "Gb", "B")},
              "provenance": {"source_path": str(raw_path), "source_sha256": source_hash}, "outputs": outputs}
    (reference_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    source = {"name": "sensor_gain", "sensor": "SC465SL", "path": str(raw_path),
              "width": 128, "height": 128, "frames_used": 16, "bit_shift": 0, "cfa": "RGGB",
              "nominal_black_dn": 256, "reference_mode": "motion_aware_split_half", "clip_starts": [0],
              "fixed_isp_wb": [1, 1, 1, 1], "fixed_isp_exposure": 1}
    config = {"protocol": "unit test", "roi_count": 1, "roi_grid_size": 1, "min_valid_fraction": 0.25,
              "sources": [{"name": "sensor_gain", "role": "train_distribution", "kind": "cross_sensor_split_half",
                           "reference_dir": str(reference_dir), "source_sha256": source_hash, "clip_starts": [0], "source": source}]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    prepared = tmp_path / "artifacts/a14_test/prepared"
    result = prepare_static_cases(config_path, prepared)
    return prepared, config_path, result


def test_preparation_uses_opposite_parity_and_fixed_native_masks(prepared_fixture: tuple[Path, Path, dict]) -> None:
    prepared, _, result = prepared_fixture
    assert len(result["cases"]) == 2
    for parity, case in enumerate(result["cases"]):
        assert case["frame_indices"] == list(range(parity, 16, 2))
        assert not set(case["frame_indices"]) & set(case["target_frame_pool"])
        assert case["target_parity"] == 1 - parity
        with np.load(prepared / case["file"]) as data:
            np.testing.assert_array_equal(data["noisy_dn"], 400 if parity == 0 else 600)
            np.testing.assert_array_equal(data["target_dn"], 600 if parity == 0 else 400)
            assert data["input_normalized"].dtype == np.float32
            assert not data["valid"][:, 0].any()
            assert data["valid"][:, 1:].all()
            assert data["input_normalized"].shape == (8, 4, 64, 64)
        assert sha256_file(prepared / case["file"]) == case["sha256"]
    assert all(result["sources"][0]["region_pixels"].values())


def test_preparation_is_exclusive(prepared_fixture: tuple[Path, Path, dict]) -> None:
    prepared, config, _ = prepared_fixture
    with pytest.raises(FileExistsError):
        prepare_static_cases(config, prepared)


class ScaleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


@pytest.fixture
def checkpoint_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": {"scale": torch.tensor(0.5)}, "ema": {"scale": torch.tensor(0.75)}}, path)
    monkeypatch.setattr(a12_paired_eval, "build_rvideformer", ScaleModel)
    return path


@pytest.mark.parametrize("weights,scale", [("model", 0.5), ("ema", 0.75)])
def test_evaluation_scores_exact_state_and_records_identity(
    tmp_path: Path, prepared_fixture: tuple[Path, Path, dict], checkpoint_fixture: Path, weights: str, scale: float,
) -> None:
    prepared, _, manifest = prepared_fixture
    output = tmp_path / f"artifacts/a14_test/{weights}"
    result = evaluate_static_checkpoint(prepared, checkpoint_fixture, weights, weights, output, "cpu", sha256_file(checkpoint_fixture))
    assert result["identity"]["weights"] == weights
    assert result["identity"]["precision"] == "float32"
    assert result["identity"]["prepared_manifest"]["sha256"] == sha256_file(prepared / "manifest.json")
    saved = np.load(result["cases"][0]["prediction"]["path"])
    np.testing.assert_allclose(saved, (400 - 256) * scale + 256, atol=1e-4)
    summary = result["summary_by_source"]["sensor_gain"]
    assert summary["role"] == "train_distribution"
    assert summary["case_count"] == len(manifest["cases"])
    assert summary["metrics"]["prediction"]["all"]["highpass_rmse_dn"] == 0
    assert json.loads((output / "status.json").read_text())["status"] == "evaluation_completed"
    with pytest.raises(FileExistsError):
        evaluate_static_checkpoint(prepared, checkpoint_fixture, weights, weights, output, "cpu")


def test_missing_ema_is_not_substituted(
    tmp_path: Path, prepared_fixture: tuple[Path, Path, dict], checkpoint_fixture: Path,
) -> None:
    torch.save({"model": {"scale": torch.tensor(0.5)}}, checkpoint_fixture)
    with pytest.raises(ValueError, match="requested ema"):
        evaluate_static_checkpoint(prepared_fixture[0], checkpoint_fixture, "ema", "ema", tmp_path / "artifacts/a14_test/missing", "cpu")


def test_changed_case_fails_before_model_loading(
    tmp_path: Path, prepared_fixture: tuple[Path, Path, dict], checkpoint_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, _, manifest = prepared_fixture
    (prepared / manifest["cases"][0]["file"]).write_bytes(b"changed")
    def forbidden_load(*args: object) -> torch.nn.Module:
        pytest.fail("loading should not occur after an identity failure")
    monkeypatch.setattr(a14_static_eval, "load_selected_model", forbidden_load)
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_static_checkpoint(prepared, checkpoint_fixture, "model", "bad", tmp_path / "artifacts/a14_test/changed", "cpu")


def test_oom_stops_without_retry_or_next_case(
    tmp_path: Path, prepared_fixture: tuple[Path, Path, dict], checkpoint_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []
    class OomModel(ScaleModel):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            called.append(value.shape)
            raise torch.cuda.OutOfMemoryError("simulated out of memory")
    monkeypatch.setattr(a12_paired_eval, "build_rvideformer", OomModel)
    output = tmp_path / "artifacts/a14_test/oom"
    with pytest.raises(torch.cuda.OutOfMemoryError):
        evaluate_static_checkpoint(prepared_fixture[0], checkpoint_fixture, "ema", "oom", output, "cpu")
    assert len(called) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "stopped_oom"
    assert status["completed_cases"] == 0
    assert not (output / "evaluation.json").exists()


def test_cuda_uuid_uses_canonical_gpu_prefix_and_rejects_another_card(
    tmp_path: Path, prepared_fixture: tuple[Path, Path, dict], checkpoint_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_uuid = "07a5046e-9b56-8557-3845-d6cca9ea0497"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(uuid=raw_uuid))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA GeForce RTX 3090")
    def stop_before_allocation(*args: object) -> torch.nn.Module:
        raise RuntimeError("test stop before model allocation")
    monkeypatch.setattr(a14_static_eval, "load_selected_model", stop_before_allocation)
    output = tmp_path / "artifacts/a14_test/gpu_identity"
    with pytest.raises(RuntimeError, match="test stop before model allocation"):
        evaluate_static_checkpoint(prepared_fixture[0], checkpoint_fixture, "ema", "uuid", output, "cuda:0",
                                   expected_gpu_uuid=f"GPU-{raw_uuid}")
    assert json.loads((output / "identity.json").read_text())["gpu_uuid"] == f"GPU-{raw_uuid}"
    wrong_card_output = tmp_path / "artifacts/a14_test/wrong_gpu"
    with pytest.raises(ValueError, match="differs from requested"):
        evaluate_static_checkpoint(prepared_fixture[0], checkpoint_fixture, "ema", "uuid", wrong_card_output, "cuda:0",
                                   expected_gpu_uuid="GPU-9ee78cda-5391-4e9e-15d2-eeca142f4d7b")
    assert not wrong_card_output.exists()


def test_source_summary_pools_squared_errors_and_preserves_roles() -> None:
    target = np.full((8, 4, 12, 12), 500, dtype=np.float32)
    mask = np.ones((8, 12, 12), dtype=bool)
    cases = [{"source": "a", "role": "train_distribution", "metrics": {"prediction": score_static_prediction(target + error, target, mask, mask, mask)}} for error in (1, 3)]
    cases.append({"source": "b", "role": "gain_holdout", "metrics": {"prediction": score_static_prediction(target + 10, target, mask, mask, mask)}})
    summary = summarize_cases(cases)
    assert summary["a"]["metrics"]["prediction"]["all"]["rmse_dn"] == pytest.approx(np.sqrt(5))
    assert summary["a"]["metrics"]["prediction"]["all"]["rmse_dn"] != 2
    assert summary["b"]["role"] == "gain_holdout"
    assert summary["b"]["metrics"]["prediction"]["all"]["rmse_dn"] == 10

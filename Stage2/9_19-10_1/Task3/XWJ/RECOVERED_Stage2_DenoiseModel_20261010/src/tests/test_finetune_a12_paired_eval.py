"""State identity, exact native frame selection, and immutable evaluation outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from raw_fusion.finetune import a3, a4, a12_paired_eval
from raw_fusion.finetune.a12_dynamic_probe import evaluate_probes, make_motion_probe
from raw_fusion.finetune.a12_paired_eval import evaluate_paired_checkpoint, load_selected_model, read_native_window
from raw_fusion.finetune.a12_train import sha256_file


@pytest.fixture
def state_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": {"weight": torch.tensor([[0.25]])}, "ema": {"weight": torch.tensor([[0.75]])}}, path)
    return path


@pytest.mark.parametrize("weights,expected", [("model", 0.25), ("ema", 0.75)])
def test_exact_checkpoint_state_is_selected(
    state_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, weights: str, expected: float,
) -> None:
    monkeypatch.setattr(a12_paired_eval, "build_rvideformer", lambda: torch.nn.Linear(1, 1, bias=False))
    model = load_selected_model(state_checkpoint, weights, torch.device("cpu"))
    assert model.weight.item() == expected
    assert not model.training


def test_missing_ema_is_not_substituted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model_only.pth"
    torch.save({"model": {"weight": torch.ones(1, 1)}}, path)
    monkeypatch.setattr(a12_paired_eval, "build_rvideformer", lambda: torch.nn.Linear(1, 1, bias=False))
    with pytest.raises(ValueError, match="requested ema"):
        load_selected_model(path, "ema", torch.device("cpu"))


@pytest.fixture
def native_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[int]]:
    supervision = tmp_path / "supervision"
    manifests = supervision / "stream_manifests"
    manifests.mkdir(parents=True)
    motion = np.broadcast_to(np.arange(200, dtype=np.float32)[:, None, None, None], (200, 1, 4, 6)).copy()
    np.save(tmp_path / "motion.npy", motion)
    manifest = {
        "group": "04", "scene": "shdarkroom", "gain_label": "1000x", "frames_used": 200,
        "path": "unused.raw", "black_levels": {name: 256 for name in ("R", "Gr", "Gb", "B")},
        "motion": {"mask_path": "motion.npy"},
    }
    (manifests / "04_shdarkroom_1000x.json").write_text(json.dumps(manifest), encoding="utf-8")
    indices = []

    class Reader:
        frames = 200

        def read_frame(self, index: int) -> np.ndarray:
            indices.append(index)
            return np.full((4, 6), (256 + index) << 4, dtype=np.uint16)

    monkeypatch.setattr(a12_paired_eval, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(a12_paired_eval, "open_stream", lambda *_: (Reader(), None))
    return supervision, indices


@pytest.mark.parametrize("start", [0, 96, 192])
def test_native_window_reads_original_frames_and_corresponding_masks(
    native_fixture: tuple[Path, list[int]], start: int,
) -> None:
    supervision, indices = native_fixture
    video, motion, black = read_native_window(supervision, start)
    assert indices == list(range(start, start + 8))
    assert video.shape == (8, 4, 2, 3)
    assert video.dtype == np.float32
    np.testing.assert_allclose(video[:, 0, 0, 0], np.arange(start, start + 8) / 3839.0)
    np.testing.assert_array_equal(motion[:, 0, 0], np.arange(start, start + 8))
    np.testing.assert_array_equal(black, np.full(4, 256))


def test_native_window_does_not_truncate_or_pad(native_fixture: tuple[Path, list[int]]) -> None:
    supervision, indices = native_fixture
    with pytest.raises(ValueError, match="exceeds"):
        read_native_window(supervision, 193)
    assert not indices
    with pytest.raises(ValueError, match="nonnegative"):
        read_native_window(supervision, -1)


@pytest.mark.parametrize("weights,expected", [("model", 0.25), ("ema", 0.75)])
def test_probe_scores_the_requested_state_and_records_its_identity(
    tmp_path: Path, state_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, weights: str, expected: float,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    probe = make_motion_probe(np.full((4, 8, 8), 500.0, dtype=np.float32), np.ones((8, 8), dtype=bool), mode="static")
    case = prepared / "case.npz"
    np.savez(case, noisy_dn=probe.target_dn, target_dn=probe.target_dn, valid=probe.valid,
             motion=probe.motion, static=probe.static, black_dn=np.zeros(4, dtype=np.float32))
    config = {"protocol": "test", "scope": "test fixture"}
    manifest = {"config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "cases": [{"name": "case", "mode": "static", "gain": "100", "file": case.name, "sha256": sha256_file(case)}]}
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(a3, "build_rvideformer", lambda: torch.nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(a4, "infer_video", lambda model, video, *_args, **_kwargs: np.full_like(video, model.weight.item()))
    output = tmp_path / "evaluation"
    baseline = {"name": "candidate", "format": weights, "checkpoint": str(state_checkpoint),
                "checkpoint_sha256": sha256_file(state_checkpoint)}
    result = evaluate_probes(prepared, output, tmp_path, "cpu", baseline)
    assert result["weights"] == weights
    assert result["baseline"]["format"] == weights
    saved = np.load(output / result["cases"][0]["prediction"]["file"])
    np.testing.assert_array_equal(saved, np.full_like(saved, expected * 4095))


def test_phases_share_identity_but_cannot_overwrite(
    tmp_path: Path, state_checkpoint: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(a12_paired_eval, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(a12_paired_eval, "PREPARED_DIR", prepared)

    def probes(_prepared: Path, output: Path, *_args: object) -> dict:
        output.mkdir()
        return {}

    def native(_checkpoint: Path, _weights: str, _label: str, output: Path, *_args: object) -> dict:
        output.mkdir()
        return {}

    monkeypatch.setattr(a12_paired_eval, "evaluate_probes", probes)
    monkeypatch.setattr(a12_paired_eval, "evaluate_native_windows", native)
    output = tmp_path / "artifacts/a12_eval"
    result = evaluate_paired_checkpoint(state_checkpoint, "model", "candidate", output, "cpu", "probes")
    assert result["status"] == "evaluation_completed"
    with pytest.raises(FileExistsError, match="already has output"):
        evaluate_paired_checkpoint(state_checkpoint, "model", "candidate", output, "cpu", "probes")
    with pytest.raises(FileExistsError, match="different"):
        evaluate_paired_checkpoint(state_checkpoint, "ema", "candidate", output, "cpu", "native")
    assert evaluate_paired_checkpoint(state_checkpoint, "model", "candidate", output, "cpu", "native")["status"] == "evaluation_completed"


def test_oom_stops_before_the_next_phase(
    tmp_path: Path, state_checkpoint: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(a12_paired_eval, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(a12_paired_eval, "PREPARED_DIR", prepared)
    called = []

    def fail(*_args: object) -> dict:
        called.append("probes")
        raise torch.cuda.OutOfMemoryError("simulated out of memory")

    monkeypatch.setattr(a12_paired_eval, "evaluate_probes", fail)
    monkeypatch.setattr(a12_paired_eval, "evaluate_native_windows", lambda *_args: called.append("native"))
    output = tmp_path / "artifacts/a12_oom"
    with pytest.raises(torch.cuda.OutOfMemoryError):
        evaluate_paired_checkpoint(state_checkpoint, "ema", "candidate", output, "cpu")
    assert called == ["probes"]
    assert json.loads((output / "status_all.json").read_text())["status"] == "stopped_oom"

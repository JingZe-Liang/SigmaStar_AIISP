"""Reject mixed evaluation identities and preserve per-window paired deltas."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/summarize_a12_paired.py"
SPEC = importlib.util.spec_from_file_location("summarize_a12_paired", SCRIPT)
summary_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_script)


@pytest.fixture
def evaluations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Path]]:
    monkeypatch.setattr(summary_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(summary_script, "NATIVE_SHAPE", (8, 4, 2, 3))
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    prepared_sha = summary_script.sha256(prepared)
    specs, directories = [], {}
    for phase in ("baseline", "shared", "fixed", "refresh"):
        checkpoint = tmp_path / f"{phase}.pth"
        checkpoint.write_bytes(phase.encode())
        checkpoint_sha = summary_script.sha256(checkpoint)
        for weights in (("ema",) if phase == "baseline" else ("model", "ema")):
            label = f"{phase}_{weights}"
            directory = tmp_path / "artifacts" / f"a12_eval_{label}"
            directories[label] = directory
            for kind in ("probes", "native"):
                (directory / kind).mkdir(parents=True)
                np.save(directory / kind / "prediction.npy", np.zeros((8, 4, 2, 3), dtype=np.float32))
            identity = {"label": label, "weights": weights, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                        "gpu_name": "NVIDIA GeForce RTX 3090", "cuda_visible_devices": "GPU-fixture", "device": "cuda:0",
                        "torch_version": "fixture", "cuda_version": "fixture", "prepared_manifest_sha256": prepared_sha,
                        "tile_batch": 1, "native_starts": [0, 96, 192], "native_frames": 8}
            value = {"baseline": 1.0, "shared": 0.9, "fixed": 0.8, "refresh": 0.7}[phase]
            metric = {key: value for key in summary_script.PROBE_ERRORS} | {"pixels": 100, "highpass_pixels": 80}
            cases = [{"name": f"case{index}", "mode": ("static", "global_translation", "moving_patch")[index // 4], "gain": str(index % 2),
                      "prediction": {"file": "prediction.npy", "sha256": summary_script.sha256(directory / "probes/prediction.npy")},
                      "model_input": {"rmse_dn": 3}, label: {"all": metric, "motion_swept": metric}} for index in range(12)]
            probes = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha, "weights": weights, "device": "cuda:0",
                      "baseline": {"name": label, "format": weights, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha},
                      "prepared_manifest": str(prepared), "prepared_manifest_sha256": prepared_sha,
                      "metric_protocol": {"version": "a12_region_errors_v2"}, "cases": cases}
            native = {"label": label, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha, "weights": weights, "device": "cuda:0",
                      "protocol": "fixture", "prediction_dtype": "float32", "tile_batch": 1, "spatial_tile": 64, "spatial_stride": 56,
                      "temporal_frames": 8, "temporal_policy": "independent", "manifest": {}, "raw": {}, "motion_mask": {}, "reference": {}, "valid_region": {},
                      "windows": [{"frame_indices": indices, "input_shape": [8, 4, 2, 3], "input_sha256": f"input{index}", "motion_window_sha256": f"mask{index}",
                                   "black_dn": [256] * 4, "prediction": {"file": "prediction.npy", "sha256": summary_script.sha256(directory / "native/prediction.npy")},
                                   "metrics": {"finetuned": {"rmse_dn": value + index, "psnr_db": 50 - value - index, "pixels": 100},
                                               "input": {"rmse_dn": 3}, "static_pixels": 100, "static_fraction": 0.9}}
                                  for index, indices in enumerate(summary_script.WINDOWS)]}
            for relative, report in (("identity.json", identity), ("probes/evaluation.json", probes), ("native/evaluation.json", native)):
                (directory / relative).write_text(json.dumps(report))
            specs.append({"label": label, "phase": phase, "weights": weights, "output_dir": str(directory)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"specs": specs}))
    return manifest, directories


def test_summary_preserves_all_units_and_both_comparators(evaluations: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    manifest, _ = evaluations
    result = summary_script.summarize(manifest, tmp_path / "artifacts/a12_summary")
    rows = [row for row in result["native_metrics"] if row["label"] == "refresh_ema" and row["metric"] == "finetuned.rmse_dn"]
    assert [row["unit"] for row in rows] == ["0-7", "96-103", "192-199"]
    assert all(row["delta_from_a3"] == pytest.approx(-0.3) and row["delta_from_fixed"] == pytest.approx(-0.1) for row in rows)
    assert len({row["unit"] for row in result["probe_metrics"]}) == 12
    assert result["primary_weights"] == "ema" and result["acceptance_status"] == "diagnostic_only_no_quality_pass_claim"


@pytest.mark.parametrize("change", ["probe_protocol", "window_input", "weights", "device"])
def test_mixed_protocol_or_identity_is_rejected(evaluations: tuple[Path, dict[str, Path]], tmp_path: Path, change: str) -> None:
    manifest, directories = evaluations
    directory = directories["refresh_ema"]
    relative = "probes/evaluation.json" if change == "probe_protocol" else "native/evaluation.json" if change == "window_input" else "identity.json"
    path = directory / relative
    data = json.loads(path.read_text())
    if change == "probe_protocol":
        data["metric_protocol"]["version"] = "different"
    elif change == "window_input":
        data["windows"][1]["input_sha256"] = "different"
    elif change == "weights":
        data["weights"] = "model"
    else:
        data["cuda_visible_devices"] = "GPU-other"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summary_script.summarize(manifest, tmp_path / "artifacts/a12_rejected")
    assert not (tmp_path / "artifacts/a12_rejected").exists()


def test_changed_prediction_bytes_are_rejected(evaluations: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    manifest, directories = evaluations
    path = directories["refresh_ema"] / "native/prediction.npy"
    prediction = np.load(path)
    prediction.flat[0] = 1
    np.save(path, prediction)
    with pytest.raises(ValueError, match="prediction changed"):
        summary_script.summarize(manifest, tmp_path / "artifacts/a12_rejected")

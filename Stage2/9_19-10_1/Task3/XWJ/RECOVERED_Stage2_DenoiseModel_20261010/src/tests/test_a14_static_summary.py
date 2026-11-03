"""Adversarial checks for the frozen A14 comparison and acceptance rules."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from raw_fusion.finetune.a14_static_summary import rank_passing, score_source, validate_comparable
from raw_fusion.finetune import a14_static_summary as summary_module


def _summary(psnr=35.0, hp=2.0, gradient=3.0, sigma=1.0):
    return {"metrics": {"prediction": {
        "all": {"pixels": 4096, "psnr_db": psnr, "highpass_rmse_dn": hp, "gradient_rmse_dn": gradient},
        "flat": {"pixels": 1024, "residual_sigma_dn": sigma},
    }}}


def test_training_source_requires_denoising_and_no_structure_regression():
    baseline = _summary()
    assert score_source(baseline, _summary(psnr=35.05, hp=2.02, gradient=3.03, sigma=1.01),
                        "sc635hai_131x_low_cct")["passed"]
    assert not score_source(baseline, _summary(psnr=35.049), "sc635hai_131x_low_cct")["passed"]
    result = score_source(baseline, _summary(psnr=35.5, hp=2.021), "sc635hai_131x_low_cct")
    assert not result["passed"]
    assert "highpass_rmse_dn_regression_or_undefined_ratio" in result["failures"]


def test_holdout_tolerance_and_missing_required_metric():
    baseline = _summary()
    assert score_source(baseline, _summary(psnr=34.97), "mis20s1_1000x")["passed"]
    assert not score_source(baseline, _summary(psnr=34.969), "mis20s1_1000x")["passed"]
    for value in (None, float("nan"), float("inf")):
        result = score_source(baseline, _summary(psnr=35.1, hp=value), "mis20s1_1000x")
        assert not result["passed"]
        assert result["candidate_highpass_rmse_dn"] is None
    no_flat = _summary(psnr=35.1)
    no_flat["metrics"]["prediction"]["flat"]["pixels"] = 0
    assert not score_source(baseline, no_flat, "mis20s1_1000x")["passed"]


def test_zero_error_ratio_is_not_divided_or_silently_accepted():
    baseline = _summary(hp=0.0)
    assert score_source(baseline, _summary(psnr=35.1, hp=0.0), "sc635hai_131x_low_cct")["passed"]
    assert not score_source(baseline, _summary(psnr=35.1, hp=0.01), "sc635hai_131x_low_cct")["passed"]


def test_rank_uses_worst_training_gain_and_then_earliest_step():
    def candidate(step, gain, passed=True):
        return {"step": step, "weights": "ema", "minimum_training_psnr_gain_db": gain,
                "passed": passed, "label": str(step), "checkpoint": {"path": str(step)}}
    result = rank_passing([candidate(900, 0.2), candidate(600, 0.2), candidate(300, 0.1), candidate(300, 0.4, False)])
    assert result["step"] == 600
    assert rank_passing([candidate(300, 0.1, False)]) is None


def test_comparison_rejects_different_inputs_masks_runtime_or_roles():
    keys = ("schema", "prepared_manifest", "device", "gpu_uuid", "gpu_name", "cuda_visible_devices",
            "torch_version", "cuda_version", "precision", "tf32", "output_clip", "code", "source_roles")
    baseline = {"identity": {key: key for key in keys}, "metric_protocol": "same",
                "cases": [{"name": "one", "source": "sensor", "role": "gain_holdout",
                           "prepared_case_sha256": "same", "metrics": {"raw_input": {"x": 1}, "model_input": {"x": 2}}}]}
    validate_comparable(baseline, deepcopy(baseline))
    for key in keys:
        changed = deepcopy(baseline)
        changed["identity"][key] = "changed"
        with pytest.raises(ValueError):
            validate_comparable(baseline, changed)
    for key in ("name", "source", "role", "prepared_case_sha256"):
        changed = deepcopy(baseline)
        changed["cases"][0][key] = "changed"
        with pytest.raises(ValueError):
            validate_comparable(baseline, changed)
    changed = deepcopy(baseline)
    changed["cases"][0]["metrics"]["model_input"]["x"] = 3
    with pytest.raises(ValueError):
        validate_comparable(baseline, changed)


def test_full_six_state_summary_pools_sources_and_writes_reviewable_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(summary_module, "REPO_ROOT", tmp_path)
    real_sha = summary_module.sha256_file
    monkeypatch.setattr(summary_module, "sha256_file", lambda path: summary_module.A3_STEP5000_SHA256
                        if Path(path).name == "baseline.pth" else real_sha(path))
    monkeypatch.setattr(summary_module, "_checkpoint_metadata", lambda identity, cache: {
        "step": int(identity["checkpoint"]["path"]), "config": {"steps": 900},
        "plan_sha256": "same_plan", "audit_sha256": "same_audit"})
    stats = {"count": 4, "error_sum": 0.0, "squared_error_sum": 400.0,
             "highpass_squared_sum": 4.0, "highpass_count": 4,
             "gradient_squared_sum": 4.0, "gradient_count": 4,
             "temporal_error_squared_sum": 4.0, "temporal_abs_sum": 4.0, "temporal_count": 4}
    region_metrics = {region: {"statistics": deepcopy(stats)} for region in ("all", "flat", "detail")}
    identity = {"schema": summary_module.SCHEMA, "source_roles": summary_module.SOURCE_ROLES,
                "precision": "float32", "tf32": False, "output_clip": [0.0, 1.0],
                "gpu_uuid": summary_module.GPU_UUID, "gpu_name": "3090", "device": "cuda:0",
                "cuda_visible_devices": summary_module.GPU_UUID, "torch_version": "same", "cuda_version": "same",
                "code": "same", "prepared_manifest": {"sha256": "same"}, "label": "a3", "weights": "ema",
                "checkpoint": {"path": "baseline.pth", "sha256": summary_module.A3_STEP5000_SHA256}}
    baseline = {"identity": identity, "metric_protocol": "same", "cases": []}
    for source, role in summary_module.SOURCE_ROLES.items():
        for index in range(18):
            baseline["cases"].append({"name": f"{source}_{index}", "source": source, "role": role,
                                      "prepared_case_sha256": f"{source}_{index}_hash",
                                      "metrics": {variant: deepcopy(region_metrics) for variant in ("raw_input", "model_input", "prediction")}})
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    paths = []
    for step, weights in sorted(summary_module.EXPECTED_STATES):
        candidate = deepcopy(baseline)
        candidate["identity"].update(label=f"{step}_{weights}", weights=weights,
                                     checkpoint={"path": str(step), "sha256": f"checkpoint_{step}"})
        scale = {300: 0.99, 600: 0.97, 900: 0.95}[step]
        for case in candidate["cases"]:
            for metrics in case["metrics"]["prediction"].values():
                for key in ("squared_error_sum", "highpass_squared_sum", "gradient_squared_sum"):
                    metrics["statistics"][key] *= scale
        path = tmp_path / f"{step}_{weights}.json"
        path.write_text(json.dumps(candidate))
        paths.append(path)
    output = tmp_path / "artifacts/a14_test/summary"
    result = summary_module.summarize_evaluations(baseline_path, paths, output)
    assert result["status"] == "roi_screen_passed"
    assert result["selected"]["step"] == 900
    assert len(result["candidates"]) == 6
    assert not result["candidates"][0]["passed"]
    assert len(result["candidates"][0]["source_results"]) == 4
    assert json.loads((output / "summary.json").read_text())["selected"] == result["selected"]
    with (output / "rows.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 24
    with pytest.raises(ValueError, match="all six candidate states"):
        summary_module.summarize_evaluations(baseline_path, paths[:-1], tmp_path / "artifacts/a14_test/partial")

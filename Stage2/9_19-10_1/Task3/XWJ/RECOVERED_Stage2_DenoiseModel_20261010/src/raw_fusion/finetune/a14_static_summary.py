"""Compare all frozen A14 states using the predeclared per-source criteria."""

import csv
import json
import math
from pathlib import Path

import torch

from raw_fusion.finetune.a12_train import A3_STEP5000_SHA256, sha256_file
from raw_fusion.finetune.a14_static_eval import SCHEMA, summarize_cases


REPO_ROOT = Path(__file__).resolve().parents[3]
GPU_UUID = "GPU-07a5046e-9b56-8557-3845-d6cca9ea0497"
SOURCE_ROLES = {
    "mis20s1_1000x": "gain_holdout",
    "sc465sl_2000x": "gain_holdout",
    "sc465sl_darkroom18_256x": "train_distribution",
    "sc635hai_131x_low_cct": "train_distribution",
}
EXPECTED_STATES = {(step, weights) for step in (300, 600, 900) for weights in ("model", "ema")}
COMPARABLE_IDENTITY = ("schema", "prepared_manifest", "device", "gpu_uuid", "gpu_name",
                       "cuda_visible_devices", "torch_version", "cuda_version", "precision",
                       "tf32", "output_clip", "code", "source_roles")


def _load_evaluation(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    identity = result["identity"]
    if identity["schema"] != SCHEMA or identity["source_roles"] != SOURCE_ROLES:
        raise ValueError("evaluation does not contain the four frozen A14 source roles")
    if (identity["precision"] != "float32" or identity["tf32"] is not False
            or identity["output_clip"] != [0.0, 1.0] or identity["gpu_uuid"] != GPU_UUID):
        raise ValueError("evaluation must use the authorized GPU and frozen FP32 inference")
    cases = result["cases"]
    if len(cases) != 72 or len({case["name"] for case in cases}) != 72:
        raise ValueError("evaluation must have exactly 72 unique frozen cases")
    for source, role in SOURCE_ROLES.items():
        source_cases = [case for case in cases if case["source"] == source]
        if len(source_cases) != 18 or any(case["role"] != role for case in source_cases):
            raise ValueError("evaluation source case count or role differs from the protocol")
    for case in cases:
        if set(case["metrics"]) != {"raw_input", "model_input", "prediction"}:
            raise ValueError("every case must include both input baselines and a prediction")
    return result


def validate_comparable(baseline: dict, candidate: dict) -> None:
    """Reject comparisons whose cached inputs, model-independent metrics, or runtime differ."""

    left, right = baseline["identity"], candidate["identity"]
    for key in COMPARABLE_IDENTITY:
        if left[key] != right[key]:
            raise ValueError(f"candidate differs from baseline in {key}")
    if baseline["metric_protocol"] != candidate["metric_protocol"]:
        raise ValueError("metric protocol changed between baseline and candidate")
    if len(baseline["cases"]) != len(candidate["cases"]):
        raise ValueError("candidate case count differs from baseline")
    for original, current in zip(baseline["cases"], candidate["cases"], strict=True):
        for key in ("name", "source", "role", "prepared_case_sha256"):
            if original[key] != current[key]:
                raise ValueError(f"candidate frozen case differs in {key}")
        for variant in ("raw_input", "model_input"):
            if original["metrics"][variant] != current["metrics"][variant]:
                raise ValueError(f"candidate {variant} metrics differ from baseline")


def _checkpoint_metadata(identity: dict, cache: dict) -> dict:
    checkpoint = identity["checkpoint"]
    path = Path(checkpoint["path"]).resolve()
    digest = checkpoint["sha256"]
    if digest not in cache:
        if sha256_file(path) != digest:
            raise ValueError(f"evaluated checkpoint identity changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("schema") != "a14_static_training_v1"
                or payload.get("source_checkpoint_sha256") != A3_STEP5000_SHA256):
            raise ValueError("candidate is not an A14 checkpoint initialized from frozen A3")
        cache[digest] = {"step": payload["step"], "config": payload["config"],
                         "plan_sha256": payload["plan_sha256"], "audit_sha256": payload["audit_sha256"]}
        del payload
    return cache[digest]


def _required(metrics: dict, region: str, key: str) -> float | None:
    entry = metrics[region]
    value = entry.get(key)
    if entry.get("pixels", 0) <= 0 or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def score_source(baseline: dict, candidate: dict, source: str) -> dict:
    """Apply thresholds to pooled errors for one source, without averaging sensors."""

    role = SOURCE_ROLES[source]
    before, after = baseline["metrics"]["prediction"], candidate["metrics"]["prediction"]
    values = {}
    failures = []
    for label, region, metric in (("psnr_db", "all", "psnr_db"),
                                   ("highpass_rmse_dn", "all", "highpass_rmse_dn"),
                                   ("gradient_rmse_dn", "all", "gradient_rmse_dn"),
                                   ("flat_residual_sigma_dn", "flat", "residual_sigma_dn")):
        previous, current = _required(before, region, metric), _required(after, region, metric)
        values[f"baseline_{label}"] = previous
        values[f"candidate_{label}"] = current
        if previous is None or current is None:
            failures.append(f"missing_or_nonfinite_{label}")
            values[f"{label}_delta" if label == "psnr_db" else f"{label}_ratio"] = None
            continue
        if label == "psnr_db":
            delta = current - previous
            values["psnr_db_delta"] = delta
            if delta + 1e-10 < (0.05 if role == "train_distribution" else -0.03):
                failures.append("insufficient_psnr_improvement" if role == "train_distribution" else "holdout_psnr_regression")
        else:
            if previous < 0 or current < 0:
                ratio = None
            elif previous == 0:
                ratio = 1.0 if current == 0 else None
            else:
                ratio = current / previous
            if ratio is not None and not math.isfinite(ratio):
                ratio = None
            values[f"{label}_ratio"] = ratio
            if ratio is None or ratio > 1.01 + 1e-10:
                failures.append(f"{label}_regression_or_undefined_ratio")
    return {"source": source, "role": role, **values, "passed": not failures, "failures": failures}


def rank_passing(candidates: list[dict]) -> dict | None:
    passing = [candidate for candidate in candidates if candidate["passed"]]
    if not passing:
        return None
    selected = min(passing, key=lambda candidate: (-candidate["minimum_training_psnr_gain_db"],
                                                   candidate["step"], candidate["weights"]))
    return {key: selected[key] for key in ("label", "step", "weights", "checkpoint", "minimum_training_psnr_gain_db")}


def _finite_json(value: object) -> object:
    """Keep invalid diagnostic values visible as null after the explicit metric gate fails."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def summarize_evaluations(baseline_path: Path, candidate_paths: list[Path], output_dir: Path) -> dict:
    """Validate all seven states, then write reviewable JSON and per-source CSV rows."""

    output = output_dir.resolve()
    relative = output.relative_to((REPO_ROOT / "artifacts").resolve())
    if not relative.parts or not relative.parts[0].startswith("a14_") or output.exists():
        raise ValueError("summary requires a new directory inside artifacts/a14_*")
    baseline = _load_evaluation(baseline_path)
    if (baseline["identity"]["weights"] != "ema"
            or baseline["identity"]["checkpoint"]["sha256"] != A3_STEP5000_SHA256):
        raise ValueError("baseline must be the original A3 step-5000 EMA")
    if sha256_file(Path(baseline["identity"]["checkpoint"]["path"])) != A3_STEP5000_SHA256:
        raise ValueError("original A3 baseline checkpoint identity changed")
    base_summary = summarize_cases(baseline["cases"])
    candidates, cache, seen = [], {}, set()
    shared_training_identity = None
    for path in candidate_paths:
        candidate = _load_evaluation(path)
        validate_comparable(baseline, candidate)
        identity = candidate["identity"]
        metadata = _checkpoint_metadata(identity, cache)
        state = (metadata["step"], identity["weights"])
        if state not in EXPECTED_STATES or state in seen:
            raise ValueError("candidate state is duplicate or outside the three frozen milestones")
        seen.add(state)
        training_identity = {key: metadata[key] for key in ("config", "plan_sha256", "audit_sha256")}
        if shared_training_identity is None:
            shared_training_identity = training_identity
        elif shared_training_identity != training_identity:
            raise ValueError("candidates do not share the same training configuration, sample plan and audit")
        pooled = summarize_cases(candidate["cases"])
        rows = [score_source(base_summary[source], pooled[source], source) for source in SOURCE_ROLES]
        training_gains = [row["psnr_db_delta"] for row in rows if row["role"] == "train_distribution"]
        candidates.append({"label": identity["label"], "step": state[0], "weights": state[1],
                           "checkpoint": identity["checkpoint"], "evaluation": {"path": str(path.resolve()), "sha256": sha256_file(path)},
                           "passed": all(row["passed"] for row in rows),
                           "minimum_training_psnr_gain_db": min(training_gains) if all(value is not None for value in training_gains) else None,
                           "source_results": rows, "pooled_metrics_by_source": pooled})
    if seen != EXPECTED_STATES:
        raise ValueError(f"all six candidate states are required; missing {sorted(EXPECTED_STATES - seen)}")
    for step in (300, 600, 900):
        if len({candidate["checkpoint"]["sha256"] for candidate in candidates if candidate["step"] == step}) != 1:
            raise ValueError("live and EMA at one milestone must come from the identical checkpoint")
    candidates.sort(key=lambda item: (item["step"], item["weights"]))
    selected = rank_passing(candidates)
    result = {"schema": "a14_static_comparison_v1", "status": "roi_screen_passed" if selected else "roi_screen_failed",
              "baseline": {"evaluation": str(baseline_path.resolve()), "evaluation_sha256": sha256_file(baseline_path),
                           "identity": baseline["identity"], "pooled_metrics_by_source": base_summary},
              "criteria": {"aggregation": "Pool sufficient statistics within each source; never average across sources.",
                           "training_source_minimum_psnr_gain_db": 0.05, "holdout_minimum_psnr_gain_db": -0.03,
                           "maximum_all_highpass_rmse_ratio": 1.01, "maximum_all_gradient_rmse_ratio": 1.01,
                           "maximum_flat_residual_sigma_ratio": 1.01,
                           "missing_or_nonfinite_required_metric": "fail",
                           "ranking": "Maximize the smaller PSNR gain of the two training sources; ties prefer earlier step, then stable alphabetical state ordering."},
              "candidates": candidates, "selected": selected,
              "limitations": ["ROI screening only; passing still requires larger-field and image comparison before static candidate delivery.",
                              "All four sources are previously observed scenes; train-distribution diagnostics and gain holdouts are identified separately.",
                              "Static references have residual noise, shared fixed-pattern error and possible drift; this is not an unseen-sensor or dynamic quality claim.",
                              "No default inference weights are changed by this summary."]}
    result = _finite_json(result)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    rows = [{"label": candidate["label"], "step": candidate["step"], "weights": candidate["weights"],
             "candidate_passed": candidate["passed"], **row, "failures": "|".join(row["failures"])}
            for candidate in candidates for row in candidate["source_results"]]
    with (output / "rows.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result

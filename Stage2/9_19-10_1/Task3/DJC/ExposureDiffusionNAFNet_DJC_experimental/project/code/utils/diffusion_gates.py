"""Evidence-bound stage transitions. Missing visual/noise evidence never passes."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def h5_gate(conditions, target_cap_ratio):
    reasons = []
    if not conditions or not all(math.isfinite(c.get("gain_db", float("nan"))) for c in conditions):
        reasons.append("missing/nonfinite RGB measurements")
    else:
        if sum(c["gain_db"] for c in conditions) / len(conditions) < 1.:
            reasons.append("mean RGB improvement below 1 dB")
        if min(c["gain_db"] for c in conditions) < -.2:
            reasons.append("a scene/noise condition regresses more than 0.2 dB")
        for scene in sorted({c["scene"] for c in conditions}):
            for seed in sorted({c["seed"] for c in conditions}):
                entries = [c for c in conditions if c["scene"] == scene and c["seed"] == seed]
                if any(not math.isfinite(c.get("temporal_gain_db", float("nan"))) for c in entries) or sum(c["temporal_gain_db"] for c in entries) / len(entries) <= 0:
                    reasons.append(f"no repeatable neighbor benefit: {scene}, seed {seed}")
        coverage = {(c["scene"], c["seed"], c["start_t"]) for c in conditions}
        expected = {(s, seed, t) for s in ("scene_9", "scene_10") for seed in (7, 17, 27) for t in (0, 4, 15)}
        if coverage != expected or any(c.get("patches", 0) < 128 for c in conditions):
            reasons.append("requires both scenes, three seeds/noise levels and 128 patches/condition")
    if target_cap_ratio != 0:
        reasons.append("clean GT changed")
    return {"numeric_passed": not reasons, "reasons": reasons,
            "visual_review": "pending", "acceptance_passed": False}


def require_gate(path, checkpoint, stage):
    report = json.loads(Path(path).read_text())
    if (report.get("stage") != stage or report.get("checkpoint_sha256") != sha256(checkpoint)
            or not report.get("numeric_passed") or (stage != "smoke" and report.get("visual_review") != "passed")
            or not report.get("acceptance_passed")):
        raise ValueError(f"STOP: missing/pending/failed or mismatched {stage} gate: {path}")
    return report


def company_gate(report):
    """Evaluation must supply paired risks, uncertainty and actual visual evidence."""
    reasons = []
    if not report.get("static_assumptions_passed"):
        reasons.append("static/independent-target assumptions unverified")
    groups = report.get("regions", [])
    if not groups or not report.get("enough_independent_frame_groups"):
        reasons.append("insufficient region/frame-group evidence")
    for region in groups:
        for key in ("paired_risk_before", "paired_risk_after", "improvement_ci_low",
                    "gradient_regression_ci_low", "color_regression_ci_low"):
            if not math.isfinite(region.get(key, float("nan"))):
                reasons.append(f"missing/nonfinite {key}")
        before, after = region.get("paired_risk_before", 0), region.get("paired_risk_after", float("inf"))
        if before <= 0 or (before - after) / before < .02 or region.get("improvement_ci_low", -1) <= 0:
            reasons.append("no confirmed 2% paired-risk improvement")
        if region.get("gradient_regression_ci_low", float("inf")) > 0 or region.get("color_regression_ci_low", float("inf")) > 0:
            reasons.append("confirmed gradient/color regression")
    if not report.get("repeat_seeds_consistent"):
        reasons.append("training-seed repeatability unverified")
    if report.get("visual_review") != "passed":
        reasons.append("full-frame/motion visual review pending or failed")
    return {**report, "numeric_passed": not reasons, "reasons": reasons,
            "acceptance_passed": not reasons}


def validate_frozen_prior_weight_transfer(prior, checkpoint_sha, metadata, source_sha=None):
    """Diagnostic only: keep the old covariance bound to its original weights."""
    if source_sha is None:
        if prior["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("prior covariance belongs to different weights")
        return False
    if (len(source_sha) != 64 or prior.get("diagnostic_only") is not True or
            prior["checkpoint_sha256"] != source_sha or
            metadata.get("phase") != "company-exploratory" or metadata.get("step") != 250 or
            metadata.get("provenance", {}).get("source_checkpoint_sha256") != source_sha or
            metadata.get("provenance", {}).get("initial_checkpoint_sha256") != source_sha or
            checkpoint_sha == source_sha):
        raise ValueError("invalid frozen prior weight transfer diagnostic")
    return True


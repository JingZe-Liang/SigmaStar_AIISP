"""Recursive exact-schema primitives used by every V2 loader."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

FUSION_PROTOCOL = "md_conditioned_frequency_fusion_v2"
CONTRACT_HASH_KEYS = (
    "raw", "b2", "morphology", "exact_stats", "bootstrap", "model", "selector",
    "composer", "md_protection", "noise_estimator", "sampler", "quantization",
    "evaluator", "cli",
)


class ContractError(ValueError):
    """Raised when a V2 contract, mapping, or artifact is invalid."""


class CliContractError(ContractError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str


def require_exact_keys(value: Mapping[str, object], expected: frozenset[str] | set[str], context: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context}: expected mapping")
    actual = frozenset(value)
    expected = frozenset(expected)
    missing = ",".join(sorted(expected - actual))
    unknown = ",".join(sorted(actual - expected))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing keys: {missing}")
        if unknown:
            detail.append(f"unknown keys: {unknown}")
        raise ContractError(f"{context}: {'; '.join(detail)}")


def require_exact_mapping(
    value: object,
    expected: frozenset[str] | set[str],
    context: str = "value",
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{context}: expected JSON object")
    require_exact_keys(value, expected, context)
    return value


def _fail(context: str, message: str) -> None:
    raise ContractError(f"{context}: {message}")


def expect_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(context, "expected mapping")
    return value


def expect_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        _fail(context, "expected string")
    return value


def expect_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(context, "expected integer")
    return value


def expect_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(context, "expected boolean")
    return value


def expect_number(value: object, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, "expected finite number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        _fail(context, "expected finite number")
    return value


def validate_list(value: object, item_validator, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(context, "expected list")
    return [item_validator(item, f"{context}[{index}]") for index, item in enumerate(value)]


def validate_exact(value: object, expected: set[str] | frozenset[str], context: str, *, field_types: Mapping[str, str] | None = None) -> Mapping[str, object]:
    mapping = expect_mapping(value, context)
    require_exact_keys(mapping, expected, context)
    if field_types:
        for key, kind in field_types.items():
            item = mapping[key]
            item_context = f"{context}.{key}"
            if kind == "string": expect_string(item, item_context)
            elif kind == "int": expect_int(item, item_context)
            elif kind == "bool": expect_bool(item, item_context)
            elif kind == "number": expect_number(item, item_context)
            elif kind == "mapping": expect_mapping(item, item_context)
            elif kind == "list" and not isinstance(item, list): _fail(item_context, "expected list")
    return mapping


def validate_protocol(value: object, expected: str, context: str) -> None:
    if value != expected:
        _fail(context, f"expected protocol {expected!r}")


def reject_duplicate_rows(rows: Sequence[Mapping[str, object]], keys: tuple[str, ...], context: str) -> None:
    seen: set[tuple[object, ...]] = set()
    for index, row in enumerate(rows):
        marker = tuple(row.get(key) for key in keys)
        try:
            duplicate = marker in seen
            seen.add(marker)
        except TypeError:
            _fail(f"{context}[{index}]", "logical row key is not hashable")
        if duplicate:
            _fail(f"{context}[{index}]", f"duplicate logical row: {marker}")


def validate_artifact_mapping(value: object, context: str = "ArtifactRef") -> Mapping[str, object]:
    mapping = validate_exact(value, {"path", "sha256"}, context)
    expect_string(mapping["path"], f"{context}.path")
    digest = expect_string(mapping["sha256"], f"{context}.sha256")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        _fail(f"{context}.sha256", "must be lowercase SHA-256")
    return mapping


def validate_array_mapping(value: object, context: str = "ArrayRef") -> Mapping[str, object]:
    mapping = validate_exact(value, {"path", "sha256", "dtype", "shape"}, context)
    validate_artifact_mapping({"path": mapping["path"], "sha256": mapping["sha256"]}, context)
    expect_string(mapping["dtype"], f"{context}.dtype")
    shape = mapping["shape"]
    if not isinstance(shape, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape):
        _fail(f"{context}.shape", "must be a list of non-negative integers")
    return mapping


def validate_raw_asset_mapping(value: object, context: str = "RawAsset") -> Mapping[str, object]:
    mapping = validate_exact(value, {"path", "sha256", "frame_count"}, context)
    validate_artifact_mapping({"path": mapping["path"], "sha256": mapping["sha256"]}, context)
    expect_int(mapping["frame_count"], f"{context}.frame_count")
    if mapping["frame_count"] < 0:
        _fail(f"{context}.frame_count", "must be non-negative")
    return mapping


_GENERIC_EXPECTED_KEYS: dict[str, frozenset[str]] = {
    # P8b/P11 metric and gate records.  These are deliberately kept as
    # schema-level key sets here; numeric/enumeration semantics belong to the
    # owning metric/hard-stop task, but an unknown or missing persisted key is
    # never acceptable at the protocol boundary.
    "MetricIdV2": frozenset(),
    "MetricSubjectV2": frozenset({"kind", "id", "variant", "model_state_sha256"}),
    "MetricValueV2": frozenset({"metric_id", "status", "value", "numerator", "denominator", "unit", "reason"}),
    "MetricSummaryV2": frozenset({"metric_id", "reduction", "sample_count", "value", "numerator", "denominator", "unit", "bootstrap", "evidence_frames", "status", "reason"}),
    "MetricFrameV2": frozenset({"schema_version", "protocol", "subject", "condition", "partition", "source_frame", "output_frame_sha256", "values"}),
    "MetricPartitionAggregateV2": frozenset({"partition", "source_frames", "summaries"}),
    "MetricAggregateV2": frozenset({"schema_version", "protocol", "subject", "condition", "partitions"}),
    "LabelStabilityReportV2": frozenset({"schema_version", "protocol", "kind", "label_bundle", "bootstrap_contract", "condition_results", "valid"}),
    "LabelStabilityConditionV2": frozenset({"schema_version", "protocol", "condition", "kind", "shard_key", "bootstrap_seed_record", "block_count", "evaluated_cell_count", "flip_count", "flip_rate", "unknown_count", "unknown_rate", "invalid_reason", "valid"}),
    "CheckpointSafetyConditionV2": frozenset({"condition", "applicable_rules", "rule_results", "passed", "failures"}),
    "CheckpointSafetyCandidateV2": frozenset({"epoch", "global_step", "model_state_sha256", "applicable_rules", "condition_results", "selection_score", "worse_condition", "passed", "failures"}),
    "CheckpointSafetyReportV2": frozenset({"schema_version", "protocol", "experiment", "evaluator_contract", "applicable_rules", "candidates", "selected", "passed", "failure_reason"}),
    "PseudoGtAuxiliaryV2": frozenset({"schema_version", "protocol", "condition", "frame_range", "pattern", "source_root", "raw_contract", "frame_assets"}),
    "DeliveryManifestV2": frozenset({"schema_version", "protocol", "generator", "evaluation_result", "audit_bundle", "smoke_result", "inference_results", "isp_contract", "fps", "conditions"}),
    "DeliveryConditionV2": frozenset({"frame_count", "source_frame_range", "label_overlay_archive", "label_coverage_report", "gev_consistency_report", "q_map_correct", "q_map_swapped", "q_map_mean", "metrics_json", "fusion_video", "comparison_video", "fallback_timeline", "estimator_timeline"}),
    "InferenceFrameLineV2": frozenset({"ordinal", "frame_id", "timestamp_ns", "stream_reset"}),
    "InferenceResultLineV2": frozenset({"ordinal", "frame_id", "output_source", "fallback_reason", "state", "q_mean", "q_cell_zero_fraction", "injected_abs_mean", "lp_diff_rms_dn", "lp_diff_p99_dn", "output_frame_sha256"}),
    "SeedTimelineRefV2": frozenset({"condition", "file", "line_count", "row_schema"}),
    "StateTimelineLineV2": frozenset({"frame_index", "state_before", "state_after", "transition_reason", "md_instance_epoch", "md_ready", "mask_frame_sha256", "candidate_valid", "candidate_reject_reason", "static_fraction", "valid_pixels_r", "valid_pixels_gr", "valid_pixels_b", "valid_pixels_gb", "sigma_candidate", "c_raw", "c_ema", "c_model", "c_tilde", "ood_distance_128x", "ood_distance_645x", "jump_distance", "valid_streak", "invalid_streak", "model_eligible", "reset_after_output"}),
    "ResetLineV2": frozenset({"frame_index", "reason", "applied_after_output", "next_frame_md_instance_epoch"}),
    "SeedTimelineLineV2": frozenset({"condition", "frame_index", "md_ready", "mask_frame_sha256", "input_frame_sha256", "affine_a", "affine_b", "shift_yx", "lowpass_correlation", "diff_energy", "diff_rms_r", "diff_rms_gr", "diff_rms_b", "diff_rms_gb", "static_fraction", "cfa_ratio_r", "cfa_ratio_gr", "cfa_ratio_b", "cfa_ratio_gb", "valid_pixels_r", "valid_pixels_gr", "valid_pixels_b", "valid_pixels_gb", "sigma_candidate", "c_raw", "candidate_valid", "candidate_reject_reason", "calibration_seed"}),
    "PowerThresholdV1": frozenset({"target_frame_count", "sample_count", "p95", "p99", "dtype", "unit"}),
    "BootstrapGoldenV1": frozenset({"stream_id", "n", "replicate", "draws", "indices"}),
    "BootstrapSeedRecordV1": frozenset({"algorithm", "material", "digest", "descriptors", "stream_ids", "golden_indices_32"}),
    "SupportEvidenceV1": frozenset({"target_condition", "split", "g_frames", "support_10x", "support_64x", "support_313x"}),
    "LabelShardV2": frozenset({"condition", "split", "target_frames", "row_order", "row_count", "arrays", "frame_meta", "summary"}),
    "FrameMetaLineV2": frozenset({"condition", "split", "source_frame", "row_index", "state", "valid_bits", "alpha_class", "confidence"}),
    "PoolSetV2": frozenset({"texture", "flat", "motion_boundary", "random_unknown"}),
    "PoolRefV2": frozenset({"file", "count", "row_schema"}),
    "PoolRowV2": frozenset({"pool_index", "condition", "split", "category", "source_frame", "cell_y", "cell_x", "label_row_index"}),
    "SamplerRowV2": frozenset({"epoch", "step_in_epoch", "batch_slot", "condition", "split", "category", "pool_index", "source_frame", "cell_y", "cell_x", "label_row_index", "augmentation"}),
    "CheckpointV2": frozenset({"schema_version", "protocol", "model_state_dict", "optimizer_state_dict", "amp_scaler_state_dict", "epoch", "global_step", "seed", "artifact_hashes", "contract_hashes", "noise_condition_state", "contract_fingerprint_sha256"}),
    "InferenceInputV2": frozenset({"schema_version", "protocol", "dataset_config", "estimator_bundle", "raw_contract_hash", "stream_id", "frame_range", "frame_count", "inputs", "md", "frame_identity"}),
    "InferenceResultV2": frozenset({"schema_version", "protocol", "input", "experiment", "checkpoint", "output_raw", "frame_manifest", "state_timeline", "summary"}),
    "SmokeResultV2": frozenset({"schema_version", "protocol", "experiment", "status", "validation", "synthetic_checks", "real_data_probe", "inference_probe", "artifacts"}),
    "ValidationReportV2": frozenset({"schema_version", "protocol", "subject_protocol", "subject", "validator", "valid", "errors", "warnings", "checked_artifacts"}),
    "TrainRunV2": frozenset({"schema_version", "protocol", "experiment", "sampler_manifest", "ablation", "resume_from", "status", "best_checkpoint", "last_checkpoint", "train_log", "selection_report"}),
    "ExperimentV2": frozenset({"schema_version", "protocol", "dataset", "artifacts", "ablation", "contract_hashes", "model", "loss", "train", "inference", "selection"}),
    "SamplerConfigV2": frozenset({"schema_version", "protocol", "sampler_contract", "seed", "epochs", "batch_size", "samples_per_epoch", "quota_per_condition", "replacement", "pool_order", "condition_order", "category_order", "augmentation_order", "batch_interleave"}),
    "ManualMotionManifestV2": frozenset({"schema_version", "protocol", "generator", "dataset_config", "source_previews", "annotation_guide", "windows", "entries"}),
    "MdSeedBundleV2": frozenset({"schema_version", "protocol", "generator", "opencv_version", "dataset_config", "common_parameters", "variants"}),
    "EstimatorBundleV2": frozenset({"schema_version", "protocol", "generator", "dataset_config", "split_manifest", "md_seed_bundle", "estimator_spec_hash", "two_pass", "model_normalization", "scalar_ranges", "clusters", "jump_limit", "sequences"}),
    "CliContractV2": frozenset({"schema_version", "protocol", "entries"}),
    "ContractHashesV2": frozenset({"raw", "b2", "morphology", "exact_stats", "bootstrap", "model", "selector", "composer", "md_protection", "noise_estimator", "sampler", "quantization", "evaluator", "cli"}),
}

_METRIC_IDS = frozenset({
    "md_recall", "md_miss_component_pixels", "md_miss_bbox_width_packed",
    "md_miss_bbox_height_packed", "md_miss_bbox_joint_min_side_packed",
    "bypass_max_abs_dn", "protect_max_abs_dn",
    "invalid_cfa_injected_max_abs_dn", "preround_min_dn", "preround_max_dn",
    "ger", "warp_flicker_dn", "warp_excess_dn", "boundary_jump_dn",
    "flat_noise_dn", "flat_flicker_dn", "texture_rho", "texture_beta",
    "texture_sigma_perp_dn", "q_contamination", "q_positive_precision",
    "lp_diff_rms_dn", "lp_diff_p99_dn", "raw_mae_dn", "raw_psnr_db",
    "pseudo_gt_mae_dn", "pseudo_gt_psnr_db", "pseudo_gt_gradient_mae_dn",
})


def validate_json_tree(value: object, context: str = "value") -> None:
    """Reject non-JSON values and non-finite numbers without imposing a
    parent schema on every nested object.

    Exact nested records are validated by their owning validator.  This helper
    is intentionally structural only; applying a top-level key set to every
    nested mapping was a subtle bug in the initial P0 implementation.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(context, "non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(context, "keys must be strings")
            validate_json_tree(item, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_tree(item, f"{context}[{index}]")
        return
    _fail(context, f"unsupported JSON value: {type(value).__name__}")


def validate_generic_named(value: object, context: str, name: str) -> Mapping[str, object]:
    """Conservative recursive validation for plan-owned types.

    Exact key checking is delegated to registered schemas where literal fields are
    frozen; unknown nested mappings are still traversed and reject non-JSON values.
    """
    mapping = expect_mapping(value, context)
    expected = _GENERIC_EXPECTED_KEYS.get(name)
    if expected is None:
        _fail(context, f"no exact schema registered for {name}")
    if name == "MetricIdV2":
        _fail(context, "MetricIdV2 is an enum, not a mapping")
    require_exact_keys(mapping, expected, name)
    validate_json_tree(mapping, context)
    return mapping


def validate_metric_id_v2(value: object, context: str = "MetricIdV2") -> str:
    metric = expect_string(value, context)
    if metric not in _METRIC_IDS:
        _fail(context, f"unknown metric id: {metric}")
    return metric

"""Read-only A10 audit for sc450ai supervision and degradation feasibility."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from raw_fusion.finetune.a8_adaptation import sha256_file
from raw_fusion.finetune.cross_sensor import (
    NativeSequenceSpec,
    SATURATION_LIMIT_DN,
    open_sequence,
)
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, pack_frame


_MAD_TO_SIGMA = 1.4826
_PAIR_SCALE = np.sqrt(2.0)
_VALID_ROLES = {"noise_statistics_source", "hard_holdout"}


@dataclass(frozen=True, slots=True)
class A10Source:
    name: str
    role: str
    source_sha256: str
    analysis_crop_sensor_xyxy: tuple[int, int, int, int]
    source: NativeSequenceSpec

    def validate(self, dataset_root: Path) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"unsupported A10 source role: {self.role}")
        if not self.source.path.resolve().is_relative_to(dataset_root):
            raise ValueError("A10 sources must remain inside the read-only dataset root")
        if self.source.reference_mode != "stress_only":
            raise ValueError("A10 sc450ai sources must remain stress-only")
        x0, y0, x1, y1 = self.analysis_crop_sensor_xyxy
        if not (0 <= x0 < x1 <= self.source.width and 0 <= y0 < y1 <= self.source.height):
            raise ValueError(f"invalid analysis crop for {self.name}")


@dataclass(frozen=True, slots=True)
class A10AuditConfig:
    dataset_root: str
    noise_profile_path: str
    noise_profile_sha256: str
    output: str
    sources: tuple[A10Source, ...]
    window_length: int = 8
    window_stride: int = 4
    pair_stride: int = 4
    spatial_sample_stride: int = 4
    flat_percentile: float = 35.0
    temporal_outlier_percentile: float = 99.5
    static_sigma_multiplier: float = 3.0
    max_correlation_train_validation_delta: float = 0.05
    min_nontrivial_correlation: float = 0.02
    semantic_constraints: dict[str, bool] | None = None

    def validate(self) -> None:
        dataset_root = Path(self.dataset_root).resolve()
        if len(self.sources) != 2:
            raise ValueError("A10 requires the complete two-stream sc450ai inventory")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("A10 source names must be unique")
        roles = [source.role for source in self.sources]
        if roles.count("noise_statistics_source") != 1 or roles.count("hard_holdout") != 1:
            raise ValueError("A10 requires one noise source and one hard holdout")
        for source in self.sources:
            source.validate(dataset_root)
        if self.window_length < 4 or self.window_stride <= 0 or self.pair_stride < 2:
            raise ValueError("invalid A10 temporal strides")
        if self.spatial_sample_stride <= 0:
            raise ValueError("spatial_sample_stride must be positive")
        if not 0.0 < self.flat_percentile <= 100.0:
            raise ValueError("flat_percentile must be in (0, 100]")
        if not 90.0 <= self.temporal_outlier_percentile < 100.0:
            raise ValueError("temporal_outlier_percentile must be in [90, 100)")
        if self.static_sigma_multiplier <= 0.0:
            raise ValueError("static_sigma_multiplier must be positive")
        if self.max_correlation_train_validation_delta <= 0.0:
            raise ValueError("correlation delta threshold must be positive")
        if self.min_nontrivial_correlation < 0.0:
            raise ValueError("minimum correlation must be non-negative")
        required_constraints = {
            "dataset_read_only": True,
            "old_teacher_used_as_clean_target": False,
            "dynamic_temporal_mean_used_as_clean_target": False,
            "hard_holdout_used_for_training": False,
            "training_launched": False,
        }
        if self.semantic_constraints != required_constraints:
            raise ValueError("A10 semantic constraints must match the read-only audit protocol")


def _source_mapping(source: NativeSequenceSpec) -> dict[str, object]:
    mapping = asdict(source)
    mapping["path"] = str(source.path)
    mapping["panel_crops"] = {
        name: list(coordinates) for name, coordinates in source.panel_crops
    }
    return mapping


def load_a10_audit_config(path: Path) -> A10AuditConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources: list[A10Source] = []
    for item in payload["sources"]:
        source_payload = dict(item["source"])
        sources.append(
            A10Source(
                name=str(item["name"]),
                role=str(item["role"]),
                source_sha256=str(item["source_sha256"]),
                analysis_crop_sensor_xyxy=tuple(
                    int(value) for value in item["analysis_crop_sensor_xyxy"]
                ),
                source=NativeSequenceSpec.from_mapping(
                    str(item["name"]), source_payload
                ),
            )
        )
    config = A10AuditConfig(
        dataset_root=str(payload["dataset_root"]),
        noise_profile_path=str(payload["noise_profile_path"]),
        noise_profile_sha256=str(payload["noise_profile_sha256"]),
        output=str(payload["output"]),
        sources=tuple(sources),
        window_length=int(payload.get("window_length", 8)),
        window_stride=int(payload.get("window_stride", 4)),
        pair_stride=int(payload.get("pair_stride", 4)),
        spatial_sample_stride=int(payload.get("spatial_sample_stride", 4)),
        flat_percentile=float(payload.get("flat_percentile", 35.0)),
        temporal_outlier_percentile=float(
            payload.get("temporal_outlier_percentile", 99.5)
        ),
        static_sigma_multiplier=float(payload.get("static_sigma_multiplier", 3.0)),
        max_correlation_train_validation_delta=float(
            payload.get("max_correlation_train_validation_delta", 0.05)
        ),
        min_nontrivial_correlation=float(
            payload.get("min_nontrivial_correlation", 0.02)
        ),
        semantic_constraints={
            str(key): bool(value)
            for key, value in payload["semantic_constraints"].items()
        },
    )
    config.validate()
    return config


def a10_config_mapping(config: A10AuditConfig) -> dict[str, object]:
    mapping = asdict(config)
    mapping["sources"] = [
        {
            "name": source.name,
            "role": source.role,
            "source_sha256": source.source_sha256,
            "analysis_crop_sensor_xyxy": list(source.analysis_crop_sensor_xyxy),
            "source": _source_mapping(source.source),
        }
        for source in config.sources
    ]
    return mapping


def _load_noise_profile(config: A10AuditConfig) -> dict[str, Any]:
    path = Path(config.noise_profile_path)
    if sha256_file(path) != config.noise_profile_sha256:
        raise ValueError("A10 noise profile SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "a9-sensor-temporal-noise-v1":
        raise ValueError("A10 requires the validated A9 temporal-noise schema")
    if not payload.get("authenticity", {}).get("passed"):
        raise ValueError("A10 requires a passed A9 noise authenticity gate")
    return payload


def _channel_vector(payload: dict[str, Any], key: str) -> np.ndarray:
    values = payload[key]
    return np.asarray([float(values[name]) for name in CHANNEL_NAMES], dtype=np.float32)


def _variance_parameters(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    models = payload["variance_model"]
    shot = np.asarray(
        [float(models[name]["shot_variance_per_signal_dn"]) for name in CHANNEL_NAMES],
        dtype=np.float32,
    )
    read = np.asarray(
        [float(models[name]["read_variance_dn2"]) for name in CHANNEL_NAMES],
        dtype=np.float32,
    )
    return shot, read


def _crop_packed(
    values: np.ndarray,
    crop_sensor_xyxy: tuple[int, int, int, int],
    spatial_stride: int,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_sensor_xyxy
    packed_x0 = x0 // 2 // spatial_stride
    packed_y0 = y0 // 2 // spatial_stride
    packed_x1 = (x1 // 2 + spatial_stride - 1) // spatial_stride
    packed_y1 = (y1 // 2 + spatial_stride - 1) // spatial_stride
    return values[..., packed_y0:packed_y1, packed_x0:packed_x1]


def _load_sampled_frames(source: A10Source, spatial_stride: int) -> np.ndarray:
    reader = open_sequence(source.source)
    output_shape = (
        source.source.frames_used,
        4,
        source.source.height // 2 // spatial_stride,
        source.source.width // 2 // spatial_stride,
    )
    frames = np.empty(output_shape, dtype=np.uint16)
    for index in range(source.source.frames_used):
        packed = pack_frame(reader.read_frame(index), source.source.adapter)
        frames[index] = packed[:, ::spatial_stride, ::spatial_stride]
    return frames


def _robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 128 or second.size != first.size:
        return 0.0
    first_centered = first.astype(np.float64) - float(np.mean(first))
    second_centered = second.astype(np.float64) - float(np.mean(second))
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _window_report(
    frames: np.ndarray,
    black: np.ndarray,
    shot: np.ndarray,
    read: np.ndarray,
    config: A10AuditConfig,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for start in range(
        0,
        frames.shape[0] - config.window_length + 1,
        config.window_stride,
    ):
        clip = frames[start : start + config.window_length].astype(np.float32)
        reference = np.median(clip, axis=0)
        signal = np.maximum(reference - black[:, None, None], 0.0)
        sigma = np.sqrt(
            shot[:, None, None] * signal + read[:, None, None]
        )
        valid = (reference > black[:, None, None] + 4.0) & (
            reference < SATURATION_LIMIT_DN
        )
        deviation = np.max(np.abs(clip - reference[None]), axis=0)
        stable = deviation <= config.static_sigma_multiplier * sigma
        highpass = np.stack(
            [channel - cv2.blur(channel, (3, 3)) for channel in reference],
            axis=0,
        )
        flat_limit = float(np.percentile(np.abs(highpass[valid]), config.flat_percentile))
        flat = valid & (np.abs(highpass) <= flat_limit)
        even_mean = np.mean(clip[::2], axis=0)
        odd_mean = np.mean(clip[1::2], axis=0)
        stable_flat = stable & flat
        split_values = (even_mean - odd_mean)[stable_flat]
        frame_means = np.mean(clip, axis=(1, 2, 3))
        result = {
            "start": start,
            "end_exclusive": start + config.window_length,
            "frame_mean_cv": float(np.std(frame_means) / np.mean(frame_means)),
            "stable_pixel_fraction": float(np.mean(stable[valid])),
            "flat_stable_pixel_fraction": float(np.mean(stable[flat])),
            "flat_pixel_fraction": float(np.mean(flat[valid])),
            "split_half_rmse_dn": float(
                np.sqrt(np.mean(split_values.astype(np.float64) ** 2))
            ),
            "stable_flat_pixels": int(stable_flat.sum()),
        }
        result["ranking_score"] = (
            result["flat_stable_pixel_fraction"]
            - 20.0 * result["frame_mean_cv"]
            - 0.001 * result["split_half_rmse_dn"]
        )
        results.append(result)
    ranked = sorted(results, key=lambda item: item["ranking_score"], reverse=True)
    return {
        "window_count": len(results),
        "best_windows": ranked[:10],
        "all_windows": results,
        "clean_reference_eligible": False,
        "clean_reference_failure_reason": (
            "All windows come from dynamic or illumination-varying stress streams; "
            "low local change cannot prove independent clean spatial truth."
        ),
    }


def _neighbor_samples(
    residual: np.ndarray, mask: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    if axis == 0:
        pair_mask = mask[:-1] & mask[1:]
        return residual[:-1][pair_mask], residual[1:][pair_mask]
    pair_mask = mask[:, :-1] & mask[:, 1:]
    return residual[:, :-1][pair_mask], residual[:, 1:][pair_mask]


def _correlation_report(
    frames: np.ndarray,
    frame_range: tuple[int, int],
    black: np.ndarray,
    config: A10AuditConfig,
) -> dict[str, Any]:
    channel_values: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    horizontal: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in CHANNEL_NAMES
    ]
    vertical: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in CHANNEL_NAMES
    ]
    row_ratios: list[list[float]] = [[] for _ in CHANNEL_NAMES]
    column_ratios: list[list[float]] = [[] for _ in CHANNEL_NAMES]
    cross_samples: list[list[np.ndarray]] = [[] for _ in CHANNEL_NAMES]
    start, end = frame_range
    pair_count = 0
    for first_index in range(start, end - 1, config.pair_stride):
        pair_count += 1
        first = frames[first_index].astype(np.float32)
        second = frames[first_index + 1].astype(np.float32)
        pair_mean = 0.5 * (first + second)
        difference = (first - second) / _PAIR_SCALE
        temporal_highpass = np.stack(
            [channel - cv2.blur(channel, (3, 3)) for channel in difference],
            axis=0,
        )
        scene_highpass = np.stack(
            [channel - cv2.blur(channel, (3, 3)) for channel in pair_mean],
            axis=0,
        )
        masks: list[np.ndarray] = []
        for channel in range(4):
            signal = pair_mean[channel] - black[channel]
            valid = (
                (signal >= 0.0)
                & (signal < 3600.0)
                & (pair_mean[channel] < SATURATION_LIMIT_DN)
            )
            flat_limit = float(
                np.percentile(
                    np.abs(scene_highpass[channel][valid]), config.flat_percentile
                )
            )
            flat = valid & (np.abs(scene_highpass[channel]) <= flat_limit)
            temporal_limit = float(
                np.percentile(
                    np.abs(temporal_highpass[channel][flat]),
                    config.temporal_outlier_percentile,
                )
            )
            mask = flat & (np.abs(temporal_highpass[channel]) <= temporal_limit)
            masks.append(mask)
            centered = difference[channel] - float(np.median(difference[channel][mask]))
            values = centered[mask]
            channel_values[channel].append(values)
            horizontal[channel].append(
                _neighbor_samples(centered, mask, axis=1)
            )
            vertical[channel].append(
                _neighbor_samples(centered, mask, axis=0)
            )
            residual_sigma = _robust_sigma(values)
            row_offsets = np.asarray(
                [
                    np.median(row[row_mask])
                    for row, row_mask in zip(
                        centered, mask, strict=True
                    )
                    if int(row_mask.sum()) >= 16
                ],
                dtype=np.float32,
            )
            column_offsets = np.asarray(
                [
                    np.median(column[column_mask])
                    for column, column_mask in zip(
                        centered.T, mask.T, strict=True
                    )
                    if int(column_mask.sum()) >= 16
                ],
                dtype=np.float32,
            )
            row_ratios[channel].append(
                _robust_sigma(row_offsets) / max(residual_sigma, 1e-6)
            )
            column_ratios[channel].append(
                _robust_sigma(column_offsets) / max(residual_sigma, 1e-6)
            )
        common = np.logical_and.reduce(masks)
        for channel in range(4):
            values = difference[channel][common]
            cross_samples[channel].append(values - float(np.median(values)))

    channels: dict[str, Any] = {}
    combined_cross = [np.concatenate(items) for items in cross_samples]
    for channel, name in enumerate(CHANNEL_NAMES):
        values = np.concatenate(channel_values[channel])
        h_first = np.concatenate([pair[0] for pair in horizontal[channel]])
        h_second = np.concatenate([pair[1] for pair in horizontal[channel]])
        v_first = np.concatenate([pair[0] for pair in vertical[channel]])
        v_second = np.concatenate([pair[1] for pair in vertical[channel]])
        channels[name] = {
            "samples": int(values.size),
            "robust_sigma_dn": _robust_sigma(values),
            "horizontal_neighbor_correlation": _correlation(h_first, h_second),
            "vertical_neighbor_correlation": _correlation(v_first, v_second),
            "median_row_common_sigma_ratio": float(np.median(row_ratios[channel])),
            "median_column_common_sigma_ratio": float(
                np.median(column_ratios[channel])
            ),
        }
    matrix = np.eye(4, dtype=np.float64)
    for first in range(4):
        for second in range(first + 1, 4):
            value = _correlation(combined_cross[first], combined_cross[second])
            matrix[first, second] = value
            matrix[second, first] = value
    return {
        "frame_range": list(frame_range),
        "pair_count": pair_count,
        "channels": channels,
        "cross_channel_correlation": matrix.tolist(),
        "common_cross_channel_samples": int(combined_cross[0].size),
    }


def _correlation_authenticity(
    train: dict[str, Any], validation: dict[str, Any], config: A10AuditConfig
) -> dict[str, Any]:
    spatial_checks: dict[str, bool] = {}
    magnitudes: list[float] = []
    for name in CHANNEL_NAMES:
        train_channel = train["channels"][name]
        validation_channel = validation["channels"][name]
        for metric in (
            "horizontal_neighbor_correlation",
            "vertical_neighbor_correlation",
            "median_row_common_sigma_ratio",
            "median_column_common_sigma_ratio",
        ):
            first = float(train_channel[metric])
            second = float(validation_channel[metric])
            spatial_checks[f"{name}_{metric}_delta"] = (
                abs(first - second) <= config.max_correlation_train_validation_delta
            )
            if "neighbor_correlation" in metric:
                magnitudes.extend((abs(first), abs(second)))
    train_matrix = np.asarray(train["cross_channel_correlation"], dtype=np.float64)
    validation_matrix = np.asarray(
        validation["cross_channel_correlation"], dtype=np.float64
    )
    off_diagonal = ~np.eye(4, dtype=bool)
    maximum_cross_delta = float(
        np.max(np.abs(train_matrix - validation_matrix)[off_diagonal])
    )
    cross_channel_passed = (
        maximum_cross_delta <= config.max_correlation_train_validation_delta
    )
    nontrivial = max(magnitudes) >= config.min_nontrivial_correlation
    per_channel_spatial_passed = all(spatial_checks.values()) and nontrivial
    return {
        "passed": per_channel_spatial_passed and cross_channel_passed,
        "per_channel_spatial_structure_passed": per_channel_spatial_passed,
        "cross_channel_coupling_passed": cross_channel_passed,
        "spatial_checks": spatial_checks,
        "maximum_cross_channel_delta": maximum_cross_delta,
        "nontrivial_spatial_correlation_observed": nontrivial,
        "interpretation": (
            "Per-channel spatial structure and cross-channel coupling are gated "
            "separately. Neither result creates a clean spatial target or identifies "
            "sensor blur."
        ),
    }


def _recommended_degradation_model(
    noise_profile_path: Path,
    noise_profile_sha256: str,
    train: dict[str, Any],
    validation: dict[str, Any],
    authenticity: dict[str, Any],
) -> dict[str, Any]:
    channel_parameters: dict[str, Any] = {}
    for name in CHANNEL_NAMES:
        first = train["channels"][name]
        second = validation["channels"][name]
        channel_parameters[name] = {
            metric: 0.5 * (float(first[metric]) + float(second[metric]))
            for metric in (
                "horizontal_neighbor_correlation",
                "vertical_neighbor_correlation",
                "median_row_common_sigma_ratio",
                "median_column_common_sigma_ratio",
            )
        }
    return {
        "status": "partial_noise_model_only",
        "signal_dependent_variance": {
            "source": str(noise_profile_path.resolve()),
            "sha256": noise_profile_sha256,
            "allowed": True,
        },
        "per_channel_spatial_structure": {
            "allowed": authenticity["per_channel_spatial_structure_passed"],
            "parameters": channel_parameters,
            "required_validation": (
                "A synthetic residual generator must reproduce these metrics on "
                "disjoint signal bins before it may enter a training mixture."
            ),
        },
        "cross_channel_coupling": {
            "allowed": authenticity["cross_channel_coupling_passed"],
            "policy": (
                "measured"
                if authenticity["cross_channel_coupling_passed"]
                else "identity"
            ),
        },
        "unsupported_components": [
            "PRNU",
            "DSNU",
            "sensor blur or MTF",
            "CFA spectral response transfer",
            "motion-detail clean target",
        ],
        "sufficient_for_model_training": False,
    }


def _quantization_report(frames: np.ndarray) -> dict[str, Any]:
    flattened = frames[:, :, ::8, ::8].reshape(-1)
    return {
        "minimum_dn": int(flattened.min()),
        "maximum_dn": int(flattened.max()),
        "fraction_at_or_below_4095": float(np.mean(flattened <= 4095)),
        "fraction_divisible_by_4": float(np.mean(flattened % 4 == 0)),
        "fraction_divisible_by_64": float(np.mean(flattened % 64 == 0)),
        "interpretation": "Low-aligned effective 10-bit samples stored in uint16.",
    }


def audit_sc450ai_supervision(config: A10AuditConfig) -> dict[str, Any]:
    config.validate()
    noise_profile = _load_noise_profile(config)
    black = _channel_vector(noise_profile, "black_levels")
    shot, read = _variance_parameters(noise_profile)
    source_reports: dict[str, Any] = {}
    correlation_train: dict[str, Any] | None = None
    correlation_validation: dict[str, Any] | None = None
    for source in config.sources:
        actual_sha256 = sha256_file(source.source.path)
        if actual_sha256 != source.source_sha256:
            raise ValueError(f"source SHA-256 mismatch: {source.name}")
        reader = open_sequence(source.source)
        if reader.frames != source.source.frames_used:
            raise ValueError(f"unexpected frame count: {source.name}")
        sampled = _load_sampled_frames(source, config.spatial_sample_stride)
        cropped = _crop_packed(
            sampled,
            source.analysis_crop_sensor_xyxy,
            config.spatial_sample_stride,
        )
        report: dict[str, Any] = {
            "role": source.role,
            "path": str(source.source.path),
            "sha256": actual_sha256,
            "frames": reader.frames,
            "native_shape": [source.source.height, source.source.width],
            "sampled_packed_shape": list(sampled.shape),
            "analysis_crop_sensor_xyxy": list(source.analysis_crop_sensor_xyxy),
            "quantization": _quantization_report(sampled),
            "windows": _window_report(cropped, black, shot, read, config),
        }
        if source.role == "noise_statistics_source":
            fit = noise_profile["fit"]
            train_range = tuple(int(value) for value in fit["train_frame_range"])
            validation_range = tuple(
                int(value) for value in fit["validation_frame_range"]
            )
            correlation_train = _correlation_report(
                cropped, train_range, black, config
            )
            correlation_validation = _correlation_report(
                cropped, validation_range, black, config
            )
            report["spatial_noise_correlation"] = {
                "train": correlation_train,
                "validation": correlation_validation,
            }
        source_reports[source.name] = report
        del sampled, cropped

    if correlation_train is None or correlation_validation is None:
        raise ValueError("noise statistics source was not audited")
    correlation_gate = _correlation_authenticity(
        correlation_train, correlation_validation, config
    )
    degradation_model = _recommended_degradation_model(
        Path(config.noise_profile_path),
        config.noise_profile_sha256,
        correlation_train,
        correlation_validation,
        correlation_gate,
    )
    result = {
        "schema": "a10-sc450ai-supervision-audit-v1",
        "status": "completed",
        "inventory": {
            "stream_count": len(config.sources),
            "dark_frame_streams": 0,
            "flat_field_streams": 0,
            "cross_gain_pairs": 0,
            "independent_clean_raw_targets": 0,
            "sources": source_reports,
        },
        "noise_profile": {
            "path": str(Path(config.noise_profile_path).resolve()),
            "sha256": config.noise_profile_sha256,
            "authenticity_passed": True,
        },
        "spatial_correlation_authenticity": correlation_gate,
        "recommended_degradation_model": degradation_model,
        "training_admission": {
            "supervised_sc450ai_finetune_allowed": False,
            "dynamic_self_supervision_reuse_allowed": False,
            "same_a9_protocol_extension_allowed": False,
            "per_channel_spatial_degradation_validation_allowed": correlation_gate[
                "per_channel_spatial_structure_passed"
            ],
            "cross_channel_coupling_allowed": correlation_gate[
                "cross_channel_coupling_passed"
            ],
            "next_model_training_allowed": False,
            "hard_holdout": "sc450ai_1435x_motion_black_object",
            "reason": (
                "No independent clean, dark-field, flat-field, or cross-gain target exists. "
                "Only validated noise/degradation statistics may be reused."
            ),
        },
        "identifiability_limits": {
            "spatial_noise_correlation": "measurable from adjacent-frame differences",
            "signal_dependent_variance": "already validated by A9",
            "persistent_prnu_dsnu": "not separable from scene texture without flat/dark fields",
            "sensor_blur_and_cfa_response": "not identifiable without paired clean spatial content",
            "motion_detail_truth": "not available in either stream",
        },
        "recommended_next_protocol": {
            "stage": "A10b degradation-only validation before any training",
            "actions": [
                "Build a synthetic sc450ai degradation using only validated temporal variance and any passed spatial-correlation terms.",
                "Keep cross-channel noise independent unless the disjoint correlation gate passes.",
                "Do not synthesize unsupported PRNU, DSNU, blur, or CFA transfer parameters from these stress streams.",
                "Validate the degradation on disjoint 804x frames using noise residual statistics, not denoised image quality.",
                "Keep the complete 1435x black-object sequence as a no-reference fixed-ISP hard holdout.",
                "Require an additional spatial-detail source before authorizing another unified-model training run.",
            ],
        },
        "config": a10_config_mapping(config),
    }
    return result


def save_a10_audit(config: A10AuditConfig, result: dict[str, Any]) -> Path:
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output

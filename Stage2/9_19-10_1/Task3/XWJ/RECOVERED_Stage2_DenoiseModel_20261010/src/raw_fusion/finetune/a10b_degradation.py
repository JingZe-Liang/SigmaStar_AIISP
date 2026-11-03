"""A10b degradation-only validation for the sc450ai sensor."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from raw_fusion.finetune.a10_sc450_audit import (
    A10AuditConfig,
    A10Source,
    load_a10_audit_config,
)
from raw_fusion.finetune.a8_adaptation import sha256_file
from raw_fusion.finetune.cross_sensor import SATURATION_LIMIT_DN, open_sequence
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, pack_frame


WHITE_LEVEL_DN = 4095.0
_PAIR_SCALE = np.sqrt(2.0)
_MAD_TO_SIGMA = 1.4826
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_MODEL_NAMES = ("legacy_a9_iid_q1", "quantized_iid_q4", "a10b_spectral_q4")


@dataclass(frozen=True, slots=True)
class A10bConfig:
    dataset_root: str
    a10_config_path: str
    a10_config_sha256: str
    a10_audit_path: str
    a10_audit_sha256: str
    noise_profile_path: str
    noise_profile_sha256: str
    output_dir: str
    noise_source_name: str
    hard_holdout_name: str
    train_frame_range: tuple[int, int]
    validation_frame_range: tuple[int, int]
    hard_holdout_frame_range: tuple[int, int]
    pair_stride: int
    metric_sample_stride: int
    correlation_lags: tuple[int, ...]
    gate_correlation_lags: tuple[int, ...]
    flat_percentile: float
    temporal_outlier_percentile: float
    signal_bin_edges_dn: tuple[float, ...]
    min_bin_pixels: int
    validation_replicas: int
    hard_holdout_replicas: int
    seed: int
    spectral_floor: float
    quantization_step_dn: float
    max_median_relative_sigma_error: float
    max_p90_relative_sigma_error: float
    max_aggregate_median_relative_sigma_error: float
    max_aggregate_p90_relative_sigma_error: float
    max_weighted_correlation_mae: float
    max_lag1_correlation_absolute_error: float
    max_row_column_ratio_error: float
    max_normalized_quantile_error: float
    max_cross_channel_correlation: float
    min_spatial_rmse_improvement_over_q4_iid: float
    min_validation_pair_wins: int
    semantic_constraints: dict[str, bool]

    def validate(self) -> None:
        if self.pair_stride < 2 or self.metric_sample_stride <= 0:
            raise ValueError("A10b pair and metric strides must be positive")
        ranges = (
            self.train_frame_range,
            self.validation_frame_range,
            self.hard_holdout_frame_range,
        )
        if any(start < 0 or end - start < 2 for start, end in ranges):
            raise ValueError("A10b frame ranges must contain adjacent pairs")
        if max(self.train_frame_range[0], self.validation_frame_range[0]) < min(
            self.train_frame_range[1], self.validation_frame_range[1]
        ):
            raise ValueError("A10b fit and validation ranges must be disjoint")
        if not self.correlation_lags or self.correlation_lags[0] != 1:
            raise ValueError("A10b correlation lags must start at one")
        if tuple(range(1, max(self.correlation_lags) + 1)) != self.correlation_lags:
            raise ValueError("A10b correlation lags must be contiguous")
        if any(lag not in self.correlation_lags for lag in self.gate_correlation_lags):
            raise ValueError("A10b gate lags must be measured correlation lags")
        if self.gate_correlation_lags != (1, 2, 4, 8):
            raise ValueError("A10b final gate requires packed lags 1, 2, 4, and 8")
        if not 0.0 < self.flat_percentile <= 100.0:
            raise ValueError("flat_percentile must be in (0, 100]")
        if not 90.0 <= self.temporal_outlier_percentile < 100.0:
            raise ValueError("temporal_outlier_percentile must be in [90, 100)")
        if any(
            high <= low
            for low, high in zip(
                self.signal_bin_edges_dn[:-1],
                self.signal_bin_edges_dn[1:],
                strict=True,
            )
        ):
            raise ValueError("signal-bin edges must be strictly increasing")
        if self.min_bin_pixels <= 0:
            raise ValueError("min_bin_pixels must be positive")
        if self.validation_replicas <= 0 or self.hard_holdout_replicas <= 0:
            raise ValueError("A10b replica counts must be positive")
        if not 0.0 < self.spectral_floor <= 1.0:
            raise ValueError("spectral_floor must be in (0, 1]")
        if self.quantization_step_dn <= 0.0:
            raise ValueError("quantization_step_dn must be positive")
        thresholds = (
            self.max_median_relative_sigma_error,
            self.max_p90_relative_sigma_error,
            self.max_aggregate_median_relative_sigma_error,
            self.max_aggregate_p90_relative_sigma_error,
            self.max_weighted_correlation_mae,
            self.max_lag1_correlation_absolute_error,
            self.max_row_column_ratio_error,
            self.max_normalized_quantile_error,
            self.max_cross_channel_correlation,
        )
        if any(value <= 0.0 for value in thresholds):
            raise ValueError("A10b gate thresholds must be positive")
        if not 0.0 <= self.min_spatial_rmse_improvement_over_q4_iid < 1.0:
            raise ValueError("A10b minimum spatial improvement must be in [0, 1)")
        validation_pairs = len(
            range(
                self.validation_frame_range[0],
                self.validation_frame_range[1] - 1,
                self.pair_stride,
            )
        )
        if not 1 <= self.min_validation_pair_wins <= validation_pairs:
            raise ValueError("A10b validation-pair win threshold is invalid")
        required_constraints = {
            "dataset_read_only": True,
            "old_teacher_used_as_clean_target": False,
            "dynamic_temporal_mean_used_as_clean_target": False,
            "adjacent_pair_mean_used_as_clean_target": False,
            "hard_holdout_used_for_fitting": False,
            "model_training_launched": False,
        }
        if self.semantic_constraints != required_constraints:
            raise ValueError("A10b semantic constraints do not match the protocol")


@dataclass(slots=True)
class _PairMoments:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(self, first: np.ndarray, second: np.ndarray) -> None:
        if first.size == 0 or first.size != second.size:
            return
        x = first.astype(np.float64, copy=False)
        y = second.astype(np.float64, copy=False)
        self.count += int(x.size)
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_xx += float(np.sum(x * x))
        self.sum_yy += float(np.sum(y * y))
        self.sum_xy += float(np.sum(x * y))

    def correlation(self) -> float:
        if self.count < 128:
            return 0.0
        count = float(self.count)
        covariance = self.sum_xy - self.sum_x * self.sum_y / count
        variance_x = self.sum_xx - self.sum_x * self.sum_x / count
        variance_y = self.sum_yy - self.sum_y * self.sum_y / count
        denominator = np.sqrt(max(variance_x * variance_y, 0.0))
        if denominator <= 0.0:
            return 0.0
        return float(covariance / denominator)


class _MetricAccumulator:
    def __init__(self, config: A10bConfig) -> None:
        self.config = config
        lag_count = len(config.correlation_lags)
        self.horizontal = [
            [_PairMoments() for _ in range(lag_count)] for _ in CHANNEL_NAMES
        ]
        self.vertical = [
            [_PairMoments() for _ in range(lag_count)] for _ in CHANNEL_NAMES
        ]
        self.cross = [
            [_PairMoments() for _ in CHANNEL_NAMES] for _ in CHANNEL_NAMES
        ]
        self.normalized_values: list[list[np.ndarray]] = [
            [] for _ in CHANNEL_NAMES
        ]
        self.row_ratios: list[list[float]] = [[] for _ in CHANNEL_NAMES]
        self.column_ratios: list[list[float]] = [[] for _ in CHANNEL_NAMES]
        self.bin_values: list[list[list[np.ndarray]]] = [
            [[] for _ in range(len(config.signal_bin_edges_dn) - 1)]
            for _ in CHANNEL_NAMES
        ]
        self.pair_count = 0
        self.total_values = 0
        self.legal_values = 0
        self.clipped_values = 0
        self.q4_values = 0

    def update(
        self,
        first: np.ndarray,
        second: np.ndarray,
        black: np.ndarray,
        shot: np.ndarray,
        read: np.ndarray,
    ) -> None:
        pair_mean = 0.5 * (first.astype(np.float32) + second.astype(np.float32))
        difference = (
            first.astype(np.float32) - second.astype(np.float32)
        ) / _PAIR_SCALE
        masks: list[np.ndarray] = []
        normalized_channels: list[np.ndarray] = []
        stride = self.config.metric_sample_stride
        for channel in range(4):
            signal = pair_mean[channel] - black[channel]
            sigma = np.sqrt(np.maximum(shot[channel] * signal + read[channel], 1e-6))
            scene_highpass = pair_mean[channel] - cv2.blur(pair_mean[channel], (3, 3))
            valid = (
                (signal >= self.config.signal_bin_edges_dn[0])
                & (signal < self.config.signal_bin_edges_dn[-1])
                & (pair_mean[channel] < SATURATION_LIMIT_DN)
            )
            flat_limit = float(
                np.percentile(
                    np.abs(scene_highpass[valid]), self.config.flat_percentile
                )
            )
            flat = valid & (np.abs(scene_highpass) <= flat_limit)
            centered = difference[channel] - float(np.median(difference[channel][flat]))
            normalized = centered / sigma
            temporal_highpass = normalized - cv2.blur(normalized, (3, 3))
            temporal_limit = float(
                np.percentile(
                    np.abs(temporal_highpass[flat]),
                    self.config.temporal_outlier_percentile,
                )
            )
            mask = flat & (np.abs(temporal_highpass) <= temporal_limit)
            masks.append(mask)
            normalized_channels.append(normalized)
            sampled_values = normalized[::stride, ::stride][mask[::stride, ::stride]]
            self.normalized_values[channel].append(sampled_values.astype(np.float32))
            residual_sigma = _robust_sigma(sampled_values)
            row_offsets = np.asarray(
                [
                    np.median(row[row_mask])
                    for row, row_mask in zip(normalized, mask, strict=True)
                    if int(row_mask.sum()) >= 32
                ],
                dtype=np.float32,
            )
            column_offsets = np.asarray(
                [
                    np.median(column[column_mask])
                    for column, column_mask in zip(
                        normalized.T, mask.T, strict=True
                    )
                    if int(column_mask.sum()) >= 32
                ],
                dtype=np.float32,
            )
            self.row_ratios[channel].append(
                _robust_sigma(row_offsets) / max(residual_sigma, 1e-6)
            )
            self.column_ratios[channel].append(
                _robust_sigma(column_offsets) / max(residual_sigma, 1e-6)
            )
            for lag_index, lag in enumerate(self.config.correlation_lags):
                horizontal_mask = mask[:, :-lag] & mask[:, lag:]
                vertical_mask = mask[:-lag] & mask[lag:]
                horizontal_mask = horizontal_mask[::stride, ::stride]
                vertical_mask = vertical_mask[::stride, ::stride]
                self.horizontal[channel][lag_index].update(
                    normalized[:, :-lag][::stride, ::stride][horizontal_mask],
                    normalized[:, lag:][::stride, ::stride][horizontal_mask],
                )
                self.vertical[channel][lag_index].update(
                    normalized[:-lag][::stride, ::stride][vertical_mask],
                    normalized[lag:][::stride, ::stride][vertical_mask],
                )
            sampled_signal = signal[::stride, ::stride]
            sampled_residual = centered[::stride, ::stride]
            sampled_mask = mask[::stride, ::stride]
            for bin_index, (low, high) in enumerate(
                zip(
                    self.config.signal_bin_edges_dn[:-1],
                    self.config.signal_bin_edges_dn[1:],
                    strict=True,
                )
            ):
                selected = sampled_mask & (sampled_signal >= low) & (sampled_signal < high)
                values = sampled_residual[selected]
                if values.size:
                    self.bin_values[channel][bin_index].append(
                        values.astype(np.float32)
                    )
        common = np.logical_and.reduce(masks)[::stride, ::stride]
        for first_channel in range(4):
            first_values = normalized_channels[first_channel][::stride, ::stride][common]
            for second_channel in range(first_channel + 1, 4):
                second_values = normalized_channels[second_channel][::stride, ::stride][common]
                self.cross[first_channel][second_channel].update(
                    first_values, second_values
                )
        stacked = np.stack((first, second), axis=0)
        self.total_values += int(stacked.size)
        self.legal_values += int(np.count_nonzero((stacked >= 0.0) & (stacked <= WHITE_LEVEL_DN)))
        self.clipped_values += int(np.count_nonzero((stacked <= 0.0) | (stacked >= 4092.0)))
        self.q4_values += int(np.count_nonzero(np.mod(stacked, 4.0) == 0.0))
        self.pair_count += 1

    def report(self) -> dict[str, Any]:
        channels: dict[str, Any] = {}
        sigma_bins: dict[str, list[dict[str, Any]]] = {}
        for channel, name in enumerate(CHANNEL_NAMES):
            values = np.concatenate(self.normalized_values[channel])
            channels[name] = {
                "normalized_robust_sigma": _robust_sigma(values),
                "normalized_quantiles": {
                    str(quantile): float(np.quantile(values, quantile))
                    for quantile in _QUANTILES
                },
                "horizontal_correlation": {
                    str(lag): self.horizontal[channel][index].correlation()
                    for index, lag in enumerate(self.config.correlation_lags)
                },
                "vertical_correlation": {
                    str(lag): self.vertical[channel][index].correlation()
                    for index, lag in enumerate(self.config.correlation_lags)
                },
                "median_row_common_sigma_ratio": float(
                    np.median(self.row_ratios[channel])
                ),
                "median_column_common_sigma_ratio": float(
                    np.median(self.column_ratios[channel])
                ),
                "samples": int(values.size),
            }
            channel_bins: list[dict[str, Any]] = []
            for bin_index, (low, high) in enumerate(
                zip(
                    self.config.signal_bin_edges_dn[:-1],
                    self.config.signal_bin_edges_dn[1:],
                    strict=True,
                )
            ):
                pieces = self.bin_values[channel][bin_index]
                bin_values = np.concatenate(pieces) if pieces else np.empty(0)
                if bin_values.size < self.config.min_bin_pixels:
                    continue
                channel_bins.append(
                    {
                        "low_dn": low,
                        "high_dn": high,
                        "center_dn": 0.5 * (low + high),
                        "sigma_dn": _robust_sigma(bin_values),
                        "pixels": int(bin_values.size),
                    }
                )
            sigma_bins[name] = channel_bins
        cross_matrix = np.eye(4, dtype=np.float64)
        for first_channel in range(4):
            for second_channel in range(first_channel + 1, 4):
                correlation = self.cross[first_channel][second_channel].correlation()
                cross_matrix[first_channel, second_channel] = correlation
                cross_matrix[second_channel, first_channel] = correlation
        return {
            "pair_count": self.pair_count,
            "channels": channels,
            "signal_bins": sigma_bins,
            "cross_channel_correlation": cross_matrix.tolist(),
            "legal_raw_fraction": self.legal_values / max(self.total_values, 1),
            "endpoint_fraction": self.clipped_values / max(self.total_values, 1),
            "fraction_divisible_by_4": self.q4_values / max(self.total_values, 1),
        }


def _robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))


def load_a10b_config(path: Path) -> A10bConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = A10bConfig(
        dataset_root=str(payload["dataset_root"]),
        a10_config_path=str(payload["a10_config_path"]),
        a10_config_sha256=str(payload["a10_config_sha256"]),
        a10_audit_path=str(payload["a10_audit_path"]),
        a10_audit_sha256=str(payload["a10_audit_sha256"]),
        noise_profile_path=str(payload["noise_profile_path"]),
        noise_profile_sha256=str(payload["noise_profile_sha256"]),
        output_dir=str(payload["output_dir"]),
        noise_source_name=str(payload["noise_source_name"]),
        hard_holdout_name=str(payload["hard_holdout_name"]),
        train_frame_range=tuple(int(value) for value in payload["train_frame_range"]),
        validation_frame_range=tuple(
            int(value) for value in payload["validation_frame_range"]
        ),
        hard_holdout_frame_range=tuple(
            int(value) for value in payload["hard_holdout_frame_range"]
        ),
        pair_stride=int(payload["pair_stride"]),
        metric_sample_stride=int(payload["metric_sample_stride"]),
        correlation_lags=tuple(int(value) for value in payload["correlation_lags"]),
        gate_correlation_lags=tuple(
            int(value) for value in payload["gate_correlation_lags"]
        ),
        flat_percentile=float(payload["flat_percentile"]),
        temporal_outlier_percentile=float(payload["temporal_outlier_percentile"]),
        signal_bin_edges_dn=tuple(
            float(value) for value in payload["signal_bin_edges_dn"]
        ),
        min_bin_pixels=int(payload["min_bin_pixels"]),
        validation_replicas=int(payload["validation_replicas"]),
        hard_holdout_replicas=int(payload["hard_holdout_replicas"]),
        seed=int(payload["seed"]),
        spectral_floor=float(payload["spectral_floor"]),
        quantization_step_dn=float(payload["quantization_step_dn"]),
        max_median_relative_sigma_error=float(
            payload["max_median_relative_sigma_error"]
        ),
        max_p90_relative_sigma_error=float(
            payload["max_p90_relative_sigma_error"]
        ),
        max_aggregate_median_relative_sigma_error=float(
            payload["max_aggregate_median_relative_sigma_error"]
        ),
        max_aggregate_p90_relative_sigma_error=float(
            payload["max_aggregate_p90_relative_sigma_error"]
        ),
        max_weighted_correlation_mae=float(
            payload["max_weighted_correlation_mae"]
        ),
        max_lag1_correlation_absolute_error=float(
            payload["max_lag1_correlation_absolute_error"]
        ),
        max_row_column_ratio_error=float(payload["max_row_column_ratio_error"]),
        max_normalized_quantile_error=float(
            payload["max_normalized_quantile_error"]
        ),
        max_cross_channel_correlation=float(
            payload["max_cross_channel_correlation"]
        ),
        min_spatial_rmse_improvement_over_q4_iid=float(
            payload["min_spatial_rmse_improvement_over_q4_iid"]
        ),
        min_validation_pair_wins=int(payload["min_validation_pair_wins"]),
        semantic_constraints={
            str(key): bool(value)
            for key, value in payload["semantic_constraints"].items()
        },
    )
    config.validate()
    return config


def _load_json_with_hash(path: str, expected_hash: str) -> dict[str, Any]:
    file_path = Path(path)
    if sha256_file(file_path) != expected_hash:
        raise ValueError(f"A10b input SHA-256 mismatch: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def _source_by_name(config: A10AuditConfig, name: str) -> A10Source:
    matches = [source for source in config.sources if source.name == name]
    if len(matches) != 1:
        raise ValueError(f"A10b source not uniquely defined: {name}")
    return matches[0]


def _crop_packed(
    packed: np.ndarray, crop_sensor_xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = crop_sensor_xyxy
    if any(value % 2 for value in crop_sensor_xyxy):
        raise ValueError("A10b analysis crops must align to packed Bayer coordinates")
    return packed[:, y0 // 2 : y1 // 2, x0 // 2 : x1 // 2]


def _iter_real_pairs(
    source: A10Source,
    frame_range: tuple[int, int],
    pair_stride: int,
):
    reader = open_sequence(source.source)
    start, end = frame_range
    if end > source.source.frames_used:
        raise ValueError(f"A10b frame range exceeds source: {source.name}")
    for first_index in range(start, end - 1, pair_stride):
        first = _crop_packed(
            pack_frame(reader.read_frame(first_index), source.source.adapter),
            source.analysis_crop_sensor_xyxy,
        ).astype(np.float32)
        second = _crop_packed(
            pack_frame(reader.read_frame(first_index + 1), source.source.adapter),
            source.analysis_crop_sensor_xyxy,
        ).astype(np.float32)
        yield first_index, first, second


def _variance_parameters(
    profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    black = np.asarray(
        [float(profile["black_levels"][name]) for name in CHANNEL_NAMES],
        dtype=np.float32,
    )
    shot = np.asarray(
        [
            float(profile["variance_model"][name]["shot_variance_per_signal_dn"])
            for name in CHANNEL_NAMES
        ],
        dtype=np.float32,
    )
    read = np.asarray(
        [
            float(profile["variance_model"][name]["read_variance_dn2"])
            for name in CHANNEL_NAMES
        ],
        dtype=np.float32,
    )
    return black, shot, read


def _fit_spatial_model(real_train: dict[str, Any], config: A10bConfig) -> dict[str, Any]:
    horizontal = [
        float(
            np.mean(
                [
                    real_train["channels"][name]["horizontal_correlation"][str(lag)]
                    for name in CHANNEL_NAMES
                ]
            )
        )
        for lag in config.correlation_lags
    ]
    vertical = [
        float(
            np.mean(
                [
                    real_train["channels"][name]["vertical_correlation"][str(lag)]
                    for name in CHANNEL_NAMES
                ]
            )
        )
        for lag in config.correlation_lags
    ]
    shared = {
        "horizontal_autocorrelation": horizontal,
        "vertical_autocorrelation": vertical,
    }
    channels = {name: dict(shared) for name in CHANNEL_NAMES}
    return {
        "schema": "a10b-separable-spatial-spectrum-v1",
        "fit_split": list(config.train_frame_range),
        "fit_scope": "shared horizontal and vertical spectrum across Bayer channels",
        "correlation_lags_packed_pixels": list(config.correlation_lags),
        "spectral_floor": config.spectral_floor,
        "cross_channel_policy": "identity",
        "channels": channels,
    }


def _axis_spectrum(
    length: int, autocorrelation: list[float], spectral_floor: float, *, real: bool
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(length) if real else np.fft.fftfreq(length)
    spectrum = np.ones_like(frequencies, dtype=np.float64)
    for lag, correlation in enumerate(autocorrelation, start=1):
        spectrum += 2.0 * correlation * np.cos(2.0 * np.pi * frequencies * lag)
    spectrum = np.maximum(spectrum, spectral_floor)
    return spectrum / float(np.mean(spectrum))


def generate_unit_spatial_noise(
    shape: tuple[int, int],
    horizontal_autocorrelation: list[float],
    vertical_autocorrelation: list[float],
    spectral_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one unit-variance field from a separable stationary spectrum."""

    height, width = shape
    horizontal = _axis_spectrum(
        width, horizontal_autocorrelation, spectral_floor, real=True
    )
    vertical = _axis_spectrum(
        height, vertical_autocorrelation, spectral_floor, real=False
    )
    power = vertical[:, None] * horizontal[None, :]
    white = rng.standard_normal(shape)
    field = np.fft.irfft2(
        np.fft.rfft2(white) * np.sqrt(power),
        s=shape,
    ).real
    field -= float(np.mean(field))
    field /= max(float(np.std(field)), 1e-8)
    return field.astype(np.float32)


def generate_degraded_frame(
    conditioning_dn: np.ndarray,
    black: np.ndarray,
    shot: np.ndarray,
    read: np.ndarray,
    rng: np.random.Generator,
    *,
    spatial_model: dict[str, Any] | None,
    quantization_step_dn: float,
    spectral_floor: float,
) -> np.ndarray:
    """Generate one packed RAW frame without treating conditioning data as a target."""

    if conditioning_dn.ndim != 3 or conditioning_dn.shape[0] != 4:
        raise ValueError("conditioning_dn must have shape (4,H,W)")
    signal = np.maximum(conditioning_dn.astype(np.float32) - black[:, None, None], 0.0)
    variance = shot[:, None, None] * signal + read[:, None, None]
    noise = np.empty_like(signal)
    for channel, name in enumerate(CHANNEL_NAMES):
        if spatial_model is None:
            unit = rng.standard_normal(signal[channel].shape).astype(np.float32)
        else:
            parameters = spatial_model["channels"][name]
            unit = generate_unit_spatial_noise(
                signal[channel].shape,
                parameters["horizontal_autocorrelation"],
                parameters["vertical_autocorrelation"],
                spectral_floor,
                rng,
            )
        noise[channel] = np.sqrt(np.maximum(variance[channel], 0.0)) * unit
    degraded = black[:, None, None] + signal + noise
    degraded = np.rint(degraded / quantization_step_dn) * quantization_step_dn
    maximum = np.floor(WHITE_LEVEL_DN / quantization_step_dn) * quantization_step_dn
    return np.clip(degraded, 0.0, maximum).astype(np.float32)


def _collect_real_metrics(
    source: A10Source,
    frame_range: tuple[int, int],
    config: A10bConfig,
    black: np.ndarray,
    shot: np.ndarray,
    read: np.ndarray,
) -> dict[str, Any]:
    accumulator = _MetricAccumulator(config)
    for _, first, second in _iter_real_pairs(source, frame_range, config.pair_stride):
        accumulator.update(first, second, black, shot, read)
    return accumulator.report()


def _model_settings(
    model_name: str,
    config: A10bConfig,
    spatial_model: dict[str, Any],
) -> tuple[dict[str, Any] | None, float]:
    if model_name == "legacy_a9_iid_q1":
        return None, 1.0
    if model_name == "quantized_iid_q4":
        return None, config.quantization_step_dn
    if model_name == "a10b_spectral_q4":
        return spatial_model, config.quantization_step_dn
    raise ValueError(f"unknown A10b model: {model_name}")


def _collect_synthetic_metrics(
    source: A10Source,
    frame_range: tuple[int, int],
    replicas: int,
    config: A10bConfig,
    black: np.ndarray,
    shot: np.ndarray,
    read: np.ndarray,
    spatial_model: dict[str, Any],
    model_name: str,
    seed_offset: int,
) -> dict[str, Any]:
    accumulator = _MetricAccumulator(config)
    selected_model, quantization = _model_settings(model_name, config, spatial_model)
    for first_index, first, second in _iter_real_pairs(
        source, frame_range, config.pair_stride
    ):
        conditioning = 0.5 * (first + second)
        for replica in range(replicas):
            seed = config.seed + seed_offset + first_index * 1009 + replica * 104729
            rng = np.random.default_rng(seed)
            synthetic_first = generate_degraded_frame(
                conditioning,
                black,
                shot,
                read,
                rng,
                spatial_model=selected_model,
                quantization_step_dn=quantization,
                spectral_floor=config.spectral_floor,
            )
            synthetic_second = generate_degraded_frame(
                conditioning,
                black,
                shot,
                read,
                rng,
                spatial_model=selected_model,
                quantization_step_dn=quantization,
                spectral_floor=config.spectral_floor,
            )
            accumulator.update(
                synthetic_first, synthetic_second, black, shot, read
            )
    return accumulator.report()


def _compare_metrics(
    real: dict[str, Any], synthetic: dict[str, Any], config: A10bConfig
) -> dict[str, Any]:
    sigma_errors: list[float] = []
    correlation_errors: list[tuple[int, float]] = []
    row_column_errors: list[float] = []
    quantile_errors: list[float] = []
    channel_reports: dict[str, Any] = {}
    for name in CHANNEL_NAMES:
        real_bins = {
            (item["low_dn"], item["high_dn"]): item
            for item in real["signal_bins"][name]
        }
        synthetic_bins = {
            (item["low_dn"], item["high_dn"]): item
            for item in synthetic["signal_bins"][name]
        }
        channel_sigma_errors = []
        for key in sorted(real_bins.keys() & synthetic_bins.keys()):
            reference = float(real_bins[key]["sigma_dn"])
            error = abs(float(synthetic_bins[key]["sigma_dn"]) - reference) / max(
                reference, 1e-6
            )
            channel_sigma_errors.append(error)
            sigma_errors.append(error)
        real_channel = real["channels"][name]
        synthetic_channel = synthetic["channels"][name]
        channel_correlation_errors = []
        for direction in ("horizontal_correlation", "vertical_correlation"):
            for lag in config.gate_correlation_lags:
                error = abs(
                    float(synthetic_channel[direction][str(lag)])
                    - float(real_channel[direction][str(lag)])
                )
                channel_correlation_errors.append(error)
                correlation_errors.append((lag, error))
        channel_row_column_errors = []
        for metric in (
            "median_row_common_sigma_ratio",
            "median_column_common_sigma_ratio",
        ):
            error = abs(
                float(synthetic_channel[metric]) - float(real_channel[metric])
            )
            channel_row_column_errors.append(error)
            row_column_errors.append(error)
        channel_quantile_errors = []
        for quantile in _QUANTILES:
            key = str(quantile)
            error = abs(
                float(synthetic_channel["normalized_quantiles"][key])
                - float(real_channel["normalized_quantiles"][key])
            )
            channel_quantile_errors.append(error)
            quantile_errors.append(error)
        channel_reports[name] = {
            "matched_signal_bins": len(channel_sigma_errors),
            "median_relative_sigma_error": float(np.median(channel_sigma_errors))
            if channel_sigma_errors
            else float("inf"),
            "p90_relative_sigma_error": float(
                np.percentile(channel_sigma_errors, 90.0)
            )
            if channel_sigma_errors
            else float("inf"),
            "maximum_correlation_error": max(channel_correlation_errors),
            "maximum_row_column_ratio_error": max(channel_row_column_errors),
            "maximum_normalized_quantile_error": max(channel_quantile_errors),
        }
    synthetic_cross = np.asarray(
        synthetic["cross_channel_correlation"], dtype=np.float64
    )
    off_diagonal = ~np.eye(4, dtype=bool)
    maximum_cross = float(np.max(np.abs(synthetic_cross[off_diagonal])))
    median_sigma = float(np.median(sigma_errors)) if sigma_errors else float("inf")
    p90_sigma = (
        float(np.percentile(sigma_errors, 90.0)) if sigma_errors else float("inf")
    )
    error_values = [error for _, error in correlation_errors]
    lag1_errors = [error for lag, error in correlation_errors if lag == 1]
    weights = np.asarray(
        [
            4.0 if lag == 1 else 2.0 if lag == 2 else 1.0
            for lag, _ in correlation_errors
        ],
        dtype=np.float64,
    )
    weighted_correlation_mae = (
        float(np.average(np.asarray(error_values), weights=weights))
        if error_values
        else float("inf")
    )
    maximum_lag1_error = max(lag1_errors, default=float("inf"))
    maximum_correlation = max(error_values, default=float("inf"))
    spatial_rmse = (
        float(np.sqrt(np.mean(np.square(error_values))))
        if error_values
        else float("inf")
    )
    maximum_row_column = max(row_column_errors, default=float("inf"))
    maximum_quantile = max(quantile_errors, default=float("inf"))
    checks = {
        "signal_bin_support": all(
            report["matched_signal_bins"] >= 3 for report in channel_reports.values()
        ),
        "per_channel_median_sigma_error": all(
            report["median_relative_sigma_error"]
            <= config.max_median_relative_sigma_error
            for report in channel_reports.values()
        ),
        "per_channel_p90_sigma_error": all(
            report["p90_relative_sigma_error"] <= config.max_p90_relative_sigma_error
            for report in channel_reports.values()
        ),
        "aggregate_median_sigma_error": median_sigma
        <= config.max_aggregate_median_relative_sigma_error,
        "aggregate_p90_sigma_error": p90_sigma
        <= config.max_aggregate_p90_relative_sigma_error,
        "weighted_correlation_mae": weighted_correlation_mae
        <= config.max_weighted_correlation_mae,
        "lag1_correlation_error": maximum_lag1_error
        <= config.max_lag1_correlation_absolute_error,
        "normalized_quantile_error": maximum_quantile
        <= config.max_normalized_quantile_error,
        "cross_channel_identity": maximum_cross
        <= config.max_cross_channel_correlation,
        "q4_quantization": float(synthetic["fraction_divisible_by_4"]) == 1.0,
        "legal_raw_fraction": float(synthetic["legal_raw_fraction"]) == 1.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "channels": channel_reports,
        "aggregate_median_relative_sigma_error": median_sigma,
        "aggregate_p90_relative_sigma_error": p90_sigma,
        "weighted_correlation_mae": weighted_correlation_mae,
        "maximum_lag1_correlation_absolute_error": maximum_lag1_error,
        "maximum_correlation_absolute_error": maximum_correlation,
        "spatial_correlation_rmse": spatial_rmse,
        "row_column_ratio_diagnostic": {
            "maximum_error": maximum_row_column,
            "tolerance": config.max_row_column_ratio_error,
            "within_tolerance": maximum_row_column
            <= config.max_row_column_ratio_error,
            "used_as_admission_gate": False,
        },
        "maximum_normalized_quantile_error": maximum_quantile,
        "maximum_cross_channel_correlation": maximum_cross,
    }


def _pairwise_spatial_comparison(
    source: A10Source,
    frame_range: tuple[int, int],
    config: A10bConfig,
    black: np.ndarray,
    shot: np.ndarray,
    read: np.ndarray,
    spatial_model: dict[str, Any],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    spectral_wins = 0
    for first_index, first, second in _iter_real_pairs(
        source, frame_range, config.pair_stride
    ):
        real_accumulator = _MetricAccumulator(config)
        real_accumulator.update(first, second, black, shot, read)
        real_report = real_accumulator.report()
        conditioning = 0.5 * (first + second)
        model_errors: dict[str, float] = {}
        for model_name in ("quantized_iid_q4", "a10b_spectral_q4"):
            model_accumulator = _MetricAccumulator(config)
            selected_model, quantization = _model_settings(
                model_name, config, spatial_model
            )
            for replica in range(config.validation_replicas):
                seed = (
                    config.seed
                    + 90_000_049
                    + first_index * 1009
                    + replica * 104729
                )
                rng = np.random.default_rng(seed)
                synthetic_first = generate_degraded_frame(
                    conditioning,
                    black,
                    shot,
                    read,
                    rng,
                    spatial_model=selected_model,
                    quantization_step_dn=quantization,
                    spectral_floor=config.spectral_floor,
                )
                synthetic_second = generate_degraded_frame(
                    conditioning,
                    black,
                    shot,
                    read,
                    rng,
                    spatial_model=selected_model,
                    quantization_step_dn=quantization,
                    spectral_floor=config.spectral_floor,
                )
                model_accumulator.update(
                    synthetic_first, synthetic_second, black, shot, read
                )
            comparison = _compare_metrics(
                real_report, model_accumulator.report(), config
            )
            model_errors[model_name] = float(comparison["weighted_correlation_mae"])
        won = model_errors["a10b_spectral_q4"] < model_errors["quantized_iid_q4"]
        spectral_wins += int(won)
        pairs.append(
            {
                "first_frame": first_index,
                "weighted_correlation_mae": model_errors,
                "spectral_won": won,
            }
        )
    return {
        "pairs": pairs,
        "spectral_wins": spectral_wins,
        "required_wins": config.min_validation_pair_wins,
        "passed": spectral_wins >= config.min_validation_pair_wins,
    }


def _smooth_conditioning_proxy(pair_mean: np.ndarray) -> np.ndarray:
    return np.stack(
        [cv2.GaussianBlur(channel, (0, 0), 3.0) for channel in pair_mean],
        axis=0,
    ).astype(np.float32)


def _visual_samples(
    source: A10Source,
    frame_range: tuple[int, int],
    config: A10bConfig,
    black: np.ndarray,
    shot: np.ndarray,
    read: np.ndarray,
    spatial_model: dict[str, Any],
    seed_offset: int,
) -> dict[str, np.ndarray]:
    first_index, first, second = next(
        _iter_real_pairs(source, frame_range, config.pair_stride)
    )
    proxy = _smooth_conditioning_proxy(0.5 * (first + second))
    rng_iid = np.random.default_rng(config.seed + seed_offset + first_index)
    iid = generate_degraded_frame(
        proxy,
        black,
        shot,
        read,
        rng_iid,
        spatial_model=None,
        quantization_step_dn=config.quantization_step_dn,
        spectral_floor=config.spectral_floor,
    )
    rng_spectral = np.random.default_rng(config.seed + seed_offset + first_index)
    spectral = generate_degraded_frame(
        proxy,
        black,
        shot,
        read,
        rng_spectral,
        spatial_model=spatial_model,
        quantization_step_dn=config.quantization_step_dn,
        spectral_floor=config.spectral_floor,
    )
    return {
        "real": first.astype(np.float32),
        "conditioning_proxy_not_gt": proxy,
        "quantized_iid_q4": iid,
        "a10b_spectral_q4": spectral,
    }


def validate_a10b_degradation(config: A10bConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config.validate()
    if Path(config.dataset_root).resolve() != Path(
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset"
    ).resolve():
        raise ValueError("A10b dataset root must remain the authoritative read-only root")
    if sha256_file(Path(config.a10_config_path)) != config.a10_config_sha256:
        raise ValueError("A10b A10 config SHA-256 mismatch")
    a10_config = load_a10_audit_config(Path(config.a10_config_path))
    a10_audit = _load_json_with_hash(config.a10_audit_path, config.a10_audit_sha256)
    profile = _load_json_with_hash(
        config.noise_profile_path, config.noise_profile_sha256
    )
    if profile.get("schema") != "a9-sensor-temporal-noise-v1":
        raise ValueError("A10b requires the validated A9 noise profile")
    if not profile.get("authenticity", {}).get("passed"):
        raise ValueError("A10b requires a passed A9 noise profile")
    spatial_gate = a10_audit["spatial_correlation_authenticity"]
    if not spatial_gate["per_channel_spatial_structure_passed"]:
        raise ValueError("A10b requires the A10 per-channel spatial gate")
    if spatial_gate["cross_channel_coupling_passed"]:
        raise ValueError("A10b protocol requires identity cross-channel coupling")
    if tuple(profile["fit"]["train_frame_range"]) != config.train_frame_range:
        raise ValueError("A10b fit range must match the frozen A9 split")
    if tuple(profile["fit"]["validation_frame_range"]) != config.validation_frame_range:
        raise ValueError("A10b validation range must match the frozen A9 split")
    noise_source = _source_by_name(a10_config, config.noise_source_name)
    hard_holdout = _source_by_name(a10_config, config.hard_holdout_name)
    if noise_source.role != "noise_statistics_source" or hard_holdout.role != "hard_holdout":
        raise ValueError("A10b source roles must remain frozen")
    for source in (noise_source, hard_holdout):
        if sha256_file(source.source.path) != source.source_sha256:
            raise ValueError(f"A10b source SHA-256 mismatch: {source.name}")
    black, shot, read = _variance_parameters(profile)
    real_train = _collect_real_metrics(
        noise_source, config.train_frame_range, config, black, shot, read
    )
    spatial_model = _fit_spatial_model(real_train, config)
    real_validation = _collect_real_metrics(
        noise_source, config.validation_frame_range, config, black, shot, read
    )
    validation_models: dict[str, Any] = {}
    for model_name in _MODEL_NAMES:
        synthetic = _collect_synthetic_metrics(
            noise_source,
            config.validation_frame_range,
            config.validation_replicas,
            config,
            black,
            shot,
            read,
            spatial_model,
            model_name,
            10_000_019,
        )
        validation_models[model_name] = {
            "metrics": synthetic,
            "comparison_to_real_validation": _compare_metrics(
                real_validation, synthetic, config
            ),
        }
    iid_comparison = validation_models["quantized_iid_q4"][
        "comparison_to_real_validation"
    ]
    spectral_comparison = validation_models["a10b_spectral_q4"][
        "comparison_to_real_validation"
    ]
    iid_rmse = float(iid_comparison["spatial_correlation_rmse"])
    spectral_rmse = float(spectral_comparison["spatial_correlation_rmse"])
    improvement = (iid_rmse - spectral_rmse) / max(iid_rmse, 1e-12)
    pairwise_comparison = _pairwise_spatial_comparison(
        noise_source,
        config.validation_frame_range,
        config,
        black,
        shot,
        read,
        spatial_model,
    )
    spectral_justified = (
        bool(spectral_comparison["passed"])
        and improvement >= config.min_spatial_rmse_improvement_over_q4_iid
        and bool(pairwise_comparison["passed"])
    )
    hard_holdout_real = _collect_real_metrics(
        hard_holdout,
        config.hard_holdout_frame_range,
        config,
        black,
        shot,
        read,
    )
    hard_holdout_models: dict[str, Any] = {}
    for model_name in ("quantized_iid_q4", "a10b_spectral_q4"):
        synthetic = _collect_synthetic_metrics(
            hard_holdout,
            config.hard_holdout_frame_range,
            config.hard_holdout_replicas,
            config,
            black,
            shot,
            read,
            spatial_model,
            model_name,
            40_000_037,
        )
        hard_holdout_models[model_name] = {
            "metrics": synthetic,
            "comparison_to_real_holdout": _compare_metrics(
                hard_holdout_real, synthetic, config
            ),
            "used_for_parameter_fitting_or_804x_model_selection": False,
        }
    iid_holdout_passed = bool(
        hard_holdout_models["quantized_iid_q4"]["comparison_to_real_holdout"][
            "passed"
        ]
    )
    spectral_holdout_passed = bool(
        hard_holdout_models["a10b_spectral_q4"]["comparison_to_real_holdout"][
            "passed"
        ]
    )
    if spectral_justified and spectral_holdout_passed:
        selected_model = "a10b_spectral_q4"
        selected_scope = "sc450ai_sensor_level_noise_statistics"
        status = "passed_correlated_generator_sensor_level"
    elif bool(iid_comparison["passed"]) and iid_holdout_passed:
        selected_model = "quantized_iid_q4"
        selected_scope = "sc450ai_sensor_level_noise_statistics"
        status = "passed_q4_iid_sensor_level_correlated_extension_rejected"
    elif spectral_justified:
        selected_model = "a10b_spectral_q4"
        selected_scope = "804x_capture_condition_only"
        status = "passed_804x_only_sensor_level_rejected"
    elif bool(iid_comparison["passed"]):
        selected_model = "quantized_iid_q4"
        selected_scope = "804x_capture_condition_only"
        status = "passed_804x_q4_iid_only_sensor_level_rejected"
    else:
        selected_model = None
        selected_scope = None
        status = "rejected_all_generators"
    result = {
        "schema": "a10b-sc450ai-degradation-validation-v1",
        "status": status,
        "selected_generator": selected_model,
        "selected_scope": selected_scope,
        "training_admission": {
            "model_training_allowed": False,
            "reason": (
                "A10b validates noise statistics only; sc450ai clean spatial detail, "
                "blur/MTF, and CFA response remain unidentified."
            ),
        },
        "fit": {
            "source": noise_source.name,
            "frame_range": list(config.train_frame_range),
            "real_metrics": real_train,
            "spatial_model": spatial_model,
            "validation_data_used_for_fit": False,
            "hard_holdout_used_for_fit": False,
        },
        "validation": {
            "source": noise_source.name,
            "frame_range": list(config.validation_frame_range),
            "real_metrics": real_validation,
            "models": validation_models,
            "spectral_spatial_rmse_improvement_over_q4_iid": improvement,
            "minimum_required_improvement": config.min_spatial_rmse_improvement_over_q4_iid,
            "pairwise_spatial_comparison": pairwise_comparison,
            "spectral_extension_justified": spectral_justified,
        },
        "hard_holdout": {
            "source": hard_holdout.name,
            "frame_range": list(config.hard_holdout_frame_range),
            "real_metrics": hard_holdout_real,
            "models": hard_holdout_models,
            "used_for_fitting": False,
            "used_for_804x_model_selection": False,
            "used_for_sensor_level_admission": True,
        },
        "identifiability_limits": [
            "PRNU",
            "DSNU",
            "sensor blur or MTF",
            "CFA spectral response transfer",
            "motion-detail clean target",
        ],
        "input_hashes": {
            "a10_config": config.a10_config_sha256,
            "a10_audit": config.a10_audit_sha256,
            "noise_profile": config.noise_profile_sha256,
            "noise_source": noise_source.source_sha256,
            "hard_holdout": hard_holdout.source_sha256,
        },
        "config": asdict(config),
    }
    visual = {}
    for key, source, frame_range, seed_offset in (
        ("validation", noise_source, config.validation_frame_range, 70_000_033),
        ("hard_holdout", hard_holdout, config.hard_holdout_frame_range, 80_000_041),
    ):
        samples = _visual_samples(
            source,
            frame_range,
            config,
            black,
            shot,
            read,
            spatial_model,
            seed_offset,
        )
        for sample_name, values in samples.items():
            visual[f"{key}_{sample_name}"] = values
    visual["black_levels"] = black
    return result, visual


def save_a10b_result(
    config: A10bConfig,
    result: dict[str, Any],
    visual: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "validation.json"
    visual_path = output_dir / "visual_samples.npz"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(visual_path, **visual)
    return result_path, visual_path

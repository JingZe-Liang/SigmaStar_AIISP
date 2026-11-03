"""Unified multi-sensor A9 fine-tuning with verified synthetic sensor noise."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from raw_fusion.finetune.a3 import A3Config, PhaseADataset, compute_loss, set_seed
from raw_fusion.finetune.a8_adaptation import (
    A8AdaptationConfig,
    StaticSplitHalfDataset,
    _build_from_a3,
    sha256_file,
    update_selected_ema,
)
from raw_fusion.finetune.cross_sensor import (
    NativeSequenceSpec,
    SATURATION_LIMIT_DN,
    TEMPORAL_FRAMES,
    open_sequence,
)
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, pack_frame


WHITE_LEVEL_DN = 4095.0
_PAIR_HIGHPASS_VARIANCE_GAIN = 8.0 / 9.0
_DEFAULT_SIGNAL_BIN_EDGES_DN = (
    0.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0,
    3072.0,
    3600.0,
)


@dataclass(frozen=True, slots=True)
class NoiseObservation:
    """One robust variance observation from a signal bin."""

    channel: str
    signal_dn: float
    sigma_dn: float
    pixels: int
    pair_first_frame: int


@dataclass(frozen=True, slots=True)
class NoiseProfileFitConfig:
    """Protocol for fitting temporal RAW noise from disjoint frame pairs."""

    dataset_root: str
    source: NativeSequenceSpec
    source_sha256: str
    black_levels: tuple[float, float, float, float]
    output_dir: str
    train_frame_range: tuple[int, int]
    validation_frame_range: tuple[int, int]
    pair_stride: int = 4
    spatial_sample_stride: int = 4
    flat_percentile: float = 35.0
    temporal_outlier_percentile: float = 99.5
    signal_bin_edges_dn: tuple[float, ...] = _DEFAULT_SIGNAL_BIN_EDGES_DN
    min_bin_pixels: int = 128
    min_validation_points_per_channel: int = 4
    max_median_relative_sigma_error: float = 0.25
    max_p90_relative_sigma_error: float = 0.50

    def validate(self) -> None:
        dataset_root = Path(self.dataset_root).resolve()
        if not self.source.path.resolve().is_relative_to(dataset_root):
            raise ValueError("noise source must be inside the authoritative dataset root")
        if self.source.reference_mode != "stress_only":
            raise ValueError("noise fitting source must be declared stress_only")
        if len(self.black_levels) != 4:
            raise ValueError("black_levels must contain four packed-channel values")
        if any(value < 0.0 or value >= WHITE_LEVEL_DN for value in self.black_levels):
            raise ValueError("black levels must be within the 12-bit RAW range")
        train_start, train_end = self.train_frame_range
        validation_start, validation_end = self.validation_frame_range
        if not (0 <= train_start < train_end <= self.source.frames_used):
            raise ValueError("invalid training frame range")
        if not (0 <= validation_start < validation_end <= self.source.frames_used):
            raise ValueError("invalid validation frame range")
        if max(train_start, validation_start) < min(train_end, validation_end):
            raise ValueError("noise-profile training and validation ranges must be disjoint")
        if train_end - train_start < 2 or validation_end - validation_start < 2:
            raise ValueError("each frame range must contain at least one adjacent pair")
        if self.pair_stride < 2 or self.spatial_sample_stride <= 0:
            raise ValueError("pair and spatial strides must be positive and disjoint")
        if not 0.0 < self.flat_percentile <= 100.0:
            raise ValueError("flat_percentile must be in (0, 100]")
        if not 90.0 <= self.temporal_outlier_percentile < 100.0:
            raise ValueError("temporal_outlier_percentile must be in [90, 100)")
        if len(self.signal_bin_edges_dn) < 3:
            raise ValueError("at least three signal-bin edges are required")
        if any(
            high <= low
            for low, high in zip(
                self.signal_bin_edges_dn[:-1],
                self.signal_bin_edges_dn[1:],
                strict=True,
            )
        ):
            raise ValueError("signal-bin edges must be strictly increasing")
        if self.min_bin_pixels <= 0 or self.min_validation_points_per_channel < 2:
            raise ValueError("noise-profile sample thresholds must be positive")
        if self.max_median_relative_sigma_error <= 0.0:
            raise ValueError("median authenticity threshold must be positive")
        if self.max_p90_relative_sigma_error <= 0.0:
            raise ValueError("p90 authenticity threshold must be positive")


@dataclass(frozen=True, slots=True)
class A9StaticSource:
    """One static split-half source in the unified training mixture."""

    name: str
    source: NativeSequenceSpec
    reference_dir: str
    source_sha256: str
    weight: float


@dataclass(frozen=True, slots=True)
class A9SyntheticSource:
    """One validated sensor-noise profile in the unified training mixture."""

    name: str
    sensor: str
    profile_path: str
    profile_sha256: str
    weight: float


@dataclass(frozen=True, slots=True)
class A9CleanReference:
    """An existing clean/static reference used only as synthetic scene content."""

    name: str
    reference_path: str
    reference_sha256: str
    valid_region_path: str
    valid_region_sha256: str
    black_levels: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class A9GeneralizationConfig:
    """Reproducible protocol for one unified multi-sensor checkpoint."""

    dataset_root: str
    a3_checkpoint: str
    a3_checkpoint_sha256: str
    replay_supervision_dir: str
    replay_noise_dir: str
    replay_weight: float
    static_sources: tuple[A9StaticSource, ...]
    synthetic_sources: tuple[A9SyntheticSource, ...]
    clean_references: tuple[A9CleanReference, ...]
    output_dir: str
    seed: int = 20261016
    steps: int = 100
    batch_size: int = 1
    learning_rate: float = 5e-6
    patch_size: int = 64
    gradient_weight: float = 0.1
    highpass_weight: float = 0.1
    min_valid_fraction: float = 0.25
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    checkpoint_interval: int = 25
    precision: str = "bfloat16"
    trainable_scope: str = "input_output_heads"
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.patch_size != 64:
            raise ValueError("RViDeformer A9 requires packed patch size 64")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.gradient_clip_norm <= 0.0 or self.checkpoint_interval <= 0:
            raise ValueError("gradient clip norm and checkpoint interval must be positive")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if self.trainable_scope != "input_output_heads":
            raise ValueError("trainable_scope must be input_output_heads")
        if not self.static_sources or not self.synthetic_sources:
            raise ValueError("A9 requires both static and synthetic target-sensor sources")
        if not self.clean_references:
            raise ValueError("A9 synthetic supervision requires clean scene references")
        dataset_root = Path(self.dataset_root).resolve()
        names = [source.name for source in self.static_sources]
        names.extend(source.name for source in self.synthetic_sources)
        if len(names) != len(set(names)):
            raise ValueError("A9 source names must be unique")
        for source in self.static_sources:
            if not source.source.path.resolve().is_relative_to(dataset_root):
                raise ValueError("static source must be inside the authoritative dataset root")
            if source.source.reference_mode != "motion_aware_split_half":
                raise ValueError("static A9 sources require split-half references")
            if source.weight <= 0.0:
                raise ValueError("static source weights must be positive")
        for source in self.synthetic_sources:
            if source.weight <= 0.0:
                raise ValueError("synthetic source weights must be positive")
        total_weight = self.replay_weight
        total_weight += sum(source.weight for source in self.static_sources)
        total_weight += sum(source.weight for source in self.synthetic_sources)
        if not np.isclose(total_weight, 1.0, atol=1e-9):
            raise ValueError(f"A9 source weights must sum to 1.0, got {total_weight}")
        if self.replay_weight <= 0.0:
            raise ValueError("replay_weight must be positive")

    def with_runtime_overrides(
        self,
        *,
        steps: int | None = None,
        output_dir: Path | None = None,
        device: str | None = None,
    ) -> "A9GeneralizationConfig":
        return replace(
            self,
            steps=self.steps if steps is None else steps,
            output_dir=self.output_dir if output_dir is None else str(output_dir),
            device=self.device if device is None else device,
        )


def _source_mapping(source: NativeSequenceSpec) -> dict[str, object]:
    mapping = asdict(source)
    mapping["path"] = str(source.path)
    mapping["panel_crops"] = {
        name: list(coordinates) for name, coordinates in source.panel_crops
    }
    return mapping


def load_noise_profile_config(path: Path) -> NoiseProfileFitConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_payload = dict(payload["source"])
    source_name = str(source_payload.pop("name"))
    config = NoiseProfileFitConfig(
        dataset_root=str(payload["dataset_root"]),
        source=NativeSequenceSpec.from_mapping(source_name, source_payload),
        source_sha256=str(payload["source_sha256"]),
        black_levels=tuple(float(value) for value in payload["black_levels"]),
        output_dir=str(payload["output_dir"]),
        train_frame_range=tuple(int(value) for value in payload["train_frame_range"]),
        validation_frame_range=tuple(
            int(value) for value in payload["validation_frame_range"]
        ),
        pair_stride=int(payload.get("pair_stride", 4)),
        spatial_sample_stride=int(payload.get("spatial_sample_stride", 4)),
        flat_percentile=float(payload.get("flat_percentile", 35.0)),
        temporal_outlier_percentile=float(
            payload.get("temporal_outlier_percentile", 99.5)
        ),
        signal_bin_edges_dn=tuple(
            float(value)
            for value in payload.get(
                "signal_bin_edges_dn", _DEFAULT_SIGNAL_BIN_EDGES_DN
            )
        ),
        min_bin_pixels=int(payload.get("min_bin_pixels", 128)),
        min_validation_points_per_channel=int(
            payload.get("min_validation_points_per_channel", 4)
        ),
        max_median_relative_sigma_error=float(
            payload.get("max_median_relative_sigma_error", 0.25)
        ),
        max_p90_relative_sigma_error=float(
            payload.get("max_p90_relative_sigma_error", 0.50)
        ),
    )
    config.validate()
    return config


def noise_profile_config_mapping(config: NoiseProfileFitConfig) -> dict[str, object]:
    mapping = asdict(config)
    mapping["source"] = {"name": config.source.name, **_source_mapping(config.source)}
    return mapping


def _parse_static_source(payload: dict[str, object]) -> A9StaticSource:
    source_payload = dict(payload["source"])
    source_name = str(source_payload.pop("name"))
    return A9StaticSource(
        name=str(payload["name"]),
        source=NativeSequenceSpec.from_mapping(source_name, source_payload),
        reference_dir=str(payload["reference_dir"]),
        source_sha256=str(payload["source_sha256"]),
        weight=float(payload["weight"]),
    )


def load_generalization_config(path: Path) -> A9GeneralizationConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = A9GeneralizationConfig(
        dataset_root=str(payload["dataset_root"]),
        a3_checkpoint=str(payload["a3_checkpoint"]),
        a3_checkpoint_sha256=str(payload["a3_checkpoint_sha256"]),
        replay_supervision_dir=str(payload["replay_supervision_dir"]),
        replay_noise_dir=str(payload["replay_noise_dir"]),
        replay_weight=float(payload["replay_weight"]),
        static_sources=tuple(
            _parse_static_source(item) for item in payload["static_sources"]
        ),
        synthetic_sources=tuple(
            A9SyntheticSource(
                name=str(item["name"]),
                sensor=str(item["sensor"]),
                profile_path=str(item["profile_path"]),
                profile_sha256=str(item["profile_sha256"]),
                weight=float(item["weight"]),
            )
            for item in payload["synthetic_sources"]
        ),
        clean_references=tuple(
            A9CleanReference(
                name=str(item["name"]),
                reference_path=str(item["reference_path"]),
                reference_sha256=str(item["reference_sha256"]),
                valid_region_path=str(item["valid_region_path"]),
                valid_region_sha256=str(item["valid_region_sha256"]),
                black_levels=tuple(float(value) for value in item["black_levels"]),
            )
            for item in payload["clean_references"]
        ),
        output_dir=str(payload["output_dir"]),
        seed=int(payload.get("seed", 20261016)),
        steps=int(payload.get("steps", 100)),
        batch_size=int(payload.get("batch_size", 1)),
        learning_rate=float(payload.get("learning_rate", 5e-6)),
        patch_size=int(payload.get("patch_size", 64)),
        gradient_weight=float(payload.get("gradient_weight", 0.1)),
        highpass_weight=float(payload.get("highpass_weight", 0.1)),
        min_valid_fraction=float(payload.get("min_valid_fraction", 0.25)),
        ema_decay=float(payload.get("ema_decay", 0.999)),
        gradient_clip_norm=float(payload.get("gradient_clip_norm", 1.0)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 25)),
        precision=str(payload.get("precision", "bfloat16")),
        trainable_scope=str(payload.get("trainable_scope", "input_output_heads")),
        device=str(payload.get("device", "cuda:0")),
    )
    config.validate()
    return config


def generalization_config_mapping(config: A9GeneralizationConfig) -> dict[str, object]:
    mapping = asdict(config)
    mapping["static_sources"] = [
        {
            "name": source.name,
            "source": {"name": source.source.name, **_source_mapping(source.source)},
            "reference_dir": source.reference_dir,
            "source_sha256": source.source_sha256,
            "weight": source.weight,
        }
        for source in config.static_sources
    ]
    return mapping


def _robust_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad / np.sqrt(_PAIR_HIGHPASS_VARIANCE_GAIN)


def _measure_noise_observations(
    config: NoiseProfileFitConfig,
    frame_range: tuple[int, int],
) -> list[NoiseObservation]:
    reader = open_sequence(config.source)
    observations: list[NoiseObservation] = []
    black = np.asarray(config.black_levels, dtype=np.float32)
    start, end = frame_range
    stride = config.spatial_sample_stride
    edges = config.signal_bin_edges_dn
    for first_index in range(start, end - 1, config.pair_stride):
        first = pack_frame(
            reader.read_frame(first_index), config.source.adapter
        ).astype(np.float32)[:, ::stride, ::stride]
        second = pack_frame(
            reader.read_frame(first_index + 1), config.source.adapter
        ).astype(np.float32)[:, ::stride, ::stride]
        pair_mean = 0.5 * (first + second)
        difference = (first - second) / np.sqrt(2.0)
        for channel, channel_name in enumerate(CHANNEL_NAMES):
            mean_channel = pair_mean[channel]
            signal = mean_channel - black[channel]
            spatial_highpass = mean_channel - cv2.blur(mean_channel, (3, 3))
            temporal_highpass = difference[channel] - cv2.blur(
                difference[channel], (3, 3)
            )
            base_valid = (
                (signal >= edges[0])
                & (signal < edges[-1])
                & (mean_channel < SATURATION_LIMIT_DN)
            )
            if int(base_valid.sum()) < config.min_bin_pixels:
                continue
            flat_limit = float(
                np.percentile(
                    np.abs(spatial_highpass[base_valid]), config.flat_percentile
                )
            )
            flat_valid = base_valid & (np.abs(spatial_highpass) <= flat_limit)
            temporal_limit = float(
                np.percentile(
                    np.abs(temporal_highpass[flat_valid]),
                    config.temporal_outlier_percentile,
                )
            )
            valid = flat_valid & (np.abs(temporal_highpass) <= temporal_limit)
            for low, high in zip(edges[:-1], edges[1:], strict=True):
                selected = valid & (signal >= low) & (signal < high)
                pixels = int(selected.sum())
                if pixels < config.min_bin_pixels:
                    continue
                sigma = _robust_sigma(temporal_highpass[selected].astype(np.float64))
                if sigma <= 0.0 or not np.isfinite(sigma):
                    continue
                observations.append(
                    NoiseObservation(
                        channel=channel_name,
                        signal_dn=float(np.median(signal[selected])),
                        sigma_dn=sigma,
                        pixels=pixels,
                        pair_first_frame=first_index,
                    )
                )
    return observations


def fit_variance_models(
    observations: list[NoiseObservation],
) -> dict[str, dict[str, float]]:
    """Fit non-negative variance = shot * signal + read using Huber IRLS."""

    models: dict[str, dict[str, float]] = {}
    for channel in CHANNEL_NAMES:
        selected = [item for item in observations if item.channel == channel]
        if len(selected) < 3:
            raise ValueError(f"insufficient noise observations for channel {channel}")
        signal = np.asarray([item.signal_dn for item in selected], dtype=np.float64)
        variance = np.square(
            np.asarray([item.sigma_dn for item in selected], dtype=np.float64)
        )
        pixel_weights = np.sqrt(
            np.asarray([item.pixels for item in selected], dtype=np.float64)
        )
        pixel_weights /= np.median(pixel_weights)
        features = np.column_stack((signal, np.ones_like(signal)))
        robust_weights = np.ones_like(signal)
        coefficients = np.zeros(2, dtype=np.float64)
        for _ in range(30):
            weights = pixel_weights * robust_weights
            weighted_features = features * np.sqrt(weights)[:, None]
            weighted_variance = variance * np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                weighted_features, weighted_variance, rcond=None
            )[0]
            coefficients = np.maximum(coefficients, 0.0)
            residual = variance - features @ coefficients
            center = float(np.median(residual))
            scale = 1.4826 * float(np.median(np.abs(residual - center)))
            if scale <= 1e-9:
                break
            robust_weights = np.minimum(
                1.0, 1.5 * scale / np.maximum(np.abs(residual - center), 1e-9)
            )
        prediction = features @ coefficients
        models[channel] = {
            "shot_variance_per_signal_dn": float(coefficients[0]),
            "read_variance_dn2": float(coefficients[1]),
            "fit_rmse_dn2": float(np.sqrt(np.mean(np.square(variance - prediction)))),
            "observations": len(selected),
        }
    return models


def _authenticity_report(
    models: dict[str, dict[str, float]],
    observations: list[NoiseObservation],
    config: NoiseProfileFitConfig,
) -> dict[str, object]:
    channel_reports: dict[str, object] = {}
    all_errors: list[float] = []
    checks: dict[str, bool] = {}
    for channel in CHANNEL_NAMES:
        selected = [item for item in observations if item.channel == channel]
        model = models[channel]
        errors = []
        for item in selected:
            predicted_variance = (
                model["shot_variance_per_signal_dn"] * item.signal_dn
                + model["read_variance_dn2"]
            )
            predicted_sigma = np.sqrt(max(predicted_variance, 1e-12))
            errors.append(abs(item.sigma_dn - predicted_sigma) / item.sigma_dn)
        enough_points = len(errors) >= config.min_validation_points_per_channel
        median_error = float(np.median(errors)) if errors else float("inf")
        p90_error = float(np.percentile(errors, 90.0)) if errors else float("inf")
        channel_passed = (
            enough_points
            and median_error <= config.max_median_relative_sigma_error
            and p90_error <= config.max_p90_relative_sigma_error
        )
        checks[f"{channel}_points"] = enough_points
        checks[f"{channel}_median_error"] = (
            median_error <= config.max_median_relative_sigma_error
        )
        checks[f"{channel}_p90_error"] = (
            p90_error <= config.max_p90_relative_sigma_error
        )
        channel_reports[channel] = {
            "points": len(errors),
            "median_relative_sigma_error": median_error,
            "p90_relative_sigma_error": p90_error,
            "passed": channel_passed,
        }
        all_errors.extend(errors)
    aggregate_median = float(np.median(all_errors)) if all_errors else float("inf")
    aggregate_p90 = (
        float(np.percentile(all_errors, 90.0)) if all_errors else float("inf")
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "channels": channel_reports,
        "aggregate_median_relative_sigma_error": aggregate_median,
        "aggregate_p90_relative_sigma_error": aggregate_p90,
    }


def _observation_mapping(item: NoiseObservation) -> dict[str, object]:
    return asdict(item)


def build_sensor_noise_profile(config: NoiseProfileFitConfig) -> dict[str, object]:
    """Fit a profile on training pairs and gate it on disjoint validation pairs."""

    config.validate()
    if sha256_file(config.source.path) != config.source_sha256:
        raise ValueError("noise-profile source SHA-256 does not match configuration")
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"noise-profile output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_observations = _measure_noise_observations(
        config, config.train_frame_range
    )
    validation_observations = _measure_noise_observations(
        config, config.validation_frame_range
    )
    models = fit_variance_models(train_observations)
    authenticity = _authenticity_report(models, validation_observations, config)
    profile = {
        "schema": "a9-sensor-temporal-noise-v1",
        "sensor": config.source.sensor,
        "source_name": config.source.name,
        "source_path": str(config.source.path.resolve()),
        "source_sha256": config.source_sha256,
        "black_levels": {
            name: config.black_levels[index]
            for index, name in enumerate(CHANNEL_NAMES)
        },
        "variance_model": models,
        "channel_correlation": {
            "policy": "identity",
            "matrix": np.eye(4, dtype=np.float64).tolist(),
            "reason": (
                "packed CFA channels are spatially offset; pairwise temporal variance "
                "is modeled without unsupported cross-channel coupling"
            ),
        },
        "fit": {
            "train_frame_range": list(config.train_frame_range),
            "validation_frame_range": list(config.validation_frame_range),
            "train_observations": [
                _observation_mapping(item) for item in train_observations
            ],
            "validation_observations": [
                _observation_mapping(item) for item in validation_observations
            ],
            "method": (
                "adjacent-pair difference divided by sqrt(2), 3x3 high-pass "
                "variance correction, flat-pixel selection, non-negative Huber IRLS"
            ),
        },
        "authenticity": authenticity,
        "quantization": {
            "domain": "12bit_dn",
            "round_before_clip": True,
            "clip_range_dn": [0.0, WHITE_LEVEL_DN],
        },
        "config": noise_profile_config_mapping(config),
    }
    profile_path = output_dir / "noise_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="ascii")
    summary = {
        "status": "passed" if authenticity["passed"] else "rejected",
        "profile": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
        "train_observations": len(train_observations),
        "validation_observations": len(validation_observations),
        "authenticity": authenticity,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    return summary


def generate_profile_noisy_frame(
    clean_normalized: np.ndarray,
    profile: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one quantized packed frame from a validated A9 noise profile."""

    if clean_normalized.ndim != 3 or clean_normalized.shape[0] != 4:
        raise ValueError("clean_normalized must have shape (4,H,W)")
    authenticity = profile["authenticity"]
    if not isinstance(authenticity, dict) or not authenticity.get("passed", False):
        raise ValueError("synthetic training requires a passed noise-profile gate")
    black_mapping = profile["black_levels"]
    variance_mapping = profile["variance_model"]
    if not isinstance(black_mapping, dict) or not isinstance(variance_mapping, dict):
        raise TypeError("noise profile black levels and variance model must be mappings")
    black = np.asarray(
        [float(black_mapping[name]) for name in CHANNEL_NAMES], dtype=np.float64
    )
    signal = np.clip(clean_normalized.astype(np.float64), 0.0, 1.0)
    signal *= (WHITE_LEVEL_DN - black)[:, None, None]
    variance = np.empty_like(signal)
    for channel, name in enumerate(CHANNEL_NAMES):
        channel_model = variance_mapping[name]
        if not isinstance(channel_model, dict):
            raise TypeError("per-channel variance model must be a mapping")
        variance[channel] = np.maximum(
            float(channel_model["shot_variance_per_signal_dn"]) * signal[channel]
            + float(channel_model["read_variance_dn2"]),
            0.0,
        )
    noisy = black[:, None, None] + signal
    noisy += np.sqrt(variance) * rng.standard_normal(signal.shape)
    return np.clip(np.rint(noisy), 0.0, WHITE_LEVEL_DN).astype(np.uint16)


def _augment(
    inputs: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        inputs = inputs[..., ::-1]
        target = target[..., ::-1]
        mask = mask[..., ::-1]
    if rng.random() < 0.5:
        inputs = inputs[..., ::-1, :]
        target = target[..., ::-1, :]
        mask = mask[..., ::-1, :]
    if rng.random() < 0.5:
        inputs = inputs.swapaxes(-2, -1)
        target = target.swapaxes(-2, -1)
        mask = mask.swapaxes(-2, -1)
    return inputs, target, mask


class SyntheticSensorNoiseDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Clean existing scene patches corrupted with one verified sensor profile."""

    def __init__(
        self,
        config: A9GeneralizationConfig,
        source: A9SyntheticSource,
        *,
        samples: int,
        source_offset: int,
    ) -> None:
        self.config = config
        self.source = source
        self.samples = samples
        self.source_offset = source_offset
        self.profile = json.loads(Path(source.profile_path).read_text(encoding="utf-8"))
        authenticity = self.profile.get("authenticity")
        if not isinstance(authenticity, dict) or not authenticity.get("passed", False):
            raise ValueError(f"noise profile failed authenticity gate: {source.name}")
        self.references = [
            (
                np.load(item.reference_path, mmap_mode="r"),
                np.load(item.valid_region_path, mmap_mode="r"),
                np.asarray(item.black_levels, dtype=np.float32),
            )
            for item in config.clean_references
        ]
        for reference, valid, black in self.references:
            if reference.ndim != 3 or reference.shape[0] != 4:
                raise ValueError("clean A9 reference must have shape (4,H,W)")
            if valid.shape != reference.shape[1:]:
                raise ValueError("clean A9 valid region must match reference shape")
            if black.shape != (4,):
                raise ValueError("clean A9 black vector must contain four values")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = np.random.default_rng(
            self.config.seed + self.source_offset + index * 104729
        )
        reference, valid_region, source_black = self.references[
            int(rng.integers(0, len(self.references)))
        ]
        patch = self.config.patch_size
        height, width = reference.shape[1:]
        for _ in range(64):
            y = int(rng.integers(0, height - patch + 1))
            x = int(rng.integers(0, width - patch + 1))
            valid = np.asarray(
                valid_region[y : y + patch, x : x + patch], dtype=bool
            ).copy()
            if float(valid.mean()) >= self.config.min_valid_fraction:
                break
        else:
            raise RuntimeError("failed to sample a valid A9 synthetic clean patch")
        clean_dn = np.asarray(
            reference[:, y : y + patch, x : x + patch], dtype=np.float32
        )
        clean_normalized = np.clip(
            (clean_dn - source_black[:, None, None])
            / (WHITE_LEVEL_DN - source_black[:, None, None]),
            0.0,
            1.0,
        )
        black_mapping = self.profile["black_levels"]
        if not isinstance(black_mapping, dict):
            raise TypeError("noise profile black levels must be a mapping")
        target_black = np.asarray(
            [float(black_mapping[name]) for name in CHANNEL_NAMES], dtype=np.float32
        )
        noisy_frames = np.stack(
            [
                generate_profile_noisy_frame(clean_normalized, self.profile, rng)
                for _ in range(TEMPORAL_FRAMES)
            ],
            axis=0,
        ).astype(np.float32)
        inputs = np.clip(
            (noisy_frames - target_black[None, :, None, None])
            / (WHITE_LEVEL_DN - target_black[None, :, None, None]),
            0.0,
            1.0,
        )
        target = np.broadcast_to(
            clean_normalized, (TEMPORAL_FRAMES, 4, patch, patch)
        ).copy()
        valid &= np.all(noisy_frames < SATURATION_LIMIT_DN, axis=(0, 1))
        mask = np.broadcast_to(valid, (TEMPORAL_FRAMES, patch, patch)).copy()
        inputs, target, mask = _augment(inputs, target, mask, rng)
        arrays = (inputs, target, mask.astype(np.float32))
        return tuple(
            torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
            for array in arrays
        )


class UnifiedA9Dataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Deterministic weighted mixture for one sensor-agnostic A9 checkpoint."""

    def __init__(self, config: A9GeneralizationConfig, *, samples: int) -> None:
        self.config = config
        self.samples = samples
        replay_config = A3Config(
            dataset_root=config.dataset_root,
            supervision_dir=config.replay_supervision_dir,
            noise_dir=config.replay_noise_dir,
            output_dir=config.output_dir,
            checkpoint=config.a3_checkpoint,
            seed=config.seed + 1_000_003,
            steps=config.steps,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            holdout_gain="1000x",
            device=config.device,
        )
        datasets: list[Dataset[tuple[Tensor, Tensor, Tensor]]] = [
            PhaseADataset(replay_config, samples=samples)
        ]
        names = ["mis20s1_replay"]
        weights = [config.replay_weight]
        for source_index, source in enumerate(config.static_sources):
            static_config = A8AdaptationConfig(
                dataset_root=config.dataset_root,
                source=source.source,
                reference_dir=source.reference_dir,
                a3_checkpoint=config.a3_checkpoint,
                a3_checkpoint_sha256=config.a3_checkpoint_sha256,
                source_sha256=source.source_sha256,
                replay_supervision_dir=config.replay_supervision_dir,
                replay_noise_dir=config.replay_noise_dir,
                output_dir=config.output_dir,
                seed=config.seed + 2_000_003 + source_index * 10_007,
                steps=config.steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                sensor_fraction=0.5,
                patch_size=config.patch_size,
                gradient_weight=config.gradient_weight,
                highpass_weight=config.highpass_weight,
                min_valid_fraction=config.min_valid_fraction,
                ema_decay=config.ema_decay,
                gradient_clip_norm=config.gradient_clip_norm,
                checkpoint_interval=config.checkpoint_interval,
                precision=config.precision,
                device=config.device,
            )
            datasets.append(StaticSplitHalfDataset(static_config, samples=samples))
            names.append(source.name)
            weights.append(source.weight)
        for source_index, source in enumerate(config.synthetic_sources):
            datasets.append(
                SyntheticSensorNoiseDataset(
                    config,
                    source,
                    samples=samples,
                    source_offset=3_000_001 + source_index * 20_011,
                )
            )
            names.append(source.name)
            weights.append(source.weight)
        self.datasets = datasets
        self.names = names
        self.cumulative_weights = np.cumsum(np.asarray(weights, dtype=np.float64))
        self.cumulative_weights[-1] = 1.0

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        selector = np.random.default_rng(
            self.config.seed + index * 32452843
        ).random()
        source_id = int(np.searchsorted(self.cumulative_weights, selector, side="right"))
        inputs, target, mask = self.datasets[source_id][index]
        return inputs, target, mask, torch.tensor(source_id, dtype=torch.int64)


def audit_generalization_inputs(
    config: A9GeneralizationConfig,
) -> dict[str, object]:
    """Verify hashes and materialize one real sample from every A9 source."""

    config.validate()
    _verify_training_inputs(config)
    samples = max(1024, config.steps * config.batch_size)
    dataset = UnifiedA9Dataset(config, samples=samples)
    source_reports: dict[str, object] = {}
    expected_shape = (TEMPORAL_FRAMES, 4, config.patch_size, config.patch_size)
    expected_mask_shape = (TEMPORAL_FRAMES, config.patch_size, config.patch_size)
    for source_id, (name, source_dataset) in enumerate(
        zip(dataset.names, dataset.datasets, strict=True)
    ):
        inputs, target, mask = source_dataset[source_id]
        if tuple(inputs.shape) != expected_shape or tuple(target.shape) != expected_shape:
            raise ValueError(f"A9 source {name} returned an unexpected tensor shape")
        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(f"A9 source {name} returned an unexpected mask shape")
        if not torch.isfinite(inputs).all() or not torch.isfinite(target).all():
            raise ValueError(f"A9 source {name} returned non-finite values")
        if float(mask.mean()) < config.min_valid_fraction:
            raise ValueError(f"A9 source {name} returned an insufficient valid mask")
        source_reports[name] = {
            "source_id": source_id,
            "input_shape": list(inputs.shape),
            "target_shape": list(target.shape),
            "mask_shape": list(mask.shape),
            "input_range": [float(inputs.min()), float(inputs.max())],
            "target_range": [float(target.min()), float(target.max())],
            "mask_fraction": float(mask.mean()),
        }
    scheduled_counts = {name: 0 for name in dataset.names}
    for index in range(config.steps * config.batch_size):
        selector = np.random.default_rng(
            config.seed + index * 32452843
        ).random()
        source_id = int(
            np.searchsorted(dataset.cumulative_weights, selector, side="right")
        )
        scheduled_counts[dataset.names[source_id]] += 1
    return {
        "status": "passed",
        "steps": config.steps,
        "batch_size": config.batch_size,
        "scheduled_source_counts": scheduled_counts,
        "sources": source_reports,
        "verified": {
            "a3_checkpoint_sha256": config.a3_checkpoint_sha256,
            "static_source_sha256": {
                source.name: source.source_sha256 for source in config.static_sources
            },
            "noise_profile_sha256": {
                source.name: source.profile_sha256
                for source in config.synthetic_sources
            },
            "clean_reference_sha256": {
                source.name: source.reference_sha256
                for source in config.clean_references
            },
        },
    }


def configure_input_output_head_training(
    model: nn.Module,
) -> tuple[list[nn.Parameter], dict[str, object]]:
    """Freeze the Transformer trunk and expose only RAW input/output heads."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_prefixes = (
        "conv_first.",
        "norm.",
        "conv_after_body.",
        "conv_last.",
    )
    names: list[str] = []
    parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad_(True)
            names.append(name)
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("RViDeformer did not expose A9 input/output head parameters")
    return parameters, {
        "scope": "input_output_heads",
        "prefixes": list(trainable_prefixes),
        "parameter_tensors": len(parameters),
        "parameters": sum(parameter.numel() for parameter in parameters),
        "names": names,
    }


def _verify_training_inputs(config: A9GeneralizationConfig) -> None:
    if sha256_file(Path(config.a3_checkpoint)) != config.a3_checkpoint_sha256:
        raise ValueError("A3 checkpoint SHA-256 does not match the A9 protocol")
    for source in config.static_sources:
        if sha256_file(source.source.path) != source.source_sha256:
            raise ValueError(f"static source SHA-256 mismatch: {source.name}")
    for source in config.synthetic_sources:
        profile_path = Path(source.profile_path)
        if sha256_file(profile_path) != source.profile_sha256:
            raise ValueError(f"noise profile SHA-256 mismatch: {source.name}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        authenticity = profile.get("authenticity")
        if not isinstance(authenticity, dict) or not authenticity.get("passed", False):
            raise ValueError(f"noise profile did not pass authenticity: {source.name}")
    for reference in config.clean_references:
        if sha256_file(Path(reference.reference_path)) != reference.reference_sha256:
            raise ValueError(f"clean reference SHA-256 mismatch: {reference.name}")
        if sha256_file(Path(reference.valid_region_path)) != reference.valid_region_sha256:
            raise ValueError(f"valid-region SHA-256 mismatch: {reference.name}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: A9GeneralizationConfig,
    source_counts: dict[str, int],
    trainable_report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "a9-unified-multi-sensor-generalization-v1",
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": generalization_config_mapping(config),
            "source_counts": source_counts,
            "trainable": trainable_report,
            "initialization": {
                "checkpoint": str(Path(config.a3_checkpoint).resolve()),
                "checkpoint_sha256": config.a3_checkpoint_sha256,
                "weights": "ema",
            },
        },
        path,
    )


def train_generalization_model(config: A9GeneralizationConfig) -> dict[str, object]:
    """Fine-tune one A3-derived checkpoint on the verified A9 source mixture."""

    config.validate()
    _verify_training_inputs(config)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"A9 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A9 requires CUDA; torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    set_seed(config.seed)
    (output_dir / "config.json").write_text(
        json.dumps(generalization_config_mapping(config), indent=2) + "\n",
        encoding="ascii",
    )
    samples = max(1024, config.steps * config.batch_size)
    dataset = UnifiedA9Dataset(config, samples=samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    checkpoint = Path(config.a3_checkpoint)
    model = _build_from_a3(checkpoint, device)
    ema = _build_from_a3(checkpoint, device)
    trainable_parameters, trainable_report = configure_input_output_head_training(model)
    trainable_names = {str(name) for name in trainable_report["names"]}
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
    source_counts = {name: 0 for name in dataset.names}
    iterator = iter(loader)
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, config.steps + 1):
            try:
                inputs, targets, masks, source_ids = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets, masks, source_ids = next(iterator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device).unsqueeze(2)
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and config.precision == "bfloat16",
            ):
                prediction = model(inputs).clamp(0.0, 1.0)
            loss, metrics = compute_loss(
                prediction.float(),
                targets,
                masks,
                config.gradient_weight,
                config.highpass_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite A9 loss at step {step}: {metrics}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip_norm
                )
                .detach()
                .cpu()
            )
            optimizer.step()
            update_selected_ema(ema, model, trainable_names, config.ema_decay)
            batch_sources = [dataset.names[int(value)] for value in source_ids.tolist()]
            for source_name in batch_sources:
                source_counts[source_name] += 1
            record = {
                "step": step,
                "sources": batch_sources,
                "gradient_norm": gradient_norm,
                **metrics,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(
                    f"[a9] step={step}/{config.steps} source={','.join(batch_sources)} "
                    f"loss={metrics['loss']:.6f} grad={gradient_norm:.4f}",
                    flush=True,
                )
            if step % config.checkpoint_interval == 0 or step == config.steps:
                _save_checkpoint(
                    output_dir / f"checkpoint_step_{step:06d}.pth",
                    model,
                    ema,
                    optimizer,
                    step,
                    config,
                    source_counts,
                    trainable_report,
                )
    summary = {
        "status": "completed",
        "steps": config.steps,
        "source_counts": source_counts,
        "trainable": trainable_report,
        "checkpoint": str(
            (output_dir / f"checkpoint_step_{config.steps:06d}.pth").resolve()
        ),
        "log": str(log_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    return summary

"""Native-domain zero-shot evaluation helpers for additional RAW sensors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from raw_fusion.finetune.a4 import infer_video
from raw_fusion.finetune.adapters import SourceAdapter
from raw_fusion.finetune.stats import (
    MIN_SIGNAL_PIXELS,
    SIGNAL_MARGIN_DN,
    compute_basic_stats,
    frame_drift_stats,
)
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, RawStreamReader, pack_frame, pack_frames

WHITE_LEVEL_DN = 4095.0
SATURATION_LIMIT_DN = 3800.0
TEMPORAL_FRAMES = 8
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class NativeSequenceSpec:
    """Explicit interpretation and evaluation protocol for one RAW stream."""

    name: str
    sensor: str
    path: Path
    width: int
    height: int
    frames_used: int
    bit_shift: int
    cfa: str
    nominal_black_dn: float
    reference_mode: str
    clip_starts: tuple[int, ...]
    fixed_isp_wb: tuple[float, float, float, float]
    fixed_isp_exposure: float
    panel_crops: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()

    @classmethod
    def from_mapping(cls, name: str, payload: dict[str, Any]) -> NativeSequenceSpec:
        crops = tuple(
            (str(crop_name), tuple(int(value) for value in coordinates))
            for crop_name, coordinates in payload.get("panel_crops", {}).items()
        )
        spec = cls(
            name=name,
            sensor=str(payload["sensor"]),
            path=Path(payload["path"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            frames_used=int(payload["frames_used"]),
            bit_shift=int(payload["bit_shift"]),
            cfa=str(payload["cfa"]),
            nominal_black_dn=float(payload["nominal_black_dn"]),
            reference_mode=str(payload["reference_mode"]),
            clip_starts=tuple(int(value) for value in payload.get("clip_starts", ())),
            fixed_isp_wb=tuple(float(value) for value in payload["fixed_isp_wb"]),
            fixed_isp_exposure=float(payload["fixed_isp_exposure"]),
            panel_crops=crops,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError(f"{self.name}: width and height must be positive and even")
        if self.frames_used <= 0:
            raise ValueError(f"{self.name}: frames_used must be positive")
        if self.bit_shift not in (0, 4):
            raise ValueError(f"{self.name}: bit_shift must be 0 or 4")
        if self.cfa not in ("RGGB", "GRBG"):
            raise ValueError(f"{self.name}: unsupported CFA {self.cfa!r}")
        if self.reference_mode not in ("motion_aware_split_half", "stress_only"):
            raise ValueError(f"{self.name}: unsupported reference_mode {self.reference_mode!r}")
        if len(self.fixed_isp_wb) != 4 or any(value <= 0.0 for value in self.fixed_isp_wb):
            raise ValueError(f"{self.name}: fixed_isp_wb must hold four positive values")
        if self.fixed_isp_exposure <= 0.0:
            raise ValueError(f"{self.name}: fixed_isp_exposure must be positive")
        for start in self.clip_starts:
            if start < 0:
                raise ValueError(f"{self.name}: clip starts must be non-negative")
            if self.reference_mode == "motion_aware_split_half":
                if start % 2:
                    raise ValueError(f"{self.name}: split-half clip starts must be even")
                clip_end = start + TEMPORAL_FRAMES * 2
            else:
                clip_end = start + TEMPORAL_FRAMES
            if clip_end > self.frames_used:
                raise ValueError(f"{self.name}: clip starting at {start} exceeds frames_used")
        for crop_name, (x0, y0, x1, y1) in self.panel_crops:
            if not crop_name or not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
                raise ValueError(f"{self.name}: invalid panel crop {crop_name!r}")

    @property
    def adapter(self) -> SourceAdapter:
        return SourceAdapter(
            group="cross_sensor",
            scene=self.name,
            role="auxiliary",
            bit_shift=self.bit_shift,
            cfa=self.cfa,
            sensor_black_dn=self.nominal_black_dn,
            reader_kind="single_file",
        )

    @property
    def packed_shape(self) -> tuple[int, int]:
        return self.height // 2, self.width // 2


@dataclass(slots=True)
class NativeReference:
    """Motion-aware full and split-half temporal references in native DN."""

    mean: np.ndarray
    even_mean: np.ndarray
    odd_mean: np.ndarray
    confidence: np.ndarray
    even_confidence: np.ndarray
    odd_confidence: np.ndarray
    valid_region: np.ndarray
    black_levels: dict[str, float]
    frame_means: np.ndarray
    motion_fraction_per_frame: np.ndarray
    motion_threshold_per_frame: np.ndarray
    saturation_fraction: float


@dataclass(slots=True)
class NativeStressProfile:
    """Native statistics for diagnostic-only, no-reference sequences."""

    temporal_mean: np.ndarray
    black_levels: dict[str, float]
    frame_means: np.ndarray
    saturation_fraction: float


def open_sequence(spec: NativeSequenceSpec) -> RawStreamReader:
    """Open a stream using explicit dimensions instead of filename parsing."""

    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    reader = RawStreamReader(spec.path, spec.width, spec.height)
    if reader.frames < spec.frames_used:
        raise ValueError(
            f"{spec.name}: stream has {reader.frames} frames, needs {spec.frames_used}"
        )
    return reader


def channel_vector(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[name] for name in CHANNEL_NAMES], dtype=np.float32)


def normalise_native_dn(values: np.ndarray, black: np.ndarray) -> np.ndarray:
    """Normalize packed native DN with a per-channel black vector."""

    black_array = np.asarray(black, dtype=np.float32)
    if black_array.shape != (4,):
        raise ValueError(f"black must have shape (4,), got {black_array.shape}")
    reshape = (1, 4, 1, 1) if values.ndim == 4 else (4, 1, 1)
    black_view = black_array.reshape(reshape)
    return np.clip(
        (values.astype(np.float32) - black_view) / (WHITE_LEVEL_DN - black_view),
        0.0,
        1.0,
    )


def denormalise_native(values: np.ndarray, black: np.ndarray) -> np.ndarray:
    black_array = np.asarray(black, dtype=np.float32)
    reshape = (1, 4, 1, 1) if values.ndim == 4 else (4, 1, 1)
    black_view = black_array.reshape(reshape)
    return values.astype(np.float32) * (WHITE_LEVEL_DN - black_view) + black_view


def split_clip_indices(spec: NativeSequenceSpec, parity: int) -> list[list[int]]:
    """Return disjoint eight-frame clips for one original-frame parity."""

    if parity not in (0, 1):
        raise ValueError(f"parity must be 0 or 1, got {parity}")
    return [
        [start + parity + offset * 2 for offset in range(TEMPORAL_FRAMES)]
        for start in spec.clip_starts
    ]


def contiguous_clip_indices(spec: NativeSequenceSpec) -> list[list[int]]:
    """Return ordinary eight-frame clips for a no-reference stress sequence."""

    return [
        list(range(start, start + TEMPORAL_FRAMES))
        for start in spec.clip_starts
    ]


def read_normalized_clip(
    reader: RawStreamReader,
    adapter: SourceAdapter,
    indices: list[int],
    black: np.ndarray,
) -> np.ndarray:
    packed = np.stack(
        [pack_frame(reader.read_frame(index), adapter) for index in indices],
        axis=0,
    ).astype(np.float32)
    return normalise_native_dn(packed, black)


def _motion_threshold(
    frame_gray: np.ndarray,
    reference_gray: np.ndarray,
    signal: np.ndarray,
    k_mad: float,
) -> tuple[np.ndarray, float]:
    deviation = np.abs(frame_gray - reference_gray)
    values = deviation[signal]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + k_mad * _MAD_TO_SIGMA * mad
    return deviation > threshold, threshold


def build_motion_aware_reference(
    spec: NativeSequenceSpec,
    *,
    chunk_frames: int = 8,
    confidence_threshold: float = 0.9,
    k_mad: float = 3.0,
) -> tuple[NativeReference, dict[str, Any]]:
    """Build a two-pass native reference without retaining full-frame masks."""

    if spec.reference_mode != "motion_aware_split_half":
        raise ValueError(f"{spec.name}: no quantitative reference for {spec.reference_mode}")
    reader = open_sequence(spec)
    basic = compute_basic_stats(
        reader,
        spec.adapter,
        chunk_frames=chunk_frames,
        max_frames=spec.frames_used,
    )
    channels, height, width = basic.temporal_mean.shape
    preliminary_gray = basic.temporal_mean.mean(axis=0)
    black_mean = float(np.mean(list(basic.black_levels.values())))
    signal = preliminary_gray > black_mean + SIGNAL_MARGIN_DN
    if int(signal.sum()) < MIN_SIGNAL_PIXELS:
        signal = np.ones_like(signal)

    total = np.zeros((channels, height, width), dtype=np.float64)
    even = np.zeros_like(total)
    odd = np.zeros_like(total)
    counts = np.zeros((height, width), dtype=np.float64)
    even_counts = np.zeros_like(counts)
    odd_counts = np.zeros_like(counts)
    thresholds = np.zeros(spec.frames_used, dtype=np.float64)
    motion_fractions = np.zeros(spec.frames_used, dtype=np.float64)

    consumed = 0
    for raw_chunk in reader.iter_chunks(chunk_frames):
        take = min(raw_chunk.shape[0], spec.frames_used - consumed)
        if take <= 0:
            break
        packed = pack_frames(raw_chunk[:take], spec.adapter).astype(np.float32)
        static = np.empty((take, height, width), dtype=bool)
        for local_index in range(take):
            motion, threshold = _motion_threshold(
                packed[local_index].mean(axis=0), preliminary_gray, signal, k_mad
            )
            static[local_index] = ~motion
            global_index = consumed + local_index
            thresholds[global_index] = threshold
            motion_fractions[global_index] = float(motion.mean())

        total += np.sum(packed * static[:, None], axis=0, dtype=np.float64)
        counts += static.sum(axis=0)
        parity_even = (np.arange(consumed, consumed + take) % 2) == 0
        if parity_even.any():
            even += np.sum(
                packed[parity_even] * static[parity_even, None],
                axis=0,
                dtype=np.float64,
            )
            even_counts += static[parity_even].sum(axis=0)
        if (~parity_even).any():
            odd += np.sum(
                packed[~parity_even] * static[~parity_even, None],
                axis=0,
                dtype=np.float64,
            )
            odd_counts += static[~parity_even].sum(axis=0)
        consumed += take

    even_total = (spec.frames_used + 1) // 2
    odd_total = spec.frames_used // 2
    confidence = (counts / spec.frames_used).astype(np.float32)
    even_confidence = (even_counts / even_total).astype(np.float32)
    odd_confidence = (odd_counts / odd_total).astype(np.float32)
    valid = (
        (confidence >= confidence_threshold)
        & (even_confidence >= confidence_threshold)
        & (odd_confidence >= confidence_threshold)
        & (even_counts > 0)
        & (odd_counts > 0)
    )
    reference = NativeReference(
        mean=(total / np.maximum(counts, 1.0)).astype(np.float32),
        even_mean=(even / np.maximum(even_counts, 1.0)).astype(np.float32),
        odd_mean=(odd / np.maximum(odd_counts, 1.0)).astype(np.float32),
        confidence=confidence,
        even_confidence=even_confidence,
        odd_confidence=odd_confidence,
        valid_region=valid,
        black_levels=basic.black_levels,
        frame_means=basic.frame_means,
        motion_fraction_per_frame=motion_fractions,
        motion_threshold_per_frame=thresholds,
        saturation_fraction=basic.saturation_fraction,
    )

    split_half: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        residual = reference.even_mean[channel][valid] - reference.odd_mean[channel][valid]
        split_half[name] = {
            "rmse_dn": float(np.sqrt(np.mean(residual * residual))),
            "median_abs_dn": float(np.median(np.abs(residual))),
        }
    report: dict[str, Any] = {
        "frames_used": spec.frames_used,
        "packed_shape": [channels, height, width],
        "black_levels": basic.black_levels,
        "saturation_fraction": basic.saturation_fraction,
        "frame_drift": frame_drift_stats(basic.frame_means),
        "motion_fraction_mean": float(motion_fractions.mean()),
        "motion_fraction_max": float(motion_fractions.max()),
        "motion_threshold_dn_min": float(thresholds.min()),
        "motion_threshold_dn_max": float(thresholds.max()),
        "valid_fraction": float(valid.mean()),
        "confidence_threshold": confidence_threshold,
        "split_half_reference_difference": split_half,
    }
    return reference, report


def build_stress_profile(
    spec: NativeSequenceSpec,
    *,
    chunk_frames: int = 8,
) -> tuple[NativeStressProfile, dict[str, Any]]:
    """Measure a dynamic sequence without treating its temporal mean as truth."""

    if spec.reference_mode != "stress_only":
        raise ValueError(f"{spec.name}: expected stress_only reference mode")
    reader = open_sequence(spec)
    basic = compute_basic_stats(
        reader,
        spec.adapter,
        chunk_frames=chunk_frames,
        max_frames=spec.frames_used,
    )
    profile = NativeStressProfile(
        temporal_mean=basic.temporal_mean,
        black_levels=basic.black_levels,
        frame_means=basic.frame_means,
        saturation_fraction=basic.saturation_fraction,
    )
    report: dict[str, Any] = {
        "frames_used": spec.frames_used,
        "packed_shape": list(basic.temporal_mean.shape),
        "black_levels": basic.black_levels,
        "saturation_fraction": basic.saturation_fraction,
        "frame_drift": frame_drift_stats(basic.frame_means),
        "temporal_mean_semantics": "display/statistics only; never a clean target",
    }
    return profile, report


def _open_resumable_prediction(
    output_path: Path,
    shape: tuple[int, ...],
    resume_key: str,
) -> tuple[np.memmap, Path, int]:
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    if output_path.is_file() and progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            candidate = np.load(output_path, mmap_mode="r+")
            if (
                progress.get("resume_key") == resume_key
                and progress.get("shape") == list(shape)
                and candidate.shape == shape
                and candidate.dtype == np.float16
            ):
                completed = int(progress.get("completed_clips", 0))
                if 0 <= completed <= shape[0] // TEMPORAL_FRAMES:
                    return candidate, progress_path, completed
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )
    progress_path.write_text(
        json.dumps(
            {"resume_key": resume_key, "shape": list(shape), "completed_clips": 0},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, progress_path, 0


def _write_prediction_progress(
    progress_path: Path,
    resume_key: str,
    shape: tuple[int, ...],
    completed_clips: int,
) -> None:
    progress_path.write_text(
        json.dumps(
            {
                "resume_key": resume_key,
                "shape": list(shape),
                "completed_clips": completed_clips,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def infer_disjoint_clips(
    model: torch.nn.Module,
    spec: NativeSequenceSpec,
    reference: NativeReference,
    parity: int,
    device: torch.device,
    output_path: Path,
    *,
    tile_batch: int,
    resume_key: str,
) -> dict[str, Any]:
    """Infer independent eight-frame clips and write one float16 NPY array."""

    clips = split_clip_indices(spec, parity)
    height, width = spec.packed_shape
    output_shape = (
        len(clips) * TEMPORAL_FRAMES,
        4,
        height,
        width,
    )
    output, progress_path, completed_clips = _open_resumable_prediction(
        output_path,
        output_shape,
        resume_key,
    )
    reader = open_sequence(spec)
    black = channel_vector(reference.black_levels)
    for clip_index, indices in enumerate(clips[completed_clips:], start=completed_clips):
        video = read_normalized_clip(reader, spec.adapter, indices, black)
        prediction = infer_video(model, video, device, tile_batch=tile_batch)
        start = clip_index * TEMPORAL_FRAMES
        output[start : start + TEMPORAL_FRAMES] = prediction.astype(np.float16)
        output.flush()
        _write_prediction_progress(
            progress_path,
            resume_key,
            output_shape,
            clip_index + 1,
        )
        print(
            f"{spec.name}: parity={parity} clip {clip_index + 1}/{len(clips)} complete",
            flush=True,
        )
    return {
        "parity": parity,
        "clips": clips,
        "prediction_shape": list(output.shape),
        "prediction_dtype": str(output.dtype),
    }


def infer_stress_clips(
    model: torch.nn.Module,
    spec: NativeSequenceSpec,
    profile: NativeStressProfile,
    device: torch.device,
    output_path: Path,
    *,
    tile_batch: int,
    resume_key: str,
) -> dict[str, Any]:
    """Infer contiguous clips for a diagnostic-only stream."""

    clips = contiguous_clip_indices(spec)
    height, width = spec.packed_shape
    output_shape = (
        len(clips) * TEMPORAL_FRAMES,
        4,
        height,
        width,
    )
    output, progress_path, completed_clips = _open_resumable_prediction(
        output_path,
        output_shape,
        resume_key,
    )
    reader = open_sequence(spec)
    black = channel_vector(profile.black_levels)
    for clip_index, indices in enumerate(clips[completed_clips:], start=completed_clips):
        video = read_normalized_clip(reader, spec.adapter, indices, black)
        prediction = infer_video(model, video, device, tile_batch=tile_batch)
        start = clip_index * TEMPORAL_FRAMES
        output[start : start + TEMPORAL_FRAMES] = prediction.astype(np.float16)
        output.flush()
        _write_prediction_progress(
            progress_path,
            resume_key,
            output_shape,
            clip_index + 1,
        )
        print(
            f"{spec.name}: stress clip {clip_index + 1}/{len(clips)} complete",
            flush=True,
        )
    return {
        "clips": clips,
        "prediction_shape": list(output.shape),
        "prediction_dtype": str(output.dtype),
    }


def _reference_region_masks(
    reference_dn: np.ndarray,
    valid: np.ndarray,
    black: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = reference_dn.mean(axis=0)
    signal = valid & (gray > float(black.mean()) + SIGNAL_MARGIN_DN)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    population = gradient[signal]
    if population.size < 1000:
        signal = valid.copy()
        population = gradient[signal]
    detail_threshold = float(np.percentile(population, 75.0))
    flat_threshold = float(np.percentile(population, 25.0))
    return signal, signal & (gradient >= detail_threshold), signal & (gradient <= flat_threshold)


def evaluate_disjoint_prediction(
    spec: NativeSequenceSpec,
    reference: NativeReference,
    parity: int,
    prediction_path: Path,
) -> dict[str, Any]:
    """Evaluate one input parity against the opposite-parity temporal mean."""

    clips = split_clip_indices(spec, parity)
    prediction = np.load(prediction_path, mmap_mode="r")
    expected_shape = (len(clips) * TEMPORAL_FRAMES, 4, *spec.packed_shape)
    if prediction.shape != expected_shape:
        raise ValueError(f"{prediction_path}: expected {expected_shape}, got {prediction.shape}")
    target = reference.odd_mean if parity == 0 else reference.even_mean
    target_name = "odd_reference" if parity == 0 else "even_reference"
    black = channel_vector(reference.black_levels)
    signal, detail, flat = _reference_region_masks(target, reference.valid_region, black)
    target_hp = np.stack(
        [target[channel] - cv2.blur(target[channel], (3, 3)) for channel in range(4)]
    )
    reference_gray = reference.mean.mean(axis=0)
    motion_signal = reference_gray > float(black.mean()) + SIGNAL_MARGIN_DN
    if int(motion_signal.sum()) < MIN_SIGNAL_PIXELS:
        motion_signal = np.ones_like(motion_signal)

    reader = open_sequence(spec)
    input_sq = prediction_sq = 0.0
    metric_count = 0
    bias_sum = 0.0
    temporal_input_abs = temporal_prediction_abs = 0.0
    temporal_count = 0
    detail_input_sq = detail_prediction_sq = detail_reference_sq = 0.0
    detail_count = 0
    flat_error_sum = flat_error_sq = 0.0
    flat_count = 0
    legal_count = clipped_count = total_count = 0
    usable_spatial = motion_spatial = 0
    motion_thresholds: list[float] = []

    for clip_index, indices in enumerate(clips):
        input_normalized = read_normalized_clip(reader, spec.adapter, indices, black)
        start = clip_index * TEMPORAL_FRAMES
        pred_normalized = np.asarray(
            prediction[start : start + TEMPORAL_FRAMES], dtype=np.float32
        )
        input_dn = denormalise_native(input_normalized, black)
        pred_dn = denormalise_native(pred_normalized, black)
        frame_masks: list[np.ndarray] = []
        for frame_index in range(TEMPORAL_FRAMES):
            motion, threshold = _motion_threshold(
                input_dn[frame_index].mean(axis=0),
                reference_gray,
                motion_signal,
                3.0,
            )
            motion_thresholds.append(threshold)
            unsaturated = np.all(input_dn[frame_index] < SATURATION_LIMIT_DN, axis=0)
            usable = reference.valid_region & ~motion & unsaturated
            frame_masks.append(usable)
            usable_spatial += int(usable.sum())
            motion_spatial += int(motion.sum())
            mask4 = np.broadcast_to(usable[None], (4, *usable.shape))
            input_error = input_dn[frame_index] - target
            pred_error = pred_dn[frame_index] - target
            input_sq += float(np.sum(np.square(input_error[mask4]), dtype=np.float64))
            prediction_sq += float(np.sum(np.square(pred_error[mask4]), dtype=np.float64))
            bias_sum += float(np.sum(pred_error[mask4], dtype=np.float64))
            metric_count += int(mask4.sum())

            detail_mask = usable & detail
            for channel in range(4):
                if detail_mask.any():
                    input_hp = input_dn[frame_index, channel] - cv2.blur(
                        input_dn[frame_index, channel], (3, 3)
                    )
                    pred_hp = pred_dn[frame_index, channel] - cv2.blur(
                        pred_dn[frame_index, channel], (3, 3)
                    )
                    detail_input_sq += float(
                        np.sum(np.square(input_hp[detail_mask]), dtype=np.float64)
                    )
                    detail_prediction_sq += float(
                        np.sum(np.square(pred_hp[detail_mask]), dtype=np.float64)
                    )
                    detail_reference_sq += float(
                        np.sum(np.square(target_hp[channel][detail_mask]), dtype=np.float64)
                    )
                    detail_count += int(detail_mask.sum())

                flat_mask = usable & flat
                if flat_mask.any():
                    values = pred_error[channel][flat_mask]
                    flat_error_sum += float(np.sum(values, dtype=np.float64))
                    flat_error_sq += float(np.sum(np.square(values), dtype=np.float64))
                    flat_count += int(values.size)

        for frame_index in range(1, TEMPORAL_FRAMES):
            temporal_mask = frame_masks[frame_index - 1] & frame_masks[frame_index]
            if not temporal_mask.any():
                continue
            temporal_mask4 = np.broadcast_to(
                temporal_mask[None], (4, *temporal_mask.shape)
            )
            temporal_input_abs += float(
                np.sum(
                    np.abs(input_dn[frame_index] - input_dn[frame_index - 1])[
                        temporal_mask4
                    ],
                    dtype=np.float64,
                )
            )
            temporal_prediction_abs += float(
                np.sum(
                    np.abs(pred_dn[frame_index] - pred_dn[frame_index - 1])[
                        temporal_mask4
                    ],
                    dtype=np.float64,
                )
            )
            temporal_count += int(temporal_mask4.sum())

        legal_count += int(((pred_dn >= 0.0) & (pred_dn <= WHITE_LEVEL_DN)).sum())
        clipped_count += int(((pred_normalized <= 0.0) | (pred_normalized >= 1.0)).sum())
        total_count += int(pred_dn.size)

    if metric_count == 0:
        raise ValueError(f"{spec.name}: no valid static pixels for parity {parity}")
    input_rmse = float(np.sqrt(input_sq / metric_count))
    prediction_rmse = float(np.sqrt(prediction_sq / metric_count))
    input_psnr = float(20.0 * np.log10(WHITE_LEVEL_DN / max(input_rmse, 1e-8)))
    prediction_psnr = float(
        20.0 * np.log10(WHITE_LEVEL_DN / max(prediction_rmse, 1e-8))
    )
    reference_rms = np.sqrt(detail_reference_sq / max(detail_count, 1))
    flat_mean = flat_error_sum / max(flat_count, 1)
    flat_variance = flat_error_sq / max(flat_count, 1) - flat_mean * flat_mean
    evaluated_frames = len(clips) * TEMPORAL_FRAMES
    spatial_total = evaluated_frames * spec.packed_shape[0] * spec.packed_shape[1]
    return {
        "input_parity": "even" if parity == 0 else "odd",
        "target": target_name,
        "clips": clips,
        "evaluated_frames": evaluated_frames,
        "input": {"rmse_dn": input_rmse, "psnr_db": input_psnr},
        "a3": {"rmse_dn": prediction_rmse, "psnr_db": prediction_psnr},
        "a3_psnr_gain_db": prediction_psnr - input_psnr,
        "static_values": metric_count,
        "usable_spatial_fraction": usable_spatial / spatial_total,
        "detected_motion_fraction": motion_spatial / spatial_total,
        "motion_threshold_dn_min": min(motion_thresholds),
        "motion_threshold_dn_max": max(motion_thresholds),
        "input_temporal_mad_dn": temporal_input_abs / max(temporal_count, 1),
        "a3_temporal_mad_dn": temporal_prediction_abs / max(temporal_count, 1),
        "a3_output_bias_dn": bias_sum / metric_count,
        "legal_raw_fraction": legal_count / total_count,
        "normalized_clip_fraction": clipped_count / total_count,
        "input_high_frequency_ratio": (
            np.sqrt(detail_input_sq / max(detail_count, 1)) / max(reference_rms, 1e-8)
        ),
        "a3_high_frequency_ratio": (
            np.sqrt(detail_prediction_sq / max(detail_count, 1))
            / max(reference_rms, 1e-8)
        ),
        "a3_flat_residual_sigma_dn": float(np.sqrt(max(flat_variance, 0.0))),
        "detail_values": detail_count,
        "flat_values": flat_count,
    }


def evaluate_stress_prediction(
    spec: NativeSequenceSpec,
    profile: NativeStressProfile,
    prediction_path: Path,
) -> dict[str, Any]:
    """Report no-reference temporal/detail diagnostics without a pseudo PSNR."""

    clips = contiguous_clip_indices(spec)
    prediction = np.load(prediction_path, mmap_mode="r")
    expected_shape = (len(clips) * TEMPORAL_FRAMES, 4, *spec.packed_shape)
    if prediction.shape != expected_shape:
        raise ValueError(f"{prediction_path}: expected {expected_shape}, got {prediction.shape}")
    black = channel_vector(profile.black_levels)
    reader = open_sequence(spec)

    output_bias_sum = change_abs_sum = 0.0
    input_temporal_sum = prediction_temporal_sum = 0.0
    motion_input_sum = motion_prediction_sum = 0.0
    static_input_sum = static_prediction_sum = 0.0
    temporal_count = motion_count = static_count = 0
    input_hf_sq = prediction_hf_sq = 0.0
    detail_count = 0
    legal_count = clipped_count = total_count = 0

    for clip_index, indices in enumerate(clips):
        input_normalized = read_normalized_clip(reader, spec.adapter, indices, black)
        start = clip_index * TEMPORAL_FRAMES
        pred_normalized = np.asarray(
            prediction[start : start + TEMPORAL_FRAMES], dtype=np.float32
        )
        input_dn = denormalise_native(input_normalized, black)
        pred_dn = denormalise_native(pred_normalized, black)
        residual = pred_dn - input_dn
        output_bias_sum += float(np.sum(residual, dtype=np.float64))
        change_abs_sum += float(np.sum(np.abs(residual), dtype=np.float64))

        for frame_index in range(TEMPORAL_FRAMES):
            gray = input_dn[frame_index].mean(axis=0)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient = cv2.magnitude(grad_x, grad_y)
            detail = gradient >= float(np.percentile(gradient, 75.0))
            for channel in range(4):
                input_hp = input_dn[frame_index, channel] - cv2.blur(
                    input_dn[frame_index, channel], (3, 3)
                )
                prediction_hp = pred_dn[frame_index, channel] - cv2.blur(
                    pred_dn[frame_index, channel], (3, 3)
                )
                input_hf_sq += float(
                    np.sum(np.square(input_hp[detail]), dtype=np.float64)
                )
                prediction_hf_sq += float(
                    np.sum(np.square(prediction_hp[detail]), dtype=np.float64)
                )
                detail_count += int(detail.sum())

        for frame_index in range(1, TEMPORAL_FRAMES):
            input_difference = np.abs(
                input_dn[frame_index] - input_dn[frame_index - 1]
            )
            prediction_difference = np.abs(
                pred_dn[frame_index] - pred_dn[frame_index - 1]
            )
            gray_difference = input_difference.mean(axis=0)
            motion_threshold = float(np.percentile(gray_difference, 90.0))
            static_threshold = float(np.percentile(gray_difference, 50.0))
            motion = gray_difference >= motion_threshold
            static = gray_difference <= static_threshold
            input_temporal_sum += float(np.sum(input_difference, dtype=np.float64))
            prediction_temporal_sum += float(
                np.sum(prediction_difference, dtype=np.float64)
            )
            temporal_count += int(input_difference.size)
            motion_mask = np.broadcast_to(motion[None], (4, *motion.shape))
            static_mask = np.broadcast_to(static[None], (4, *static.shape))
            motion_input_sum += float(
                np.sum(input_difference[motion_mask], dtype=np.float64)
            )
            motion_prediction_sum += float(
                np.sum(prediction_difference[motion_mask], dtype=np.float64)
            )
            static_input_sum += float(
                np.sum(input_difference[static_mask], dtype=np.float64)
            )
            static_prediction_sum += float(
                np.sum(prediction_difference[static_mask], dtype=np.float64)
            )
            motion_count += int(motion_mask.sum())
            static_count += int(static_mask.sum())

        legal_count += int(((pred_dn >= 0.0) & (pred_dn <= WHITE_LEVEL_DN)).sum())
        clipped_count += int(((pred_normalized <= 0.0) | (pred_normalized >= 1.0)).sum())
        total_count += int(pred_dn.size)

    input_hf_rms = np.sqrt(input_hf_sq / max(detail_count, 1))
    prediction_hf_rms = np.sqrt(prediction_hf_sq / max(detail_count, 1))
    high_frequency_ratio = (
        float(prediction_hf_rms / input_hf_rms)
        if input_hf_rms > 1e-6
        else None
    )
    return {
        "protocol": "diagnostic-only no-reference stress evaluation",
        "clean_target_available": False,
        "clips": clips,
        "evaluated_frames": len(clips) * TEMPORAL_FRAMES,
        "mean_output_bias_dn": output_bias_sum / total_count,
        "mean_abs_output_change_dn": change_abs_sum / total_count,
        "input_temporal_mad_dn": input_temporal_sum / max(temporal_count, 1),
        "a3_temporal_mad_dn": prediction_temporal_sum / max(temporal_count, 1),
        "temporal_mad_ratio": prediction_temporal_sum / max(input_temporal_sum, 1e-8),
        "motion_temporal_ratio": motion_prediction_sum / max(motion_input_sum, 1e-8),
        "static_temporal_ratio": static_prediction_sum / max(static_input_sum, 1e-8),
        "spatial_high_frequency_ratio": high_frequency_ratio,
        "legal_raw_fraction": legal_count / total_count,
        "normalized_clip_fraction": clipped_count / total_count,
        "diagnostic_populations": {
            "temporal_values": temporal_count,
            "motion_values": motion_count,
            "static_values": static_count,
            "detail_values": detail_count,
        },
        "interpretation": (
            "Ratios are diagnostics only; none is a clean-target quality score or pass gate."
        ),
    }


def render_fixed_isp(
    frame_dn: np.ndarray,
    black: np.ndarray,
    wb: tuple[float, float, float, float],
    exposure: float,
) -> np.ndarray:
    """Render canonical packed RGGB data with one deterministic fixed ISP."""

    if frame_dn.ndim != 3 or frame_dn.shape[0] != 4:
        raise ValueError(f"expected [4,H,W], got {frame_dn.shape}")
    black_view = np.asarray(black, dtype=np.float32)[:, None, None]
    white_balance = np.asarray(wb, dtype=np.float32)[:, None, None]
    linear = np.clip(
        (frame_dn.astype(np.float32) - black_view) / (WHITE_LEVEL_DN - black_view),
        0.0,
        1.0,
    )
    linear = np.clip(linear * white_balance * exposure, 0.0, 1.0)
    height, width = frame_dn.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = linear[0]
    mosaic[0::2, 1::2] = linear[1]
    mosaic[1::2, 0::2] = linear[2]
    mosaic[1::2, 1::2] = linear[3]
    raw16 = np.rint(mosaic * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535.0
    threshold = 0.0031308
    srgb = np.where(
        rgb <= threshold,
        rgb * 12.92,
        1.055 * np.power(np.maximum(rgb, threshold), 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def make_fixed_isp_panel(
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    black: np.ndarray,
    wb: tuple[float, float, float, float],
    exposure: float,
    *,
    crop: tuple[int, int, int, int] | None = None,
    cell_width: int = 560,
    column_labels: tuple[str, str, str] = (
        "Input",
        "A3 EMA",
        "Opposite reference",
    ),
) -> np.ndarray:
    """Compose input/A3/reference rows using the exact same ISP settings."""

    rendered_rows: list[np.ndarray] = []
    for row_label, input_dn, prediction_dn, reference_dn in rows:
        images = [
            render_fixed_isp(frame, black, wb, exposure)
            for frame in (input_dn, prediction_dn, reference_dn)
        ]
        if crop is not None:
            x0, y0, x1, y1 = crop
            images = [image[y0:y1, x0:x1] for image in images]
        cells: list[np.ndarray] = []
        for image, column in zip(images, column_labels, strict=True):
            scale = cell_width / image.shape[1]
            resized = cv2.resize(
                image,
                (cell_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            canvas = cv2.copyMakeBorder(
                resized, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)
            )
            cv2.putText(
                canvas,
                f"{row_label} | {column}",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            cells.append(canvas)
        rendered_rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(rendered_rows, axis=0)

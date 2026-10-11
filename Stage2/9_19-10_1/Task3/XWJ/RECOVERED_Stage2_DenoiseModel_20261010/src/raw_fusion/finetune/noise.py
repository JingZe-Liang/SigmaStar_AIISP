"""A2 calibration noise model, scene PTC diagnostics, and synthetic pairs."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from raw_fusion.finetune.adapters import calibration_adapter, resolve_adapter
from raw_fusion.finetune.reference import _channel_vector
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, open_stream, pack_frame, pack_frames

_CODE_VERSION = "finetune_a2_v1"
_CALIBRATION_GAINS = ("100", "200", "400", "800", "1600", "3200", "6400", "12800", "25600")
_PTC_EDGES_DN = np.asarray((0.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 3800.0), dtype=np.float64)
_PAIR_SAMPLE_STRIDE = 8
_SCENE_PTC_SAMPLE_STRIDE = 8
_SCENE_PTC_FRAME_STRIDE = 4
_AUTHENTICITY_LIMIT = 0.15
_WHITE_CLIP_DN = 4095.0


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """One paired black/flat calibration operating point."""

    gain_label: str
    analog_gain: int
    gain_factor: float
    signal_dn: dict[str, float]
    flat_sigma_dn: dict[str, float]
    read_sigma_dn: dict[str, float]
    flat_variance_dn2: dict[str, float]
    read_variance_dn2: dict[str, float]
    shot_variance_dn2: dict[str, float]


@dataclass(frozen=True, slots=True)
class ScenePtcPoint:
    """One brightness-bin point from a real scene stream."""

    stream: str
    channel: str
    signal_low_dn: float
    signal_high_dn: float
    pixels: int
    sigma_dn: float
    variance_dn2: float


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, array)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _manifest_path(supervision_dir: Path, gain_label: str, variant: str) -> Path:
    return supervision_dir / "stream_manifests" / f"04_mis20s1_calibrationdata_{gain_label}_{variant}.json"


def _read_mean(manifest: dict, chunk_frames: int) -> np.ndarray:
    adapter = calibration_adapter(manifest["variant"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    total = np.zeros((4, 540, 960), dtype=np.float64)
    consumed = 0
    for raw_chunk in reader.iter_chunks(chunk_frames):
        take = min(raw_chunk.shape[0], int(manifest["frames_used"]) - consumed)
        total += pack_frames(raw_chunk[:take], adapter).astype(np.float64).sum(axis=0)
        consumed += take
        if consumed >= int(manifest["frames_used"]):
            break
    return (total / consumed).astype(np.float32)


def collect_calibration_points(supervision_dir: Path, chunk_frames: int = 10) -> tuple[list[CalibrationPoint], dict[str, np.ndarray]]:
    """Read all nine calibration pairs and derive empirical variance tables."""

    points: list[CalibrationPoint] = []
    flat_means: dict[str, np.ndarray] = {}
    for gain_label in _CALIBRATION_GAINS:
        black_manifest = _load_json(_manifest_path(supervision_dir, gain_label, "black"))
        flat_manifest = _load_json(_manifest_path(supervision_dir, gain_label, "color"))
        black_mean = _read_mean(black_manifest, chunk_frames)
        flat_mean = _read_mean(flat_manifest, chunk_frames)
        flat_means[gain_label] = flat_mean
        signal_dn = {
            name: float(np.median(flat_mean[channel] - black_mean[channel]))
            for channel, name in enumerate(CHANNEL_NAMES)
        }
        flat_sigma = {
            name: float(flat_manifest["sigma_split_half"]["per_channel"][name])
            for name in CHANNEL_NAMES
        }
        read_sigma = {
            name: float(black_manifest["sigma_split_half"]["per_channel"][name])
            for name in CHANNEL_NAMES
        }
        flat_variance = {name: value * value for name, value in flat_sigma.items()}
        read_variance = {name: value * value for name, value in read_sigma.items()}
        shot_variance = {
            name: max(flat_variance[name] - read_variance[name], 0.0)
            for name in CHANNEL_NAMES
        }
        analog_gain = int(flat_manifest["filename_metadata"]["analog_gain"])
        points.append(CalibrationPoint(
            gain_label=gain_label,
            analog_gain=analog_gain,
            gain_factor=analog_gain / 1024.0,
            signal_dn=signal_dn,
            flat_sigma_dn=flat_sigma,
            read_sigma_dn=read_sigma,
            flat_variance_dn2=flat_variance,
            read_variance_dn2=read_variance,
            shot_variance_dn2=shot_variance,
        ))
    return points, flat_means


def _fit_huber_coefficients(points: list[CalibrationPoint], channel: str) -> tuple[float, float, float, list[float]]:
    """Fit total variance = shot * signal * gain + read * gain^2 robustly."""

    features = np.asarray([
        (point.signal_dn[channel] * point.gain_factor, point.gain_factor ** 2)
        for point in points
    ], dtype=np.float64)
    observations = np.asarray([point.flat_variance_dn2[channel] for point in points], dtype=np.float64)
    weights = np.ones(len(points), dtype=np.float64)
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(20):
        weighted_features = features * weights[:, None]
        weighted_observations = observations * weights
        coefficients = np.linalg.lstsq(weighted_features, weighted_observations, rcond=None)[0]
        residual = observations - features @ coefficients
        scale = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
        if scale <= 1e-8:
            break
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-8))
    coefficients = np.maximum(coefficients, 0.0)
    residual = (observations - features @ coefficients).tolist()
    return float(coefficients[0]), float(coefficients[1]), float(np.sqrt(np.mean(np.square(residual)))), residual


def _median_correlation(matrices: list[np.ndarray]) -> np.ndarray:
    matrix = np.median(np.stack(matrices), axis=0)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    matrix = (eigenvectors * np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(matrix), 1e-8))
    matrix = matrix / scale[:, None] / scale[None, :]
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _pair_noise_stats(
    manifest: dict,
    frames: list[np.ndarray] | None = None,
    pair_count: int = 8,
) -> dict:
    adapter = calibration_adapter("color")
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    if frames is None:
        frames = [
            pack_frame(reader.read_frame(index), adapter).astype(np.float32)[:, ::_PAIR_SAMPLE_STRIDE, ::_PAIR_SAMPLE_STRIDE]
            for index in range(min(reader.frames, pair_count * 2))
        ]
    sigmas: list[list[float]] = []
    correlations: list[np.ndarray] = []
    for pair_index in range(min(pair_count, len(frames) // 2)):
        first = frames[2 * pair_index]
        second = frames[2 * pair_index + 1]
        difference = (first - second) / np.sqrt(2.0)
        filtered = difference - np.stack([
            cv2.blur(difference[channel], (3, 3)) for channel in range(4)
        ])
        samples = filtered.reshape(4, -1).astype(np.float64)
        sigmas.append([float(np.std(samples[channel], ddof=1)) for channel in range(4)])
        correlations.append(np.corrcoef(samples))
    sigma = np.median(np.asarray(sigmas), axis=0)
    return {
        "sigma_dn": {name: float(sigma[channel]) for channel, name in enumerate(CHANNEL_NAMES)},
        "correlation": _median_correlation(correlations).tolist(),
        "pairs": len(sigmas),
    }


def build_calibration_model(
    points: list[CalibrationPoint],
    correlations: dict[str, list[list[float]]],
) -> dict:
    shot_coefficients: dict[str, float] = {}
    parametric: dict[str, dict] = {}
    empirical_shot_coefficients: dict[str, dict[str, float]] = {name: {} for name in CHANNEL_NAMES}
    read_variance: dict[str, dict[str, float]] = {name: {} for name in CHANNEL_NAMES}
    for name in CHANNEL_NAMES:
        shot, read, rmse, residual = _fit_huber_coefficients(points, name)
        shot_coefficients[name] = shot
        parametric[name] = {
            "shot_variance_per_signal_gain": shot,
            "read_variance_per_gain_squared": read,
            "rmse_dn2": rmse,
            "residual_dn2": {point.gain_label: value for point, value in zip(points, residual)},
        }
        for point in points:
            empirical_shot_coefficients[name][point.gain_label] = point.shot_variance_dn2[name] / max(point.signal_dn[name], 1e-6)
            read_variance[name][point.gain_label] = point.read_variance_dn2[name]

    return {
        "code_version": _CODE_VERSION,
        "channels": list(CHANNEL_NAMES),
        "gain_labels": [point.gain_label for point in points],
        "gain_factors": {point.gain_label: point.gain_factor for point in points},
        "shot_coefficients": shot_coefficients,
        "parametric_fit": parametric,
        "empirical_shot_coefficients": empirical_shot_coefficients,
        "read_variance_dn2": read_variance,
        "channel_correlation": correlations,
        "quantization": {
            "domain": "12bit_dn",
            "lsb_dn": 1.0,
            "round_before_clip": True,
            "clip_range_dn": [0.0, _WHITE_CLIP_DN],
        },
        "non_monotonic_gain_policy": "retain empirical 1600/3200 calibration points; interpolate in log gain only between exact points",
        "calibration_points": [
            {
                "gain_label": point.gain_label,
                "analog_gain": point.analog_gain,
                "gain_factor": point.gain_factor,
                "signal_dn": point.signal_dn,
                "flat_sigma_dn": point.flat_sigma_dn,
                "read_sigma_dn": point.read_sigma_dn,
                "shot_variance_dn2": point.shot_variance_dn2,
            }
            for point in points
        ],
    }


def _scene_jobs(supervision_dir: Path) -> list[dict]:
    jobs: list[dict] = []
    manifest_dir = supervision_dir / "stream_manifests"
    for path in sorted(manifest_dir.glob("04_shdarkroom_*.json")):
        jobs.append(_load_json(path))
    for path in sorted(manifest_dir.glob("04_mis20s1_outdoor_*.json")):
        jobs.append(_load_json(path))
    return jobs


def measure_scene_ptc(
    manifest: dict,
    frame_stride: int = _SCENE_PTC_FRAME_STRIDE,
) -> list[ScenePtcPoint]:
    """Measure pair-difference PTC points in long-term static scene pixels."""

    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames_total = int(manifest["frames_used"])
    frame_indices = list(range(0, frames_total, frame_stride))
    mask = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")
    stride = _SCENE_PTC_SAMPLE_STRIDE
    sampled_height = len(range(0, 540, stride))
    sampled_width = len(range(0, 960, stride))
    total = np.zeros((4, sampled_height, sampled_width), dtype=np.float64)
    counts = np.zeros((sampled_height, sampled_width), dtype=np.float64)
    pair_squared = np.zeros((4, sampled_height, sampled_width), dtype=np.float64)
    pair_counts = np.zeros((sampled_height, sampled_width), dtype=np.float64)
    previous: np.ndarray | None = None
    previous_valid: np.ndarray | None = None
    for frame_index in frame_indices:
        packed = pack_frame(reader.read_frame(frame_index), adapter).astype(np.float32)
        packed = packed[:, ::stride, ::stride]
        current_valid = np.asarray(mask[frame_index, 0, ::2, ::2][::stride, ::stride] < 0.5)
        total += packed * current_valid[None]
        counts += current_valid
        if previous is not None and previous_valid is not None:
            pair_valid = previous_valid & current_valid
            difference = packed - previous
            pair_squared += difference.astype(np.float64) ** 2 * pair_valid[None]
            pair_counts += pair_valid
        previous = packed
        previous_valid = current_valid

    mean = total / np.maximum(counts, 1.0)
    signal = mean - _channel_vector({str(k): float(v) for k, v in manifest["black_levels"].items()})
    pair_variance = pair_squared / np.maximum(2.0 * pair_counts[None], 1.0)
    points: list[ScenePtcPoint] = []
    for channel, name in enumerate(CHANNEL_NAMES):
        valid = (counts > 0) & (pair_counts > 0)
        for low, high in zip(_PTC_EDGES_DN[:-1], _PTC_EDGES_DN[1:]):
            selected = valid & (signal[channel] >= low) & (signal[channel] < high)
            if int(selected.sum()) < 100:
                continue
            variance = float(np.median(pair_variance[channel][selected]))
            points.append(ScenePtcPoint(
                stream=f"{manifest['scene']}/{manifest['gain_label']}",
                channel=name,
                signal_low_dn=float(low),
                signal_high_dn=float(high),
                pixels=int(selected.sum()),
                sigma_dn=float(np.sqrt(max(variance, 0.0))),
                variance_dn2=variance,
            ))
    return points


def _interpolate_log_gain(table: dict[str, float], gain_factor: float, gain_factors: dict[str, float]) -> float:
    ordered = sorted((float(value), float(table[label])) for label, value in gain_factors.items())
    if gain_factor <= ordered[0][0]:
        return ordered[0][1]
    if gain_factor >= ordered[-1][0]:
        return ordered[-1][1]
    for (low_gain, low_value), (high_gain, high_value) in zip(ordered[:-1], ordered[1:]):
        if low_gain <= gain_factor <= high_gain:
            fraction = (np.log(gain_factor) - np.log(low_gain)) / (np.log(high_gain) - np.log(low_gain))
            return float(low_value + fraction * (high_value - low_value))
    raise ValueError(f"gain factor {gain_factor} is outside interpolation table")


def _channel_correlation(model: dict) -> np.ndarray:
    return np.asarray(model["channel_correlation"]["aggregate"], dtype=np.float64)


def generate_noisy_frame(
    clean: np.ndarray,
    model: dict,
    gain_label: str,
    black_levels: dict[str, float],
    fpn_offset: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one quantized packed noisy frame in 12-bit DN."""

    if clean.ndim != 3 or clean.shape[0] != 4:
        raise ValueError(f"expected packed clean shape (4,H,W), got {clean.shape}")
    gain_factors = model["gain_factors"]
    gain_factor = float(gain_factors[gain_label])
    black = _channel_vector(black_levels).astype(np.float64)
    base = clean.astype(np.float64)
    if fpn_offset is not None:
        if fpn_offset.shape != clean.shape:
            raise ValueError(f"FPN shape {fpn_offset.shape} does not match clean shape {clean.shape}")
        base = base + fpn_offset
    signal = np.maximum(base - black, 0.0)
    variance = np.zeros_like(base)
    for channel, name in enumerate(CHANNEL_NAMES):
        empirical = model["empirical_shot_coefficients"][name]
        shot_coefficient = _interpolate_log_gain(empirical, gain_factor, gain_factors)
        read_variance = _interpolate_log_gain(model["read_variance_dn2"][name], gain_factor, gain_factors)
        variance[channel] = np.maximum(shot_coefficient * signal[channel] + read_variance, 0.0)
    standard_normal = rng.standard_normal(clean.shape)
    correlation = _channel_correlation(model)
    cholesky = np.linalg.cholesky(correlation)
    correlated = np.einsum("ij,jhw->ihw", cholesky, standard_normal, optimize=True)
    noisy = np.rint(base + np.sqrt(variance) * correlated)
    return np.clip(noisy, 0.0, _WHITE_CLIP_DN).astype(np.uint16)


def _synthetic_pair_stats(first: np.ndarray, second: np.ndarray) -> dict:
    difference = (first.astype(np.float32) - second.astype(np.float32)) / np.sqrt(2.0)
    filtered = difference - np.stack([
        cv2.blur(difference[channel], (3, 3)) for channel in range(4)
    ])
    return {
        "sigma_dn": {
            name: float(np.std(filtered[channel], ddof=1))
            for channel, name in enumerate(CHANNEL_NAMES)
        },
        "highpass_p95_dn": {
            name: float(np.percentile(np.abs(filtered[channel]), 95.0))
            for channel, name in enumerate(CHANNEL_NAMES)
        },
    }


def build_synthetic_pairs(
    model: dict,
    clean_reference: np.ndarray,
    fpn_offset: np.ndarray,
    black_levels: dict[str, float],
    output_dir: Path,
    seed: int = 20260921,
) -> dict:
    """Write one deterministic synthetic pair per calibration gain."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_array(output_dir / "clean_reference.npy", clean_reference.astype(np.float32))
    pairs: list[dict] = []
    for index, gain_label in enumerate(model["gain_labels"]):
        noisy = generate_noisy_frame(
            clean_reference,
            model,
            gain_label,
            black_levels,
            fpn_offset,
            np.random.default_rng(seed + index),
        )
        noisy_path = output_dir / f"synthetic_ag{gain_label}_noisy.npy"
        _save_array(noisy_path, noisy)
        pairs.append({
            "gain_label": gain_label,
            "noisy_path": str(noisy_path),
            "clean_path": str(output_dir / "clean_reference.npy"),
            "seed": seed + index,
            "saturation_fraction": float(np.mean(noisy >= _WHITE_CLIP_DN)),
        })
    return {"seed": seed, "pairs": pairs}


def _scene_model_alignment(model: dict, scene_points: list[ScenePtcPoint], manifests: list[dict]) -> dict:
    stream_gain: dict[str, float] = {}
    for manifest in manifests:
        metadata = manifest["filename_metadata"]
        sensor_gain = metadata.get("sensor_gain")
        if sensor_gain is not None:
            stream_gain[f"{manifest['scene']}/{manifest['gain_label']}"] = float(sensor_gain) / 1024.0
    ratios: list[float] = []
    diagnostics: list[dict] = []
    for point in scene_points:
        gain_factor = stream_gain.get(point.stream)
        if gain_factor is None:
            continue
        predicted_variances = []
        for channel_name in [point.channel]:
            shot = _interpolate_log_gain(model["empirical_shot_coefficients"][channel_name], gain_factor, model["gain_factors"])
            read = _interpolate_log_gain(model["read_variance_dn2"][channel_name], gain_factor, model["gain_factors"])
            center = (point.signal_low_dn + point.signal_high_dn) / 2.0
            predicted_variances.append(shot * center + read)
        predicted_sigma = float(np.sqrt(max(predicted_variances[0], 1e-8)))
        ratio = point.sigma_dn / predicted_sigma
        ratios.append(ratio)
        diagnostics.append({
            "stream": point.stream,
            "channel": point.channel,
            "signal_range_dn": [point.signal_low_dn, point.signal_high_dn],
            "observed_sigma_dn": point.sigma_dn,
            "predicted_sigma_dn": predicted_sigma,
            "observed_to_predicted": ratio,
        })
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    return {
        "diagnostics": diagnostics,
        "median_observed_to_calibration_predicted": median_ratio,
        "within_15_percent": bool(ratios) and 0.85 <= median_ratio <= 1.15,
        "interpretation": "scene PTC is a diagnostic; real scene streams remain authoritative for real-pair training",
    }


def build_a2_artifacts(
    dataset_root: Path,
    out_dir: Path,
    chunk_frames: int = 10,
    seed: int = 20260921,
) -> dict:
    """Build the A2 model, scene PTC report, synthetic pairs, and gate report."""

    supervision_dir = out_dir / "supervision"
    noise_dir = out_dir / "noise_model"
    synthetic_dir = noise_dir / "synthetic_pairs"
    points, flat_means = collect_calibration_points(supervision_dir, chunk_frames)

    real_pair: dict[str, dict] = {}
    correlation_matrices: list[np.ndarray] = []
    for point in points:
        manifest = _load_json(_manifest_path(supervision_dir, point.gain_label, "color"))
        stats = _pair_noise_stats(manifest)
        real_pair[point.gain_label] = stats
        correlation_matrices.append(np.asarray(stats["correlation"], dtype=np.float64))
    aggregate_correlation = _median_correlation(correlation_matrices)
    model = build_calibration_model(
        points,
        {"per_gain": {label: real_pair[label]["correlation"] for label in real_pair}, "aggregate": aggregate_correlation.tolist()},
    )

    scene_points: list[ScenePtcPoint] = []
    scene_manifests = _scene_jobs(supervision_dir)
    for index, manifest in enumerate(scene_manifests, start=1):
        print(f"[a2] scene_ptc {index}/{len(scene_manifests)} {manifest['scene']}/{manifest['gain_label']}", flush=True)
        scene_points.extend(measure_scene_ptc(manifest))
    scene_ptc_payload = {
        "code_version": _CODE_VERSION,
        "bin_edges_dn": _PTC_EDGES_DN.tolist(),
        "frame_stride": _SCENE_PTC_FRAME_STRIDE,
        "spatial_stride": _SCENE_PTC_SAMPLE_STRIDE,
        "points": [{
            "stream": point.stream,
            "channel": point.channel,
            "signal_low_dn": point.signal_low_dn,
            "signal_high_dn": point.signal_high_dn,
            "pixels": point.pixels,
            "sigma_dn": point.sigma_dn,
            "variance_dn2": point.variance_dn2,
        } for point in scene_points],
    }
    scene_alignment = _scene_model_alignment(model, scene_points, scene_manifests)

    reference_manifest = _load_json(supervision_dir / "references" / "a1_reference_manifest.json")
    report_manifest = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    clean_reference = np.load(reference_manifest["clean_reference_path"], mmap_mode="r")
    fpn_offset = np.load(reference_manifest["fpn_paths"]["offset_dn"], mmap_mode="r")
    target_black = report_manifest["black_levels"]
    synthetic = build_synthetic_pairs(
        model,
        np.asarray(clean_reference),
        np.asarray(fpn_offset),
        target_black,
        synthetic_dir,
        seed,
    )

    synthetic_pair: dict[str, dict] = {}
    for index, point in enumerate(points):
        flat_clean = flat_means[point.gain_label]
        first = generate_noisy_frame(flat_clean, model, point.gain_label, {
            name: float(_load_json(_manifest_path(supervision_dir, point.gain_label, "black"))["black_levels"][name])
            for name in CHANNEL_NAMES
        }, None, np.random.default_rng(seed + index * 2))
        second = generate_noisy_frame(flat_clean, model, point.gain_label, {
            name: float(_load_json(_manifest_path(supervision_dir, point.gain_label, "black"))["black_levels"][name])
            for name in CHANNEL_NAMES
        }, None, np.random.default_rng(seed + index * 2 + 1))
        synthetic_pair[point.gain_label] = _synthetic_pair_stats(first, second)

    gate: dict[str, dict] = {}
    for point in points:
        real = real_pair[point.gain_label]["sigma_dn"]
        generated = synthetic_pair[point.gain_label]["sigma_dn"]
        per_channel = {
            name: {
                "real_sigma_dn": real[name],
                "synthetic_sigma_dn": generated[name],
                "relative_error": abs(generated[name] - real[name]) / max(real[name], 1e-8),
            }
            for name in CHANNEL_NAMES
        }
        gate[point.gain_label] = {
            "per_channel": per_channel,
            "passed": all(value["relative_error"] <= _AUTHENTICITY_LIMIT for value in per_channel.values()),
        }
    authenticity_passed = all(item["passed"] for item in gate.values())

    _save_json(noise_dir / "noise_model.json", model)
    _save_json(noise_dir / "scene_ptc.json", scene_ptc_payload)
    _save_json(noise_dir / "synthetic_pairs_manifest.json", synthetic)
    authenticity_report = {
        "code_version": _CODE_VERSION,
        "gate_limit_relative_error": _AUTHENTICITY_LIMIT,
        "real_calibration_pair_stats": real_pair,
        "synthetic_calibration_pair_stats": synthetic_pair,
        "per_gain_gate": gate,
        "passed": authenticity_passed,
        "training_role": "primary_synthetic" if authenticity_passed else "augmentation_only",
        "fallback_policy": "after two failed validation rounds, use real cross-gain pairs only and retain synthetic pairs as augmentation",
        "validation_rounds": 1,
    }
    _save_json(noise_dir / "authenticity_report.json", authenticity_report)
    a2_report = {
        "code_version": _CODE_VERSION,
        "dataset_root": str(dataset_root),
        "calibration_points": len(points),
        "scene_ptc_streams": len(scene_manifests),
        "scene_ptc_alignment": scene_alignment,
        "authenticity": authenticity_report,
        "artifacts": {
            "noise_model": str(noise_dir / "noise_model.json"),
            "scene_ptc": str(noise_dir / "scene_ptc.json"),
            "synthetic_pairs": str(noise_dir / "synthetic_pairs_manifest.json"),
            "authenticity": str(noise_dir / "authenticity_report.json"),
        },
        "passed": authenticity_passed,
    }
    _save_json(noise_dir / "a2_report.json", a2_report)
    return a2_report

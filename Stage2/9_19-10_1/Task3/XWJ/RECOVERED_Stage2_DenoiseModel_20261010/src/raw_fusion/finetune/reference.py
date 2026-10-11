"""A1 temporal references and fixed-pattern-noise correction.

The main reference is a motion-aware temporal mean of the 04/shdarkroom/10x
stream.  Calibration black and flat captures provide the fixed-pattern-noise
maps; no denoising model output is used anywhere in this module.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from raw_fusion.finetune.adapters import calibration_adapter, resolve_adapter
from raw_fusion.finetune.manifest import manifest_name
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, open_stream, pack_frame, pack_frames

_CODE_VERSION = "finetune_a1_v2"
_DEFAULT_FPN_GAIN = "800"
_STATIC_CONFIDENCE_THRESHOLD = 0.9
_SPLIT_SIGMA_LIMIT_DN = 1.0
_EXPOSURE_MEDIAN_ABS_LIMIT_DN = 2.0
_HIGH_FREQUENCY_RATIO_LIMIT = 0.9
_FPN_BLUR_SIGMA = 15.0
_FPN_CLIP = 0.05
_SIGNAL_MARGIN_DN = 15.0
_SATURATION_LIMIT_DN = 3800.0
_SPLIT_VALIDATION_WINDOWS = {
    "left_static": (slice(152, 184), slice(168, 200)),
    "lower_static": (slice(408, 440), slice(36, 68)),
}
_CHART_SENSOR_REGION = (slice(800, 1000), slice(320, 580))
_DETAIL_FRAME_INDICES = (77, 177)


@dataclass(slots=True)
class TemporalReference:
    """Motion-aware temporal means for one stream."""

    mean: np.ndarray
    even_mean: np.ndarray
    odd_mean: np.ndarray
    confidence: np.ndarray
    valid_region: np.ndarray
    black_levels: dict[str, float]
    frames_total: int
    exposure_scales: np.ndarray


@dataclass(slots=True)
class FpnMaps:
    """Calibration-derived additive and multiplicative FPN components."""

    dsnu_dn: np.ndarray
    prnu_fraction: np.ndarray
    reference_offset_dn: np.ndarray


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _adapter_for_manifest(manifest: dict):
    if manifest["scene"] == "mis20s1_calibrationdata":
        return calibration_adapter(str(manifest["variant"]))
    return resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])


def _manifest_path(supervision_dir: Path, group: str, scene: str, gain: str, variant: str | None = None) -> Path:
    return supervision_dir / "stream_manifests" / f"{manifest_name(group, scene, gain, variant)}.json"


def _motion_mask_path(manifest: dict) -> Path:
    motion = manifest.get("motion")
    if motion is None:
        raise ValueError(f"{manifest['scene']}/{manifest['gain_label']}: A1 requires a motion mask")
    path = Path(motion["mask_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _accumulate_stream(
    manifest: dict,
    chunk_frames: int,
    use_motion_mask: bool,
    normalize_exposure: bool,
) -> TemporalReference:
    """Accumulate a stream without retaining all frames in memory."""

    adapter = _adapter_for_manifest(manifest)
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames_total = int(manifest["frames_used"])
    if frames_total <= 0 or frames_total > reader.frames:
        raise ValueError(f"invalid frames_used={frames_total} for {manifest['path']}")

    mask = None
    if use_motion_mask:
        mask = np.load(_motion_mask_path(manifest), mmap_mode="r")
        if mask.shape[0] < frames_total:
            raise ValueError(f"motion mask has {mask.shape[0]} frames, expected {frames_total}")

    first = pack_frames(next(reader.iter_chunks(1)), adapter)[0]
    channels, height, width = first.shape
    total = np.zeros((channels, height, width), dtype=np.float64)
    even = np.zeros_like(total)
    odd = np.zeros_like(total)
    counts = np.zeros((height, width), dtype=np.float64)
    even_counts = np.zeros_like(counts)
    odd_counts = np.zeros_like(counts)

    chunk_iterator = iter(reader.iter_chunks(chunk_frames))
    consumed = 0
    while consumed < frames_total:
        raw_chunk = next(chunk_iterator)
        take = min(raw_chunk.shape[0], frames_total - consumed)
        packed = pack_frames(raw_chunk, adapter).astype(np.float64)
        packed = packed[:take]
        valid = np.ones((take, height, width), dtype=bool)
        if mask is not None:
            valid = np.asarray(mask[consumed : consumed + take, 0, ::2, ::2] < 0.5)

        total += (packed * valid[:, None]).sum(axis=0)
        counts += valid.sum(axis=0)
        parity = (np.arange(consumed, consumed + take) % 2) == 0
        if parity.any():
            even += (packed[parity] * valid[parity, None]).sum(axis=0)
            even_counts += valid[parity].sum(axis=0)
        if (~parity).any():
            odd += (packed[~parity] * valid[~parity, None]).sum(axis=0)
            odd_counts += valid[~parity].sum(axis=0)
        consumed += take

    confidence = (counts / frames_total).astype(np.float32)
    valid_region = (confidence >= _STATIC_CONFIDENCE_THRESHOLD) & (counts > 0)
    exposure_scales = np.ones((frames_total, channels), dtype=np.float32)
    if normalize_exposure:
        preliminary_mean = total / np.maximum(counts, 1.0)
        black = _channel_vector({str(k): float(v) for k, v in manifest["black_levels"].items()}).astype(np.float64)
        signal = (
            valid_region[None]
            & (preliminary_mean > black + _SIGNAL_MARGIN_DN)
            & (preliminary_mean < _SATURATION_LIMIT_DN)
        )
        if any(int(signal[channel].sum()) < 1000 for channel in range(channels)):
            raise ValueError(f"{manifest['scene']}/{manifest['gain_label']}: insufficient stable signal for exposure normalization")

        total.fill(0.0)
        even.fill(0.0)
        odd.fill(0.0)
        counts.fill(0.0)
        even_counts.fill(0.0)
        odd_counts.fill(0.0)
        consumed = 0
        for raw_chunk in reader.iter_chunks(chunk_frames):
            take = min(raw_chunk.shape[0], frames_total - consumed)
            packed = pack_frames(raw_chunk[:take], adapter).astype(np.float64)
            valid = np.ones((take, height, width), dtype=bool)
            if mask is not None:
                valid = np.asarray(mask[consumed : consumed + take, 0, ::2, ::2] < 0.5)
            for local_index in range(take):
                frame = packed[local_index]
                for channel in range(channels):
                    selected = signal[channel] & valid[local_index]
                    target_level = float((preliminary_mean[channel][selected] - black[channel, 0, 0]).mean())
                    frame_level = float((frame[channel][selected] - black[channel, 0, 0]).mean())
                    if frame_level <= 0.0:
                        raise ValueError(
                            f"{manifest['scene']}/{manifest['gain_label']} frame {consumed + local_index}: "
                            f"non-positive exposure signal in {CHANNEL_NAMES[channel]}"
                        )
                    scale = target_level / frame_level
                    if not 0.5 <= scale <= 2.0:
                        raise ValueError(
                            f"{manifest['scene']}/{manifest['gain_label']} frame {consumed + local_index}: "
                            f"exposure scale {scale:.4f} outside [0.5, 2.0]"
                        )
                    exposure_scales[consumed + local_index, channel] = scale
                    frame[channel] = black[channel, 0, 0] + (frame[channel] - black[channel, 0, 0]) * scale

            total += (packed * valid[:, None]).sum(axis=0)
            counts += valid.sum(axis=0)
            parity = (np.arange(consumed, consumed + take) % 2) == 0
            if parity.any():
                even += (packed[parity] * valid[parity, None]).sum(axis=0)
                even_counts += valid[parity].sum(axis=0)
            if (~parity).any():
                odd += (packed[~parity] * valid[~parity, None]).sum(axis=0)
                odd_counts += valid[~parity].sum(axis=0)
            consumed += take
            if consumed >= frames_total:
                break

    safe_counts = np.maximum(counts, 1.0)
    safe_even = np.maximum(even_counts, 1.0)
    safe_odd = np.maximum(odd_counts, 1.0)
    return TemporalReference(
        mean=(total / safe_counts).astype(np.float32),
        even_mean=(even / safe_even).astype(np.float32),
        odd_mean=(odd / safe_odd).astype(np.float32),
        confidence=confidence,
        valid_region=valid_region,
        black_levels={str(k): float(v) for k, v in manifest["black_levels"].items()},
        frames_total=frames_total,
        exposure_scales=exposure_scales,
    )


def _channel_vector(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[name] for name in CHANNEL_NAMES], dtype=np.float32)[:, None, None]


def estimate_fpn_maps(
    black_reference: TemporalReference,
    flat_reference: TemporalReference,
    target_reference: TemporalReference,
) -> FpnMaps:
    """Estimate DSNU and local PRNU from matched calibration captures."""

    black_mean = black_reference.mean.astype(np.float32)
    dsnu = black_mean - np.median(black_mean, axis=(1, 2), keepdims=True)
    flat_signal = np.maximum(flat_reference.mean.astype(np.float32) - black_mean, 1.0)
    prnu = np.zeros_like(flat_signal, dtype=np.float32)
    for channel in range(flat_signal.shape[0]):
        broad = cv2.GaussianBlur(flat_signal[channel], (0, 0), _FPN_BLUR_SIGMA)
        usable = broad > 10.0
        ratio = np.zeros_like(flat_signal[channel], dtype=np.float32)
        ratio[usable] = flat_signal[channel][usable] / broad[usable] - 1.0
        prnu[channel] = np.clip(ratio, -_FPN_CLIP, _FPN_CLIP)

    target_black = _channel_vector(target_reference.black_levels)
    target_signal = np.maximum(target_reference.mean - target_black, 0.0)
    offset = dsnu + prnu * target_signal
    for channel in range(offset.shape[0]):
        selected = target_reference.valid_region & (target_signal[channel] > _SIGNAL_MARGIN_DN)
        offset[channel] -= np.median(offset[channel][selected])
    return FpnMaps(dsnu_dn=dsnu, prnu_fraction=prnu, reference_offset_dn=offset.astype(np.float32))


def apply_fpn_correction(reference: np.ndarray, fpn: FpnMaps) -> np.ndarray:
    """Remove the calibration-derived additive FPN offset from a reference."""

    if reference.shape != fpn.reference_offset_dn.shape:
        raise ValueError(f"reference/FPN shape mismatch: {reference.shape} vs {fpn.reference_offset_dn.shape}")
    corrected = reference.astype(np.float32) - fpn.reference_offset_dn
    return np.clip(corrected, 0.0, 4095.0).astype(np.float32)


def _robust_sigma(values: np.ndarray) -> float:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad == 0.0:
        return float(values.std(ddof=1))
    selected = np.abs(values - center) < 3.5 * mad / 0.6745
    return float(values[selected].std(ddof=1))


def measure_split_half_sigma(
    reference: TemporalReference,
    corrected_even: np.ndarray,
    corrected_odd: np.ndarray,
) -> dict:
    """Measure the uncertainty of the temporal mean from even/odd halves."""

    signal = reference.mean > (_channel_vector(reference.black_levels) + _SIGNAL_MARGIN_DN)
    residual = (corrected_even - corrected_odd) / 2.0
    by_region: dict[str, dict[str, float]] = {}
    for region_name, (rows, columns) in _SPLIT_VALIDATION_WINDOWS.items():
        region_values: dict[str, float] = {}
        for channel, name in enumerate(CHANNEL_NAMES):
            selected = reference.valid_region[rows, columns] & signal[channel, rows, columns]
            values = residual[channel, rows, columns][selected]
            if values.size < 500:
                raise ValueError(f"channel {name}: too few pixels in split-half region {region_name}")
            region_values[name] = _robust_sigma(values)
        by_region[region_name] = region_values
    per_channel = {
        name: max(by_region[region][name] for region in by_region)
        for name in CHANNEL_NAMES
    }
    return {"per_channel_dn": per_channel, "validation_regions": by_region}


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


def _registration_shift(manifest: dict) -> tuple[int, int]:
    registration = manifest.get("registration")
    if registration is None:
        return 0, 0
    return int(registration["integer_shift_x"]), int(registration["integer_shift_y"])


def _aligned_arrays(
    reference: np.ndarray,
    reference_confidence: np.ndarray,
    reference_valid: np.ndarray,
    moving: np.ndarray,
    moving_confidence: np.ndarray,
    moving_valid: np.ndarray,
    shift_x: int,
    shift_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = reference.shape[-2:]
    base_y0 = max(0, -shift_y)
    base_y1 = min(height, height - shift_y)
    base_x0 = max(0, -shift_x)
    base_x1 = min(width, width - shift_x)
    base = reference[..., base_y0:base_y1, base_x0:base_x1]
    base_confidence = reference_confidence[base_y0:base_y1, base_x0:base_x1]
    base_valid = reference_valid[base_y0:base_y1, base_x0:base_x1]
    moving_crop = moving[..., base_y0 + shift_y : base_y1 + shift_y, base_x0 + shift_x : base_x1 + shift_x]
    conf = moving_confidence[base_y0 + shift_y : base_y1 + shift_y, base_x0 + shift_x : base_x1 + shift_x]
    valid = moving_valid[base_y0 + shift_y : base_y1 + shift_y, base_x0 + shift_x : base_x1 + shift_x]
    return base, base_confidence, base_valid, moving_crop, conf, valid


def _check_exposure_consistency(
    target: TemporalReference,
    moving: TemporalReference,
    moving_manifest: dict,
    corrected_target: np.ndarray,
) -> dict:
    fit = moving_manifest.get("exposure_fit")
    if fit is None:
        raise ValueError("64x manifest has no A0 exposure fit")
    ref_shift = _registration_shift(moving_manifest)
    target_crop, target_conf, target_valid, moving_crop, moving_conf, moving_valid = _aligned_arrays(
        corrected_target,
        target.confidence,
        target.valid_region,
        moving.mean,
        moving.confidence,
        moving.valid_region,
        ref_shift[0],
        ref_shift[1],
    )
    selected_base = target_valid & (target_conf >= _STATIC_CONFIDENCE_THRESHOLD) & moving_valid & (moving_conf >= _STATIC_CONFIDENCE_THRESHOLD)
    per_channel: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        slope = float(fit["slopes"][name])
        intercept = float(fit["intercepts"][name])
        normalized = slope * moving_crop[channel] + intercept
        selected = (
            selected_base
            & (target_crop[channel] > target.black_levels[name] + _SIGNAL_MARGIN_DN)
            & (moving_crop[channel] > moving.black_levels[name] + _SIGNAL_MARGIN_DN)
            & (target_crop[channel] < _SATURATION_LIMIT_DN)
            & (moving_crop[channel] < _SATURATION_LIMIT_DN)
        )
        residual = (normalized - target_crop[channel])[selected]
        if residual.size < 1000:
            raise ValueError(f"channel {name}: too few pixels for 64x consistency check")
        per_channel[name] = {
            "signed_median_dn": float(np.median(residual)),
            "absolute_median_dn": float(np.median(np.abs(residual))),
            "pixels": int(residual.size),
        }
    return {
        "limit_abs_signed_median_dn": _EXPOSURE_MEDIAN_ABS_LIMIT_DN,
        "per_channel": per_channel,
        "passed": all(abs(item["signed_median_dn"]) <= _EXPOSURE_MEDIAN_ABS_LIMIT_DN for item in per_channel.values()),
    }


def _unpack_rggb(packed: np.ndarray) -> np.ndarray:
    """Restore a canonical packed array to an RGGB sensor mosaic."""

    height, width = packed.shape[-2:]
    mosaic = np.empty((height * 2, width * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 0::2] = packed[2]
    mosaic[1::2, 1::2] = packed[3]
    return mosaic


def _mosaic_black(black_levels: dict[str, float], shape: tuple[int, int]) -> np.ndarray:
    packed = np.stack([
        np.full(shape, black_levels[name], dtype=np.float32)
        for name in CHANNEL_NAMES
    ])
    return _unpack_rggb(packed)


def _box_highpass_energy(image: np.ndarray, selected: np.ndarray) -> float:
    highpass = image - cv2.boxFilter(image, cv2.CV_32F, (5, 5))
    return float(np.mean(np.abs(highpass[selected])))


def _unpack_valid_region(valid_region: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(valid_region, 2, axis=0), 2, axis=1)


def _high_frequency_check(
    target: TemporalReference,
    corrected_target: np.ndarray,
    noisy_manifest: dict,
    noisy_reference: TemporalReference,
) -> dict:
    adapter = _adapter_for_manifest(noisy_manifest)
    reader, _ = open_stream(Path(noisy_manifest["path"]), adapter)
    rows, columns = _CHART_SENSOR_REGION
    target_mosaic = _unpack_rggb(corrected_target)
    target_black = _mosaic_black(target.black_levels, corrected_target.shape[-2:])
    noisy_mean_mosaic = _unpack_rggb(noisy_reference.mean)
    noisy_black = _mosaic_black(noisy_reference.black_levels, noisy_reference.mean.shape[-2:])
    valid_mosaic = _unpack_valid_region(target.valid_region)
    valid_chart = cv2.erode(
        valid_mosaic[rows, columns].astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)
    if int(valid_chart.sum()) < 1000:
        raise ValueError("too few valid pixels in A1 high-frequency chart region")
    target_chart = target_mosaic[rows, columns]
    target_black_chart = target_black[rows, columns]
    noisy_mean_chart = noisy_mean_mosaic[rows, columns]
    noisy_black_chart = noisy_black[rows, columns]
    target_signal = float(np.mean((target_chart - target_black_chart)[valid_chart]))
    noisy_signal = float(np.mean((noisy_mean_chart - noisy_black_chart)[valid_chart]))
    if target_signal <= 0.0 or noisy_signal <= 0.0:
        raise ValueError("non-positive chart signal in A1 high-frequency check")
    contrast_ratio = target_signal / noisy_signal
    target_energy = _box_highpass_energy(target_chart, valid_chart) / contrast_ratio

    per_frame: list[dict[str, float | int]] = []
    black = _channel_vector(noisy_reference.black_levels)
    for frame_index in _DETAIL_FRAME_INDICES:
        noisy_frame = pack_frame(reader.read_frame(frame_index), adapter).astype(np.float32)
        scale = noisy_reference.exposure_scales[frame_index][:, None, None]
        noisy_frame = black + (noisy_frame - black) * scale
        noisy_chart = _unpack_rggb(noisy_frame)[rows, columns]
        single_energy = _box_highpass_energy(noisy_chart, valid_chart)
        pair_index = frame_index - 1
        previous_frame = pack_frame(reader.read_frame(pair_index), adapter).astype(np.float32)
        previous_scale = noisy_reference.exposure_scales[pair_index][:, None, None]
        previous_frame = black + (previous_frame - black) * previous_scale
        previous_chart = _unpack_rggb(previous_frame)[rows, columns]
        noise_floor = _box_highpass_energy(
            (noisy_chart - previous_chart) / np.sqrt(2.0),
            valid_chart,
        )
        detail_energy = max(single_energy - noise_floor, 0.0)
        per_frame.append({
            "frame": frame_index,
            "single_highpass_dn": single_energy,
            "noise_floor_dn": noise_floor,
            "noise_corrected_detail_dn": detail_energy,
        })
    noisy_energy = float(np.median([item["noise_corrected_detail_dn"] for item in per_frame]))
    ratio = target_energy / max(noisy_energy, 1e-6)
    valid_fraction = float(valid_chart.mean())
    return {
        "sensor_region": [rows.start, rows.stop, columns.start, columns.stop],
        "valid_fraction": valid_fraction,
        "contrast_ratio_10x_to_645x": contrast_ratio,
        "reference_detail_at_645x_scale_dn": target_energy,
        "645x_noise_corrected_detail_dn": noisy_energy,
        "per_frame": per_frame,
        "energy_ratio": ratio,
        "limit_ratio": _HIGH_FREQUENCY_RATIO_LIMIT,
        "passed": ratio >= _HIGH_FREQUENCY_RATIO_LIMIT,
    }


def build_a1_references(
    dataset_root: Path,
    out_dir: Path,
    chunk_frames: int = 25,
    fpn_gain: str = _DEFAULT_FPN_GAIN,
) -> dict:
    """Build A1 references and write a self-contained quality report."""

    supervision_dir = out_dir / "supervision"
    reference_dir = supervision_dir / "references"
    main_manifest = _load_manifest(_manifest_path(supervision_dir, "04", "shdarkroom", "10x"))
    ref10 = _accumulate_stream(
        main_manifest,
        chunk_frames,
        use_motion_mask=True,
        normalize_exposure=True,
    )

    calibration_black_manifest = _load_manifest(
        _manifest_path(supervision_dir, "04", "mis20s1_calibrationdata", fpn_gain, "black")
    )
    calibration_flat_manifest = _load_manifest(
        _manifest_path(supervision_dir, "04", "mis20s1_calibrationdata", fpn_gain, "color")
    )
    cal_black = _accumulate_stream(
        calibration_black_manifest,
        chunk_frames,
        use_motion_mask=False,
        normalize_exposure=False,
    )
    cal_flat = _accumulate_stream(
        calibration_flat_manifest,
        chunk_frames,
        use_motion_mask=False,
        normalize_exposure=False,
    )
    fpn = estimate_fpn_maps(cal_black, cal_flat, ref10)
    clean10 = apply_fpn_correction(ref10.mean, fpn)
    clean_even = apply_fpn_correction(ref10.even_mean, fpn)
    clean_odd = apply_fpn_correction(ref10.odd_mean, fpn)

    split_sigma = measure_split_half_sigma(ref10, clean_even, clean_odd)
    split_check = {
        "limit_dn": _SPLIT_SIGMA_LIMIT_DN,
        **split_sigma,
        "passed": all(
            value <= _SPLIT_SIGMA_LIMIT_DN
            for value in split_sigma["per_channel_dn"].values()
        ),
    }

    secondary: dict[str, dict] = {}
    secondary_refs: dict[str, TemporalReference] = {}
    for gain in ("64x", "128x", "645x"):
        manifest = _load_manifest(_manifest_path(supervision_dir, "04", "shdarkroom", gain))
        temporal = _accumulate_stream(
            manifest,
            chunk_frames,
            use_motion_mask=True,
            normalize_exposure=True,
        )
        secondary_refs[gain] = temporal
        mean_path = reference_dir / f"04_shdarkroom_{gain}_temporal_mean.npy"
        confidence_path = reference_dir / f"04_shdarkroom_{gain}_confidence.npy"
        _save_array(mean_path, temporal.mean)
        _save_array(confidence_path, temporal.confidence)
        secondary[gain] = {
            "mean_path": str(mean_path),
            "confidence_path": str(confidence_path),
            "frames": temporal.frames_total,
            "static_fraction": float(temporal.valid_region.mean()),
            "exposure_scale_min": {
                name: float(temporal.exposure_scales[:, channel].min())
                for channel, name in enumerate(CHANNEL_NAMES)
            },
            "exposure_scale_max": {
                name: float(temporal.exposure_scales[:, channel].max())
                for channel, name in enumerate(CHANNEL_NAMES)
            },
        }

    manifest64 = _load_manifest(_manifest_path(supervision_dir, "04", "shdarkroom", "64x"))
    consistency_check = _check_exposure_consistency(ref10, secondary_refs["64x"], manifest64, clean10)
    manifest645 = _load_manifest(_manifest_path(supervision_dir, "04", "shdarkroom", "645x"))
    high_frequency_check = _high_frequency_check(
        ref10,
        clean10,
        manifest645,
        secondary_refs["645x"],
    )

    reference_path = reference_dir / "04_shdarkroom_10x_clean_reference.npy"
    confidence_path = reference_dir / "04_shdarkroom_10x_confidence.npy"
    valid_path = reference_dir / "04_shdarkroom_10x_valid_region.npy"
    dsnu_path = reference_dir / "04_shdarkroom_fpn_dsnu_dn.npy"
    prnu_path = reference_dir / "04_shdarkroom_fpn_prnu_fraction.npy"
    offset_path = reference_dir / "04_shdarkroom_fpn_offset_dn.npy"
    _save_array(reference_path, clean10)
    _save_array(confidence_path, ref10.confidence)
    _save_array(valid_path, ref10.valid_region)
    _save_array(dsnu_path, fpn.dsnu_dn)
    _save_array(prnu_path, fpn.prnu_fraction)
    _save_array(offset_path, fpn.reference_offset_dn)

    checks = {
        "split_half": split_check,
        "cross_gain_64x": consistency_check,
        "high_frequency_645x": high_frequency_check,
    }
    report = {
        "code_version": _CODE_VERSION,
        "dataset_root": str(dataset_root),
        "main_stream": "04/shdarkroom/10x",
        "frames_used": ref10.frames_total,
        "shape": list(clean10.shape),
        "black_levels": ref10.black_levels,
        "static_confidence_threshold": _STATIC_CONFIDENCE_THRESHOLD,
        "static_fraction": float(ref10.valid_region.mean()),
        "exposure_scale_min": {
            name: float(ref10.exposure_scales[:, channel].min())
            for channel, name in enumerate(CHANNEL_NAMES)
        },
        "exposure_scale_max": {
            name: float(ref10.exposure_scales[:, channel].max())
            for channel, name in enumerate(CHANNEL_NAMES)
        },
        "paths": {
            "clean_reference": str(reference_path),
            "confidence": str(confidence_path),
            "valid_region": str(valid_path),
            "fpn_dsnu_dn": str(dsnu_path),
            "fpn_prnu_fraction": str(prnu_path),
            "fpn_offset_dn": str(offset_path),
        },
        "fpn_calibration_gain": fpn_gain,
        "secondary_references": secondary,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
    report_path = reference_dir / "a1_reference_report.json"
    manifest_path = reference_dir / "a1_reference_manifest.json"
    _save_json(report_path, report)
    _save_json(manifest_path, {
        "code_version": _CODE_VERSION,
        "label_semantics": "clean_reference_for_raw_denoising",
        "source_stream": "04/shdarkroom/10x",
        "frames_used": ref10.frames_total,
        "shape": list(clean10.shape),
        "valid_region_path": str(valid_path),
        "confidence_path": str(confidence_path),
        "clean_reference_path": str(reference_path),
        "fpn_paths": {
            "dsnu_dn": str(dsnu_path),
            "prnu_fraction": str(prnu_path),
            "offset_dn": str(offset_path),
        },
        "quality_report_path": str(report_path),
        "quality_passed": report["passed"],
    })
    return report

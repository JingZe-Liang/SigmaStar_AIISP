"""Frozen ROI measurements from existing A3 predictions and native RAW."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from raw_fusion.finetune.cross_sensor import (
    NativeSequenceSpec,
    _motion_threshold,
    channel_vector,
    contiguous_clip_indices,
    denormalise_native,
    open_sequence,
    read_normalized_clip,
    split_clip_indices,
)
from raw_fusion.finetune.stats import MIN_SIGNAL_PIXELS, SIGNAL_MARGIN_DN


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spatial_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.stack([cv2.blur(channel, (3, 3)) for channel in values])
    return low, values - low


def static_error_sums(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, region: np.ndarray | None = None) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 4:
        raise ValueError("prediction and target must have matching [4,H,W] shapes")
    if valid.shape != target.shape[1:] or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean [H,W] mask")
    if region is not None and (region.shape != valid.shape or region.dtype != np.bool_):
        raise ValueError("region must be a boolean [H,W] mask")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("non-finite RAW values")
    support = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    selection = support if region is None else support & region
    mask4 = np.broadcast_to(selection, target.shape)
    error = prediction - target
    low, high = spatial_components(error)
    values = error[mask4].astype(np.float64)
    horizontal_mask = np.broadcast_to(support[:, 1:] & selection[:, :-1], error[:, :, 1:].shape)
    vertical_mask = np.broadcast_to(support[1:] & selection[:-1], error[:, 1:, :].shape)
    horizontal = np.diff(error, axis=2)[horizontal_mask].astype(np.float64)
    vertical = np.diff(error, axis=1)[vertical_mask].astype(np.float64)
    result = {
        "count": int(values.size),
        "error_sum": float(values.sum()),
        "error_sq": float(np.square(values).sum()),
        "highpass_sq": float(np.square(high[mask4].astype(np.float64)).sum()),
        "lowpass_sq": float(np.square(low[mask4].astype(np.float64)).sum()),
        "gradient_sq": float(np.square(horizontal).sum() + np.square(vertical).sum()),
        "gradient_count": int(horizontal.size + vertical.size),
    }
    for channel, name in enumerate(("R", "Gr", "Gb", "B")):
        result[f"error_sum_{name}"] = float(np.sum(error[channel][selection], dtype=np.float64))
    return result


def summarize_static(sums: dict) -> dict:
    count = sums.get("count", 0)
    if count == 0:
        return {"status": "no_valid_support", "values": 0}
    else:
        mse = sums["error_sq"] / count
        bias = sums["error_sum"] / count
        return {
            "status": "measured",
            "values": count,
            "rmse_dn": float(np.sqrt(mse)),
            "psnr_db_4095": float(10 * np.log10(4095.0 ** 2 / mse)) if mse > 0 else None,
            "bias_dn": bias,
            "bias_dn_per_channel": {name: sums[f"error_sum_{name}"] / (count / 4) for name in ("R", "Gr", "Gb", "B")},
            "pooled_residual_std_dn": float(np.sqrt(max(mse - bias * bias, 0))),
            "highpass_error_rmse_dn": float(np.sqrt(sums["highpass_sq"] / count)),
            "lowpass_error_rmse_dn": float(np.sqrt(sums["lowpass_sq"] / count)),
            "signed_gradient_error_rmse_dn": float(np.sqrt(sums["gradient_sq"] / sums["gradient_count"])) if sums["gradient_count"] else None,
            "gradient_values": sums["gradient_count"],
        }


def reference_regions(target: np.ndarray, valid: np.ndarray, black: np.ndarray) -> dict[str, np.ndarray]:
    if not valid.any():
        raise ValueError("reference has no valid pixels")
    gray = target.mean(axis=0)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    signal = (target - black[:, None, None]).mean(axis=0)
    return {
        "all_static": valid,
        "detail": valid & (gradient >= np.percentile(gradient[valid], 75)) & (gradient > 0),
        "flat": valid & (gradient <= np.percentile(gradient[valid], 25)),
        "dark": valid & (signal <= np.percentile(signal[valid], 25)),
    }


def stress_sums(inputs: np.ndarray, prediction: np.ndarray, normalized: np.ndarray) -> dict:
    if inputs.shape != prediction.shape or normalized.shape != prediction.shape or inputs.ndim != 4:
        raise ValueError("stress inputs must have matching [T,4,H,W] shapes")
    if not np.isfinite(inputs).all() or not np.isfinite(prediction).all():
        raise ValueError("non-finite stress values")
    input_hp = np.stack([spatial_components(frame)[1] for frame in inputs])[:, :, 1:-1, 1:-1]
    pred_hp = np.stack([spatial_components(frame)[1] for frame in prediction])[:, :, 1:-1, 1:-1]
    return {
        "count": int(inputs.size),
        "change_sum": float(np.sum(prediction - inputs, dtype=np.float64)),
        "clipped_count": int(((normalized <= 0) | (normalized >= 1)).sum()),
        "lower_count": int((normalized <= 0).sum()),
        "upper_count": int((normalized >= 1).sum()),
        "input_temporal_abs": float(np.sum(np.abs(np.diff(inputs, axis=0)), dtype=np.float64)),
        "output_temporal_abs": float(np.sum(np.abs(np.diff(prediction, axis=0)), dtype=np.float64)),
        "temporal_count": int(inputs[1:].size),
        "input_hp_sq": float(np.sum(np.square(input_hp), dtype=np.float64)),
        "output_hp_sq": float(np.sum(np.square(pred_hp), dtype=np.float64)),
        "hp_count": int(input_hp.size),
    }


def _add(totals: dict, values: dict) -> None:
    for key, value in values.items():
        totals[key] = totals.get(key, 0) + value


def _summarize_stress(sums: dict) -> dict:
    return {
        "status": "diagnostic_only_no_clean_dynamic_target",
        "values": sums["count"],
        "mean_change_from_noisy_input_dn": sums["change_sum"] / sums["count"],
        "endpoint_clip_fraction": sums["clipped_count"] / sums["count"],
        "lower_endpoint_fraction": sums["lower_count"] / sums["count"],
        "upper_endpoint_fraction": sums["upper_count"] / sums["count"],
        "input_temporal_mad_dn": sums["input_temporal_abs"] / sums["temporal_count"],
        "output_temporal_mad_dn": sums["output_temporal_abs"] / sums["temporal_count"],
        "temporal_response_ratio_to_noisy_input": sums["output_temporal_abs"] / sums["input_temporal_abs"] if sums["input_temporal_abs"] > 0 else None,
        "hf_rms_ratio_to_noisy_input": float(np.sqrt(sums["output_hp_sq"] / sums["input_hp_sq"])) if sums["input_hp_sq"] > 0 else None,
    }


def measure_sequence(entry: dict, checkpoint_hash: str, verify_hashes: bool = True) -> dict:
    source_config = json.loads(Path(entry["source_config"]).read_text())
    spec = NativeSequenceSpec.from_mapping(entry["source_key"], source_config["sequences"][entry["source_key"]])
    root = Path(entry["artifact_dir"])
    manifest_path = root / "inference/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest["provenance"]
    if file_sha256(Path(entry["source_config"])) != provenance["protocol_sha256"]:
        raise ValueError(f"{spec.name}: original source configuration hash mismatch")
    if provenance["checkpoint_sha256"] != checkpoint_hash or provenance["weights"] != "ema":
        raise ValueError(f"{spec.name}: cache is not the frozen A3 EMA")
    if Path(provenance["source_path"]).resolve() != spec.path.resolve() or spec.path.stat().st_size != provenance["source_size_bytes"]:
        raise ValueError(f"{spec.name}: RAW identity differs from cache manifest")
    if any(provenance[key] != getattr(spec, key) for key in ("width", "height", "bit_shift", "cfa", "frames_used")):
        raise ValueError(f"{spec.name}: RAW adapter differs from cache manifest")
    static = entry["assessment_kind"] == "cross_sensor_static"
    profile_dir = root / ("reference" if static else "profile")
    profile = json.loads((profile_dir / "report.json").read_text())
    if any(provenance[key] != value for key, value in profile["provenance"].items()):
        raise ValueError(f"{spec.name}: reference/profile provenance differs from inference")
    black = channel_vector(profile["black_levels"])
    frozen_rois = entry["roi_plan"]["panel_crops_raw_xyxy"]
    if frozen_rois != {name: list(bounds) for name, bounds in spec.panel_crops}:
        raise ValueError(f"{spec.name}: frozen ROI differs from source protocol")
    roi_slices = {}
    for name, bounds in spec.panel_crops:
        x0, y0, x1, y1 = bounds
        roi_slices[name] = (slice((y0 + 1) // 2, y1 // 2), slice((x0 + 1) // 2, x1 // 2))
    if not roi_slices:
        raise ValueError(f"{spec.name}: fixed ROI definitions are required")
    reader = open_sequence(spec)
    directions = {}
    for direction, cache in manifest["directions"].items():
        prediction_path = Path(cache["path"])
        if verify_hashes and file_sha256(prediction_path) != cache["sha256"]:
            raise ValueError(f"{spec.name}: prediction SHA-256 mismatch")
        prediction = np.load(prediction_path, mmap_mode="r")
        if list(prediction.shape) != cache["prediction_shape"] or str(prediction.dtype) != cache["prediction_dtype"]:
            raise ValueError(f"{spec.name}: prediction cache metadata mismatch")
        clips = cache["clips"]
        expected_clips = split_clip_indices(spec, cache["parity"]) if static else contiguous_clip_indices(spec)
        if clips != expected_clips:
            raise ValueError(f"{spec.name}: cached frame indices differ from frozen protocol")
        if prediction.shape[0] != sum(map(len, clips)) or any(len(indices) != 8 for indices in clips):
            raise ValueError(f"{spec.name}: expected complete eight-frame cached clips")
        if prediction.shape[1:] != (4, *spec.packed_shape):
            raise ValueError(f"{spec.name}: cached dimensions differ from RAW")
        totals = {name: {} for name in roi_slices}
        endpoint_totals = {name: {} for name in roi_slices}
        if static:
            parity = cache["parity"]
            if parity not in (0, 1) or any(index % 2 != parity for clip in clips for index in clip):
                raise ValueError(f"{spec.name}: input parity is not disjoint")
            target_name = "odd_mean" if parity == 0 else "even_mean"
            loaded = {}
            for key in (target_name, "mean", "valid_region", "confidence", "even_confidence", "odd_confidence"):
                record = profile["outputs"][key]
                path = Path(record["path"])
                if verify_hashes and file_sha256(path) != record["sha256"]:
                    raise ValueError(f"{spec.name}: reference SHA-256 mismatch: {key}")
                loaded[key] = np.load(path, mmap_mode="r")
            target = np.asarray(loaded[target_name])
            valid = loaded["valid_region"].astype(bool)
            for key in ("confidence", "even_confidence", "odd_confidence"):
                valid &= loaded[key] >= 0.9
            reference_gray = loaded["mean"].mean(axis=0)
            motion_signal = reference_gray > float(black.mean()) + SIGNAL_MARGIN_DN
            if int(motion_signal.sum()) < MIN_SIGNAL_PIXELS:
                motion_signal = np.ones_like(valid)
            regions = reference_regions(target, valid, black)
        else:
            target_name = None
        for clip_index, indices in enumerate(clips):
            normalized_input = read_normalized_clip(reader, spec.adapter, indices, black)
            normalized_output = np.asarray(prediction[clip_index * 8:(clip_index + 1) * 8], dtype=np.float32)
            if not np.isfinite(normalized_output).all() or np.any((normalized_output < 0) | (normalized_output > 1)):
                raise ValueError(f"{spec.name}: cached output outside normalized RAW contract")
            inputs = denormalise_native(normalized_input, black)
            outputs = denormalise_native(normalized_output, black)
            for roi_name, (ys, xs) in roi_slices.items():
                roi_output = normalized_output[:, :, ys, xs]
                _add(endpoint_totals[roi_name], {"count": int(roi_output.size), "lower": int((roi_output <= 0).sum()), "upper": int((roi_output >= 1).sum())})
            if static:
                for index in range(8):
                    moving, _ = _motion_threshold(inputs[index].mean(axis=0), reference_gray, motion_signal, 3.0)
                    usable = ~moving & np.all(inputs[index] < 3800, axis=0)
                    for roi_name, (ys, xs) in roi_slices.items():
                        for region_name, region in regions.items():
                            key = f"{region_name}"
                            region_totals = totals[roi_name].setdefault(key, {"input": {}, "a3": {}})
                            mask = (usable & valid)[ys, xs]
                            for label, frame in (("input", inputs[index]), ("a3", outputs[index])):
                                _add(region_totals[label], static_error_sums(frame[:, ys, xs], target[:, ys, xs], mask, region[ys, xs]))
            else:
                for roi_name, (ys, xs) in roi_slices.items():
                    _add(totals[roi_name], stress_sums(inputs[:, :, ys, xs], outputs[:, :, ys, xs], normalized_output[:, :, ys, xs]))
        if static:
            measurements = {roi: {region: {label: summarize_static(sums) for label, sums in branches.items()} for region, branches in values.items()} for roi, values in totals.items()}
        else:
            measurements = {roi: _summarize_stress(sums) for roi, sums in totals.items()}
        directions[direction] = {
            "clips": clips,
            "target": target_name,
            "prediction_sha256": cache["sha256"],
            "roi_metrics": measurements,
            "roi_endpoint_fractions_all_pixels": {name: {"lower": values["lower"] / values["count"], "upper": values["upper"] / values["count"], "values": values["count"]} for name, values in endpoint_totals.items()},
        }
    return {
        "id": entry["id"],
        "sensor": entry["sensor"],
        "assessment_kind": entry["assessment_kind"],
        "data_identity": entry["data_identity"],
        "measurement_origin": "new_CPU_measurements_from_historical_A3_prediction_cache",
        "raw_source_size_verified": True,
        "raw_source_full_hash_rechecked": False,
        "prediction_and_reference_hashes_rechecked": verify_hashes,
        "cache_manifest_sha256": file_sha256(manifest_path),
        "historical_evidence": entry["historical_evidence"],
        "roi_value_budgets_per_frame": {name: 4 * (ys.stop - ys.start) * (xs.stop - xs.start) for name, (ys, xs) in roi_slices.items()},
        "residual_interpretation": "Pooled residual spread includes channel bias differences and reference error; 3x3 lowpass is a local residual measure, not a complete large-scale chroma-blotch measurement.",
        "directions": directions,
        "acceptance": "baseline_recorded_not_candidate_acceptance",
    }

"""Frozen native static ROIs and target-relative multi-sensor evaluation."""

import hashlib
import json
import os
from pathlib import Path
import re
import time
from uuid import UUID

import cv2
import numpy as np
import torch

from raw_fusion.finetune.a12_dynamic_probe import _erode_masks, _spatial_highpass
from raw_fusion.finetune.a12_paired_eval import load_selected_model
from raw_fusion.finetune.a12_train import sha256_file
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.cross_sensor import (
    NativeSequenceSpec, channel_vector, denormalise_native, normalise_native_dn,
)
from raw_fusion.finetune.stream_io import pack_frame


REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_SIZE = 64
FRAMES = 8
SCHEMA = "a14_static_roi_v1"
ROLES = {"gain_holdout", "train_distribution"}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _output_path(value: Path) -> Path:
    path = value.resolve()
    relative = path.relative_to((REPO_ROOT / "artifacts").resolve())
    if not relative.parts or not relative.parts[0].startswith("a14_"):
        raise ValueError("output must be within a repository artifacts/a14_* directory")
    return path


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _array_identity(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def select_reference_rois(
    reference: np.ndarray, valid: np.ndarray, *, count: int = 3, grid_size: int = 7,
    min_valid_fraction: float = 0.25,
    eligible_rois: set[tuple[int, int]] | None = None,
) -> list[dict]:
    """Freeze low, middle, and high texture crops using reference evidence only."""

    if reference.ndim != 3 or reference.shape[0] != 4 or valid.shape != reference.shape[1:]:
        raise ValueError("reference and valid region shapes do not match")
    if count <= 0 or grid_size <= 0 or not 0 < min_valid_fraction <= 1:
        raise ValueError("invalid reference ROI selection parameters")
    height, width = valid.shape
    if min(height, width) < PATCH_SIZE or not np.isfinite(reference).all():
        raise ValueError("reference must be finite and at least 64 packed pixels wide and high")
    gray = reference.astype(np.float32).mean(axis=0)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    candidates = []
    for y in np.unique(np.linspace(0, height - PATCH_SIZE, grid_size).astype(int)):
        for x in np.unique(np.linspace(0, width - PATCH_SIZE, grid_size).astype(int)):
            if eligible_rois is not None and (int(y), int(x)) not in eligible_rois:
                continue
            mask = valid[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if float(mask.mean()) >= min_valid_fraction:
                candidates.append({"y": int(y), "x": int(x), "size": PATCH_SIZE,
                                   "reference_valid_fraction": float(mask.mean()),
                                   "reference_gradient_mean": float(gradient[y:y + PATCH_SIZE, x:x + PATCH_SIZE][mask].mean())})
    candidates.sort(key=lambda item: (item["reference_gradient_mean"], item["y"], item["x"]))
    if len(candidates) < count:
        raise ValueError("too few valid reference grid ROIs; no candidate-dependent replacement is allowed")
    indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [candidates[int(index)] for index in indices]


def _region_masks(reference: np.ndarray, valid: np.ndarray, black: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    gray = reference.astype(np.float32).mean(axis=0)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    signal = valid & (gray > float(black.mean()) + 15.0)
    if not signal.any():
        raise ValueError("reference has no valid signal pixels for detail/flat masks")
    flat_threshold, detail_threshold = np.percentile(gradient[signal], [25, 75])
    return signal & (gradient <= flat_threshold), signal & (gradient >= detail_threshold), {
        "selector": "reference-only Sobel magnitude; signal > mean black + 15 DN",
        "flat_percentile": 25, "detail_percentile": 75,
        "flat_threshold_dn": float(flat_threshold), "detail_threshold_dn": float(detail_threshold),
    }


def _load_source(config: dict) -> dict:
    reference_dir = _path(config["reference_dir"])
    report_path = reference_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    files = {"report": _identity(report_path)}
    kind = config["kind"]
    if kind == "cross_sensor_split_half":
        spec = NativeSequenceSpec.from_mapping(config["source"]["name"], config["source"])
        if spec.reference_mode != "motion_aware_split_half":
            raise ValueError("a static evaluation source must have a split-half reference")
        raw_path, adapter = spec.path.resolve(), spec.adapter
        width, height, frames_used = spec.width, spec.height, spec.frames_used
        provenance = report["provenance"]
        if Path(provenance["source_path"]).resolve() != raw_path:
            raise ValueError("reference RAW path does not match the source")
        if provenance["source_sha256"] != config["source_sha256"]:
            raise ValueError("reference source SHA-256 does not match configuration")
        names = {"mean": "full_mean_dn.npy", "even": "even_mean_dn.npy", "odd": "odd_mean_dn.npy",
                 "valid": "valid_region.npy", "even_confidence": "even_confidence.npy",
                 "odd_confidence": "odd_confidence.npy", "thresholds": "motion_threshold_per_frame.npy"}
        black = channel_vector(report["black_levels"])
        extra = {"source_sha256_reported": provenance["source_sha256"],
                 "brightness_limitation": "Native split-half means; residual target noise and temporal drift remain."}
    elif kind == "mis20s1_native_split_half":
        manifest_path = _path(config["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["group"], manifest["scene"], manifest["gain_label"]) != ("04", "shdarkroom", "1000x"):
            raise ValueError("MIS native holdout must be the original darkroom 1000x manifest")
        raw_path = _path(manifest["path"])
        if Path(report["source_raw"]).resolve() != raw_path:
            raise ValueError("MIS reference RAW path does not match the manifest")
        adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
        width, height, frames_used = manifest["width"], manifest["height"], manifest["frames_used"]
        names = {"mean": "temporal_mean_native_dn.npy", "even": "even_mean_native_dn.npy",
                 "odd": "odd_mean_native_dn.npy", "valid": "valid_region.npy", "scales": "exposure_scales.npy"}
        black = channel_vector(manifest["black_levels"])
        motion_path = _path(manifest["motion"]["mask_path"])
        motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
        if motion.shape != (frames_used, 1, height, width):
            raise ValueError("MIS motion mask does not match the native RAW stream")
        files.update(manifest=_identity(manifest_path), motion=_identity(motion_path))
        extra = {"motion": motion, "source_sha256_reported": config.get("source_sha256"),
                 "brightness_limitation": "Existing opposite-parity means were exposure-normalized during construction. Inputs and predictions remain native DN, without per-case brightness fitting. Small drift and reference-estimation dependence remain; this is a development gain holdout, not independent clean GT."}
    else:
        raise ValueError(f"unsupported static source kind {kind!r}")
    arrays = {}
    for name, filename in names.items():
        path = reference_dir / filename
        files[name] = _identity(path)
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        reported = next((entry for entry in report["outputs"].values() if Path(entry["path"]).name == filename), None)
        if reported is None or reported["sha256"] != files[name]["sha256"]:
            raise ValueError(f"reference file identity is missing or mismatched: {path}")
    shape = (height // 2, width // 2)
    for name in ("mean", "even", "odd"):
        if arrays[name].shape != (4, *shape):
            raise ValueError(f"{name} reference shape does not match the source")
    if arrays["valid"].shape != shape or arrays["valid"].dtype != bool:
        raise ValueError("reference valid region must be a packed boolean mask")
    frame_bytes = width * height * 2
    size = raw_path.stat().st_size
    if size % frame_bytes or size // frame_bytes < frames_used or report["frames_used"] != frames_used:
        raise ValueError("RAW or reference frame count does not match the source")
    raw = np.memmap(raw_path, dtype="<u2", mode="r", shape=(size // frame_bytes, height, width))
    return {"kind": kind, "raw": raw, "raw_path": raw_path, "adapter": adapter,
            "black": black, "frames_used": frames_used, "files": files, **arrays, **extra}


def _roi_case(
    source: dict, reference_valid: np.ndarray, y: int, x: int, start: int, parity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch = PATCH_SIZE
    area = (slice(y, y + patch), slice(x, x + patch))
    indices = np.arange(start + parity, start + parity + FRAMES * 2, 2)
    noisy = np.stack([pack_frame(np.asarray(source["raw"][index, 2*y:2*(y+patch), 2*x:2*(x+patch)]), source["adapter"]) for index in indices]).astype(np.float32)
    target_half = "odd" if parity == 0 else "even"
    target_frame = np.asarray(source[target_half][:, *area], dtype=np.float32)
    valid = reference_valid[area].copy()
    if source["kind"] == "cross_sensor_split_half":
        thresholds = np.asarray(source["thresholds"][indices], dtype=np.float32)[:, None, None]
        ref_gray = np.asarray(source["mean"][:, *area], dtype=np.float32).mean(axis=0)
        valid &= np.all(np.abs(noisy.mean(axis=1) - ref_gray) <= thresholds, axis=0)
        valid &= source[f"{target_half}_confidence"][area] >= 0.9
    else:
        motion_roi = np.asarray(source["motion"][:, 0, 2*y:2*(y+patch):2, 2*x:2*(x+patch):2])
        if not np.isfinite(motion_roi).all():
            raise ValueError("motion mask contains non-finite values")
        valid &= (motion_roi[1 - parity::2] < 0.5).mean(axis=0) >= 0.9
        valid &= np.all(motion_roi[indices] < 0.5, axis=0)
    valid &= np.all(noisy < 3800.0, axis=(0, 1)) & np.all(target_frame < 3800.0, axis=0)
    return noisy, target_frame, valid, indices


def prepare_static_cases(config_path: Path, prepared_dir: Path) -> dict:
    """Cache model-independent ROIs, exact native inputs, opposite-parity targets and masks."""

    config_path, prepared_dir = _path(config_path), _output_path(prepared_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    minimum = float(config.get("min_valid_fraction", 0.25))
    if not 0 < minimum <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    source_names = [source["name"] for source in config["sources"]]
    if not source_names or len(source_names) != len(set(source_names)):
        raise ValueError("source names must be nonempty and unique")
    prepared_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "config": config, "config_identity": _identity(config_path),
              "roi_policy": "Fixed spatial grid filtered by reference validity and original input masks for all predefined windows/parities, then ranked by reference gradient. No model outputs are inspected. ROIs and masks are frozen before comparing checkpoints.",
              "inference": {"frames": FRAMES, "packed_patch_size": PATCH_SIZE, "batch_size": 1,
                            "precision": "float32", "tf32": False, "output_clip": [0.0, 1.0]},
              "scope": "Native static ROI diagnostics; training-domain reevaluation is not unseen-scene or unseen-sensor generalization. Opposite parity prevents direct target averaging of evaluated input frames, but masks, reference selection and brightness estimation may still depend on the full sequence.",
              "sources": [], "cases": []}
    for config_source in config["sources"]:
        name, role = config_source["name"], config_source["role"]
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None or role not in ROLES:
            raise ValueError("source name or evaluation role is invalid")
        source = _load_source(config_source)
        starts = config_source["clip_starts"]
        if not starts or len(starts) != len(set(starts)) or any(start < 0 or start % 2 or start + FRAMES * 2 > source["frames_used"] for start in starts):
            raise ValueError("clip starts must be unique even indices covering complete split-half clips")
        reference_valid = np.asarray(source["valid"], dtype=bool)
        if source["kind"] == "cross_sensor_split_half":
            reference_valid = reference_valid & (source["even_confidence"] >= 0.9) & (source["odd_confidence"] >= 0.9)
        grid_size = int(config.get("roi_grid_size", 7))
        eligible = set()
        height, width = reference_valid.shape
        for y in np.unique(np.linspace(0, height - PATCH_SIZE, grid_size).astype(int)):
            for x in np.unique(np.linspace(0, width - PATCH_SIZE, grid_size).astype(int)):
                if float(reference_valid[y:y+PATCH_SIZE, x:x+PATCH_SIZE].mean()) < minimum:
                    continue
                if all(float(_roi_case(source, reference_valid, int(y), int(x), start, parity)[2].mean()) >= minimum for start in starts for parity in (0, 1)):
                    eligible.add((int(y), int(x)))
        rois = select_reference_rois(source["mean"], reference_valid, count=int(config.get("roi_count", 3)),
                                     grid_size=grid_size, min_valid_fraction=minimum, eligible_rois=eligible)
        flat, detail, region_policy = _region_masks(source["mean"], reference_valid, source["black"])
        source_record = {"name": name, "role": role, "kind": source["kind"], "rois": rois,
                         "reference_files": source["files"], "region_policy": region_policy,
                         "raw": {"path": str(source["raw_path"]), "size_bytes": source["raw_path"].stat().st_size,
                                 "sha256_reported": source["source_sha256_reported"],
                                 "identity_note": "Complete RAW was not rehashed. Existing source identity is recorded; exact selected raw, normalized input, target, and mask bytes are hashed per cached case."},
                         "brightness_limitation": source["brightness_limitation"]}
        result["sources"].append(source_record)
        source_region_pixels = {"valid": 0, "flat": 0, "detail": 0}
        for roi_index, roi in enumerate(rois):
            y, x, patch = roi["y"], roi["x"], PATCH_SIZE
            area = (slice(y, y + patch), slice(x, x + patch))
            for start in starts:
                for parity in (0, 1):
                    noisy, target_frame, valid, indices = _roi_case(source, reference_valid, y, x, start, parity)
                    target_half = "odd" if parity == 0 else "even"
                    if float(valid.mean()) < minimum:
                        raise ValueError(f"{name} ROI {roi_index} start {start} parity {parity} has {valid.mean():.3f} valid pixels; frozen reference-only ROI cannot be replaced based on model results")
                    target = np.broadcast_to(target_frame, noisy.shape).copy()
                    input_normalized = normalise_native_dn(noisy, source["black"])
                    arrays = {"noisy_dn": noisy, "input_normalized": input_normalized, "target_dn": target,
                              "valid": np.broadcast_to(valid, (FRAMES, patch, patch)).copy(),
                              "flat": np.broadcast_to(valid & flat[area], (FRAMES, patch, patch)).copy(),
                              "detail": np.broadcast_to(valid & detail[area], (FRAMES, patch, patch)).copy(),
                              "black_dn": source["black"]}
                    if not all(np.isfinite(array).all() for array in arrays.values()):
                        raise ValueError("prepared static case contains non-finite data")
                    for region in source_region_pixels:
                        source_region_pixels[region] += int(arrays[region].sum())
                    case_name = f"{name}_roi{roi_index}_f{start:06d}_p{parity}"
                    path = prepared_dir / f"{case_name}.npz"
                    with path.open("xb") as stream:
                        np.savez(stream, **arrays)
                    record = {"name": case_name, "source": name, "role": role, "file": path.name,
                              "sha256": sha256_file(path), "array_sha256": {key: _array_identity(value) for key, value in arrays.items()},
                              "roi": roi, "frame_indices": indices.tolist(), "input_parity": parity,
                              "target_parity": 1 - parity, "target_reference": source["files"][target_half],
                              "target_frame_pool": list(range(1 - parity, source["frames_used"], 2)),
                              "valid_fraction": float(valid.mean()), "temporal_stride_original_frames": 2,
                              "input_metrics": {"raw_input": score_static_prediction(noisy, target, arrays["valid"], arrays["flat"], arrays["detail"]),
                                                "model_input": score_static_prediction(denormalise_native(input_normalized, source["black"]), target, arrays["valid"], arrays["flat"], arrays["detail"])}}
                    result["cases"].append(record)
                    print(f"[a14-prepare] {case_name} valid={valid.mean():.3f}", flush=True)
        if not all(source_region_pixels.values()):
            raise ValueError(f"{name} frozen ROIs lack source-level valid, flat, or detail pixels")
        source_record["region_pixels"] = source_region_pixels
    _write_json(prepared_dir / "manifest.json", result)
    return result


def _finalize_statistics(stats: dict) -> dict:
    count = stats["count"]
    if count == 0:
        return {"pixels": 0, "statistics": stats}
    rmse = float(np.sqrt(stats["squared_error_sum"] / count))
    mean = stats["error_sum"] / count
    return {
        "pixels": count // 4, "rmse_dn": rmse,
        "psnr_db": float(20 * np.log10(4095.0 / max(rmse, 1e-12))),
        "bias_dn": mean,
        "residual_sigma_dn": float(np.sqrt(max(0, stats["squared_error_sum"] / count - mean**2))),
        "highpass_rmse_dn": float(np.sqrt(stats["highpass_squared_sum"] / stats["highpass_count"])) if stats["highpass_count"] else None,
        "gradient_rmse_dn": float(np.sqrt(stats["gradient_squared_sum"] / stats["gradient_count"])) if stats["gradient_count"] else None,
        "temporal_difference_error_rmse_dn": float(np.sqrt(stats["temporal_error_squared_sum"] / stats["temporal_count"])) if stats["temporal_count"] else None,
        "temporal_mad_dn": stats["temporal_abs_sum"] / stats["temporal_count"] if stats["temporal_count"] else None,
        "statistics": stats,
    }


def score_static_prediction(
    prediction_dn: np.ndarray, target_dn: np.ndarray, valid: np.ndarray,
    flat: np.ndarray, detail: np.ndarray,
) -> dict:
    """Measure reconstruction, structure, noise, and temporal errors on frozen masks."""

    prediction, target = np.asarray(prediction_dn, dtype=np.float64), np.asarray(target_dn, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError("prediction and target must have matching [T,4,H,W] shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    expected_mask = (prediction.shape[0], *prediction.shape[2:])
    if any(mask.shape != expected_mask or mask.dtype != bool for mask in (valid, flat, detail)):
        raise ValueError("region masks must be boolean [T,H,W] arrays")
    if np.any(flat & ~valid) or np.any(detail & ~valid) or not valid.any():
        raise ValueError("flat/detail must be subsets of a nonempty valid mask")
    residual = prediction - target
    hp_residual = _spatial_highpass(residual)
    results = {}
    for name, mask in (("all", valid), ("flat", flat), ("detail", detail)):
        selected = np.broadcast_to(mask[:, None], residual.shape)
        hp_mask = np.broadcast_to((_erode_masks(valid) & mask)[:, None], residual.shape)
        temporal_mask = np.broadcast_to((mask[1:] & mask[:-1])[:, None], residual[1:].shape)
        values = residual[selected]
        stats = {"count": int(values.size), "error_sum": float(values.sum()),
                 "squared_error_sum": float(np.square(values).sum()),
                 "highpass_squared_sum": float(np.square(hp_residual[hp_mask]).sum()),
                 "highpass_count": int(hp_mask.sum()), "gradient_squared_sum": 0.0, "gradient_count": 0,
                 "temporal_error_squared_sum": float(np.square(np.diff(residual, axis=0)[temporal_mask]).sum()),
                 "temporal_abs_sum": float(np.abs(np.diff(prediction, axis=0)[temporal_mask]).sum()),
                 "temporal_count": int(temporal_mask.sum())}
        for axis in (-2, -1):
            gradient = np.diff(residual, axis=axis)
            if axis == -2:
                gradient_mask = mask[:, 1:] & mask[:, :-1]
            else:
                gradient_mask = mask[:, :, 1:] & mask[:, :, :-1]
            selection = np.broadcast_to(gradient_mask[:, None], gradient.shape)
            stats["gradient_squared_sum"] += float(np.square(gradient[selection]).sum())
            stats["gradient_count"] += int(selection.sum())
        results[name] = _finalize_statistics(stats)
    return results


def summarize_cases(cases: list[dict]) -> dict:
    """Pool squared errors within each source, preserving its evaluation role."""

    sources = {}
    for case in cases:
        entry = sources.setdefault(case["source"], {"role": case["role"], "case_count": 0, "metrics": {}})
        if entry["role"] != case["role"]:
            raise ValueError("a source cannot have conflicting evaluation roles")
        entry["case_count"] += 1
        for variant, regions in case["metrics"].items():
            for region, metrics in regions.items():
                total = entry["metrics"].setdefault(variant, {}).setdefault(region, {})
                for key, value in metrics["statistics"].items():
                    total[key] = total.get(key, 0) + value
    for source in sources.values():
        source["metrics"] = {variant: {region: _finalize_statistics(stats) for region, stats in regions.items()}
                             for variant, regions in source["metrics"].items()}
    return sources


def evaluate_static_checkpoint(
    prepared_dir: Path, checkpoint: Path, weights: str, label: str, output_dir: Path,
    device_name: str, expected_checkpoint_sha256: str | None = None,
    expected_gpu_uuid: str | None = None,
) -> dict:
    """Evaluate exactly one state once; any OOM propagates without retry or substitution."""

    if weights not in {"model", "ema"} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
        raise ValueError("invalid checkpoint state or evaluation label")
    prepared_dir, checkpoint, output_dir = _path(prepared_dir), _path(checkpoint), _output_path(output_dir)
    manifest_path = prepared_dir / "manifest.json"
    manifest_identity, checkpoint_identity = _identity(manifest_path), _identity(checkpoint)
    if expected_checkpoint_sha256 is not None and checkpoint_identity["sha256"] != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match requested identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or not manifest["cases"]:
        raise ValueError("prepared manifest has unsupported schema or no cases")
    for case in manifest["cases"]:
        if Path(case["file"]).name != case["file"] or sha256_file(prepared_dir / case["file"]) != case["sha256"]:
            raise ValueError("prepared case hash mismatch or invalid case path")
    device = torch.device(device_name)
    gpu_uuid = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; device substitution is forbidden")
        gpu_uuid = f"GPU-{UUID(str(torch.cuda.get_device_properties(device).uuid))}"
        if expected_gpu_uuid is not None and gpu_uuid != expected_gpu_uuid:
            raise ValueError(f"GPU UUID {gpu_uuid} differs from requested {expected_gpu_uuid}")
    elif device.type != "cpu" or expected_gpu_uuid is not None:
        raise ValueError("CPU evaluation cannot meet a requested GPU UUID")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = {"schema": SCHEMA, "label": label, "weights": weights,
                "checkpoint": checkpoint_identity, "prepared_manifest": manifest_identity,
                "device": device_name, "gpu_uuid": gpu_uuid,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "precision": "float32", "tf32": False, "output_clip": [0.0, 1.0],
                "code": [_identity(Path(__file__)), _identity(Path(__file__).with_name("a12_paired_eval.py"))],
                "scope": manifest["scope"], "source_roles": {source["name"]: source["role"] for source in manifest["sources"]}}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "identity.json", identity)
    status_path = output_dir / "status.json"
    status = {"status": "running", "completed_cases": 0, "oom_policy": "stop_without_retry"}
    _write_json(status_path, status)
    started = time.monotonic()
    result = {"identity": identity, "metric_protocol": "Native DN error against opposite-parity means; PSNR peak=4095; HP uses 3x3 box highpass and eroded valid support; gradients require both pixels valid; temporal stride is two original frames; flat residual sigma is a noise/detail diagnostic, not standalone quality proof.", "cases": []}
    try:
        model = load_selected_model(checkpoint, weights, device)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("model must have FP32 parameters")
        for case in manifest["cases"]:
            print(f"[a14-eval] {label} {case['name']}", flush=True)
            path = prepared_dir / case["file"]
            if sha256_file(path) != case["sha256"]:
                raise ValueError("prepared case changed during evaluation")
            with np.load(path, allow_pickle=False) as data:
                inputs = np.array(data["input_normalized"], dtype=np.float32)
                if inputs.shape != (FRAMES, 4, PATCH_SIZE, PATCH_SIZE) or not np.isfinite(inputs).all():
                    raise ValueError("prepared input must be a finite [8,4,64,64] array")
                with torch.inference_mode():
                    output = model(torch.from_numpy(inputs[None]).to(device))
                if output.shape != (1, *inputs.shape) or output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
                    raise ValueError("prediction must be finite FP32 and match the exact input shape")
                outside_fraction = float(((output < 0) | (output > 1)).float().mean().item())
                prediction_normalized = output.clamp(0.0, 1.0)[0].cpu().numpy()
                prediction = denormalise_native(prediction_normalized, data["black_dn"])
                metrics = {**case["input_metrics"], "prediction": score_static_prediction(prediction, data["target_dn"], data["valid"], data["flat"], data["detail"])}
            prediction_path = output_dir / f"{case['name']}_prediction_dn.npy"
            with prediction_path.open("xb") as stream:
                np.save(stream, prediction, allow_pickle=False)
            record = {"name": case["name"], "source": case["source"], "role": case["role"],
                      "prepared_case_sha256": case["sha256"], "prediction": _identity(prediction_path),
                      "unclipped_normalized_outside_fraction": outside_fraction, "metrics": metrics}
            _write_json(output_dir / f"{case['name']}_metrics.json", record)
            result["cases"].append(record)
            status["completed_cases"] += 1
        if sha256_file(checkpoint) != checkpoint_identity["sha256"] or sha256_file(manifest_path) != manifest_identity["sha256"]:
            raise ValueError("checkpoint or prepared manifest changed during evaluation")
        result["summary_by_source"] = summarize_cases(result["cases"])
        _write_json(output_dir / "evaluation.json", result)
        status.update(status="evaluation_completed", evaluation=str(output_dir / "evaluation.json"))
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        status.update(status="stopped_oom" if is_oom else "stopped_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result

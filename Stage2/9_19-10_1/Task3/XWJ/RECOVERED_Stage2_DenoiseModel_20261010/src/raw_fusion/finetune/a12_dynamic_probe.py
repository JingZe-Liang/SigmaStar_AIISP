"""Controlled packed-RAW motion probes with known constructed frame targets."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from raw_fusion.finetune.noise import generate_noisy_frame


CHANNELS = ("R", "Gr", "Gb", "B")


@dataclass(frozen=True, slots=True)
class MotionProbe:
    target_dn: np.ndarray
    valid: np.ndarray
    motion: np.ndarray
    static: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_shift(array: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    height, width = array.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("translation must retain some source pixels")
    result = np.full_like(array, fill)
    src_y = slice(max(0, -dy), min(height, height - dy))
    src_x = slice(max(0, -dx), min(width, width - dx))
    dst_y = slice(max(0, dy), min(height, height + dy))
    dst_x = slice(max(0, dx), min(width, width + dx))
    result[..., dst_y, dst_x] = array[..., src_y, src_x]
    return result


def make_motion_probe(
    background: np.ndarray,
    background_valid: np.ndarray,
    *,
    mode: str,
    frames: int = 8,
    velocity_yx: tuple[int, int] = (0, 2),
    foreground: np.ndarray | None = None,
    foreground_valid: np.ndarray | None = None,
    origin_yx: tuple[int, int] = (16, 8),
) -> MotionProbe:
    """Move same-colour packed planes without interpolation or circular wrapping."""
    if background.ndim != 3 or background.shape[0] != 4:
        raise ValueError("background must have shape [4,H,W]")
    if background_valid.shape != background.shape[-2:] or frames < 2:
        raise ValueError("invalid spatial mask or frame count")
    if any(not isinstance(value, int) for value in (*velocity_yx, *origin_yx)):
        raise ValueError("motion must use integer packed-pixel coordinates")
    target = np.repeat(background[None].astype(np.float32), frames, axis=0)
    valid = np.repeat(background_valid[None].astype(bool), frames, axis=0)
    swept = np.zeros(background_valid.shape, dtype=bool)
    dy, dx = velocity_yx

    if mode == "static":
        pass
    elif mode == "global_translation":
        if dy == 0 and dx == 0:
            raise ValueError("translation mode needs nonzero velocity")
        for frame in range(frames):
            target[frame] = _integer_shift(background, frame * dy, frame * dx)
            valid[frame] = _integer_shift(background_valid, frame * dy, frame * dx).astype(bool)
        swept[:] = True
    elif mode == "moving_patch":
        if foreground is None or foreground_valid is None:
            raise ValueError("moving_patch needs foreground and its validity mask")
        if foreground.ndim != 3 or foreground.shape[0] != 4 or foreground_valid.shape != foreground.shape[-2:]:
            raise ValueError("invalid foreground shapes")
        if dy == 0 and dx == 0:
            raise ValueError("moving_patch needs nonzero velocity")
        height, width = background.shape[-2:]
        obj_h, obj_w = foreground.shape[-2:]
        for frame in range(frames):
            y, x = origin_yx[0] + frame * dy, origin_yx[1] + frame * dx
            if y < 0 or x < 0 or y + obj_h > height or x + obj_w > width:
                raise ValueError("moving foreground must remain inside the patch")
            target[frame, :, y : y + obj_h, x : x + obj_w] = foreground
            valid[frame, y : y + obj_h, x : x + obj_w] = foreground_valid
            swept[y : y + obj_h, x : x + obj_w] = True
    else:
        raise ValueError(f"unknown motion mode: {mode}")

    motion = np.broadcast_to(swept, valid.shape).copy() & valid
    static = ~np.broadcast_to(swept, valid.shape) & valid
    return MotionProbe(target, valid, motion, static)


def add_measured_noise(
    target_dn: np.ndarray,
    noise_model: dict,
    gain: str,
    black: np.ndarray,
    fpn_dn: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Keep FPN fixed in sensor coordinates while independently sampling each frame."""
    if target_dn.ndim != 4 or target_dn.shape[1] != 4:
        raise ValueError("target must have shape [T,4,H,W]")
    if fpn_dn.shape != target_dn.shape[1:] or black.shape != (4,):
        raise ValueError("invalid FPN or black-level shape")
    black_dict = {channel: float(value) for channel, value in zip(CHANNELS, black, strict=True)}
    return np.stack([
        generate_noisy_frame(
            target, noise_model, gain, black_dict, fpn_dn,
            np.random.default_rng(np.random.SeedSequence([seed, frame])),
        )
        for frame, target in enumerate(target_dn)
    ])


def _spatial_highpass(video: np.ndarray) -> np.ndarray:
    return np.stack([
        np.stack([channel - cv2.blur(channel, (3, 3), borderType=cv2.BORDER_REFLECT_101) for channel in frame])
        for frame in video
    ])


def _erode_masks(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return np.stack([
        cv2.erode(frame.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        for frame in mask
    ])


def preprocess_model_input(noisy_dn: np.ndarray, black_dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact clipped model input and its reconstruction in native DN."""
    black = np.asarray(black_dn, dtype=np.float32)
    if black.shape != (4,) or np.any(black >= 4095.0):
        raise ValueError("black levels must have four channels below the white level")
    black = black[None, :, None, None]
    normalized = np.clip((noisy_dn.astype(np.float32) - black) / (4095.0 - black), 0.0, 1.0)
    return normalized, normalized * (4095.0 - black) + black


def measure_regions(prediction_dn: np.ndarray, probe: MotionProbe, *, black_dn: np.ndarray) -> dict:
    """Measure target-relative errors, never equating high-frequency energy with fidelity."""
    target = probe.target_dn.astype(np.float64)
    prediction = np.asarray(prediction_dn, dtype=np.float64)
    if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction shape must match target and values must be finite")
    black = np.asarray(black_dn, dtype=np.float64)
    if black.shape != (4,) or np.any(black >= 4095.0):
        raise ValueError("black levels must have four channels below the white level")
    normalized = (prediction - black[None, :, None, None]) / (4095.0 - black[None, :, None, None])
    residual = prediction - target
    hp_residual = _spatial_highpass(prediction) - _spatial_highpass(target)
    result: dict = {}
    for name, mask in (("all", probe.valid), ("motion_swept", probe.motion), ("static", probe.static)):
        pixels = int(mask.sum())
        if pixels == 0:
            result[name] = None
            continue
        selected = np.broadcast_to(mask[:, None], residual.shape)
        hp_mask = _erode_masks(mask)
        hp_selected = np.broadcast_to(hp_mask[:, None], residual.shape)
        pair_mask = mask[1:] & mask[:-1]
        pair_selected = np.broadcast_to(pair_mask[:, None], residual[1:].shape)
        gradient_error_sum = 0.0
        gradient_count = 0
        for axis in (-2, -1):
            grad_error = np.diff(prediction, axis=axis) - np.diff(target, axis=axis)
            if axis == -2:
                grad_mask = mask[:, 1:, :] & mask[:, :-1, :]
            else:
                grad_mask = mask[:, :, 1:] & mask[:, :, :-1]
            grad_selected = np.broadcast_to(grad_mask[:, None], grad_error.shape)
            gradient_error_sum += float(np.square(grad_error[grad_selected]).sum())
            gradient_count += int(grad_selected.sum())
        result[name] = {
            "pixels": pixels,
            "rmse_dn": float(np.sqrt(np.mean(np.square(residual[selected])))),
            "highpass_rmse_dn": float(np.sqrt(np.mean(np.square(hp_residual[hp_selected])))) if hp_selected.any() else None,
            "highpass_pixels": int(hp_mask.sum()),
            "gradient_rmse_dn": float(np.sqrt(gradient_error_sum / gradient_count)) if gradient_count else None,
            "temporal_difference_error_rmse_dn": float(np.sqrt(np.mean(np.square(np.diff(residual, axis=0)[pair_selected])))) if pair_selected.any() else None,
            "channel_bias_dn": {channel: float(residual[:, index][mask].mean()) for index, channel in enumerate(CHANNELS)},
            "storage_code_endpoints": {
                "zero_fraction": float(np.mean(prediction[selected] == 0.0)),
                "white_fraction": float(np.mean(prediction[selected] == 4095.0)),
                "outside_12bit_range_fraction": float(np.mean((prediction[selected] < 0.0) | (prediction[selected] > 4095.0))),
            },
            "normalized_domain_limits": {
                "black_endpoint_fraction": float(np.mean(np.abs(normalized[selected]) <= 1e-7)),
                "white_endpoint_fraction": float(np.mean(np.abs(normalized[selected] - 1.0) <= 1e-7)),
                "below_black_fraction": float(np.mean(normalized[selected] < -1e-7)),
                "above_white_fraction": float(np.mean(normalized[selected] > 1.0 + 1e-7)),
            },
        }
    return result


def _score_variants(noisy: np.ndarray, black: np.ndarray, prediction: np.ndarray, probe: MotionProbe, name: str) -> dict:
    _, model_input = preprocess_model_input(noisy, black)
    raw_mean = np.broadcast_to(noisy.mean(axis=0, keepdims=True), noisy.shape)
    model_mean = np.broadcast_to(model_input.mean(axis=0, keepdims=True), noisy.shape)
    return {
        "raw_input": measure_regions(noisy, probe, black_dn=black),
        "model_input": measure_regions(model_input, probe, black_dn=black),
        "unaligned_temporal_mean_raw_input": measure_regions(raw_mean, probe, black_dn=black),
        "unaligned_temporal_mean_model_input": measure_regions(model_mean, probe, black_dn=black),
        name: measure_regions(prediction, probe, black_dn=black),
    }


def _score_protocol() -> dict:
    return {
        "version": "a12_region_errors_v2",
        "raw_input": "stored noisy RAW before black-level clipping",
        "model_input": "the same input after channel-wise black subtraction, [0,1] clipping and DN reconstruction",
        "storage_code_endpoints": "exact storage codes 0 and 4095; distinct from black-level normalization endpoints",
        "normalized_domain_limits": "channel-wise (DN-black)/(4095-black); endpoint tolerance 1e-7; raw below-black samples are not claimed to be hardware clipping",
        "acceptance": "report separate reconstruction, structural and temporal target-relative errors; no single-metric pass claim",
    }


def prepare_probes(config: dict, repo_root: Path, output_dir: Path) -> dict:
    """Freeze constructed inputs, targets and provenance in a new directory."""
    source = config["source"]
    paths = {name: repo_root / source[name] for name in ("reference", "valid_mask", "fpn", "reference_report", "noise_model")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = np.load(paths["reference"], allow_pickle=False)
    valid = np.load(paths["valid_mask"], allow_pickle=False).astype(bool)
    fpn = np.load(paths["fpn"], allow_pickle=False)
    report = json.loads(paths["reference_report"].read_text(encoding="utf-8"))
    noise_model = json.loads(paths["noise_model"].read_text(encoding="utf-8"))
    black = np.asarray([report["black_levels"][name] for name in CHANNELS], dtype=np.float32)
    valid &= np.all((reference >= black[:, None, None]) & (reference < 3800.0), axis=0)
    if reference.shape[0] != 4 or fpn.shape != reference.shape or valid.shape != reference.shape[-2:]:
        raise ValueError("reference, FPN and validity shapes are inconsistent")
    patch = int(config["patch_size"])
    obj_size = int(config["foreground"]["size"])
    obj_y, obj_x = config["foreground"]["source_yx"]
    foreground = reference[:, obj_y : obj_y + obj_size, obj_x : obj_x + obj_size]
    fg_valid = valid[obj_y : obj_y + obj_size, obj_x : obj_x + obj_size]
    if foreground.shape != (4, obj_size, obj_size):
        raise ValueError("foreground source is outside reference")
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for crop_index, crop in enumerate(config["crops"]):
        y, x = crop["source_yx"]
        background = reference[:, y : y + patch, x : x + patch]
        background_valid = valid[y : y + patch, x : x + patch]
        crop_fpn = fpn[:, y : y + patch, x : x + patch]
        if background.shape != (4, patch, patch):
            raise ValueError("background source is outside reference")
        for mode in config["modes"]:
            probe = make_motion_probe(
                background, background_valid, mode=mode, frames=int(config["frames"]),
                velocity_yx=tuple(config["velocity_yx"]), foreground=foreground,
                foreground_valid=fg_valid, origin_yx=tuple(config["foreground"]["origin_yx"]),
            )
            for gain_index, gain in enumerate(config["gains"]):
                seed = int(config["seed"]) + crop_index * 1009 + gain_index * 104729
                noisy = add_measured_noise(probe.target_dn, noise_model, gain, black, crop_fpn, seed)
                name = f"{crop['name']}__{mode}__gain_{gain}"
                path = output_dir / f"{name}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, noisy_dn=noisy, target_dn=probe.target_dn,
                                        valid=probe.valid, motion=probe.motion, static=probe.static, black_dn=black)
                records.append({
                    "name": name, "file": path.name, "sha256": sha256_file(path), "seed": seed,
                    "crop": crop, "mode": mode, "gain": gain,
                    "valid_fraction": float(probe.valid.mean()),
                    "input_metrics": measure_regions(noisy, probe, black_dn=black),
                })
    manifest = {
        "protocol": config["protocol"], "scope": config["scope"], "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "sources": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "motion_mask_definition": "swept trajectory union, including vacated background; global translation uses all valid pixels",
        "highpass_mask_definition": "3x3 spatial erosion of region and source validity; image boundary excluded",
        "claim": "constructed-target mechanism diagnostic; source scene was used by A3; no independent dynamic or cross-sensor generalization claim",
        "cases": records,
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


def evaluate_probes(
    prepared_dir: Path, output_dir: Path, repo_root: Path, device_name: str,
    baseline_config: dict | None = None,
) -> dict:
    """Evaluate a hash-pinned model/EMA state or official CRVD module."""
    import torch
    from raw_fusion.finetune.a3 import build_loaded_model, build_rvideformer
    from raw_fusion.finetune.a4 import infer_video

    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("prepared configuration hash does not match")
    if baseline_config is None:
        baseline = {"name": "a3_ema", "format": "ema", "checkpoint": config["checkpoint"],
                    "checkpoint_sha256": config["checkpoint_sha256"]}
    else:
        baseline = baseline_config
    if baseline["format"] not in {"model", "ema", "official_serialized_module"}:
        raise ValueError("unknown baseline checkpoint format")
    checkpoint = repo_root / baseline["checkpoint"]
    if sha256_file(checkpoint) != baseline["checkpoint_sha256"]:
        raise ValueError("baseline checkpoint SHA-256 mismatch")
    for record in manifest["cases"]:
        if sha256_file(prepared_dir / record["file"]) != record["sha256"]:
            raise ValueError(f"prepared case SHA-256 mismatch: {record['name']}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; select a verified device explicitly")
    if baseline["format"] in {"model", "ema"}:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_key = baseline["format"]
        if state_key not in payload:
            raise ValueError(f"fixed baseline must contain {state_key} state")
        model = build_rvideformer()
        model.load_state_dict(payload[state_key], strict=True)
        model = model.to(device).eval()
    else:
        model = build_loaded_model(checkpoint, device).eval()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for record in manifest["cases"]:
        with np.load(prepared_dir / record["file"], allow_pickle=False) as item:
            noisy = item["noisy_dn"].astype(np.float32)
            black = item["black_dn"]
            probe = MotionProbe(item["target_dn"], item["valid"], item["motion"], item["static"])
            normalized, _ = preprocess_model_input(noisy, black)
            black_broadcast = black[None, :, None, None]
            prediction = infer_video(model, normalized, device, tile_batch=1) * (4095.0 - black_broadcast) + black_broadcast
            path = output_dir / f"{record['name']}__{baseline['name']}_dn.npy"
            with path.open("xb") as handle:
                np.save(handle, prediction, allow_pickle=False)
            records.append({
                "name": record["name"], "mode": record["mode"], "gain": record["gain"],
                "prediction": {"file": path.name, "sha256": sha256_file(path)},
                **_score_variants(noisy, black, prediction, probe, baseline["name"]),
            })
    result = {
        "protocol": config["protocol"], "scope": config["scope"],
        "prepared_manifest": str(manifest_path), "prepared_manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "weights": baseline["format"], "baseline": baseline,
        "metric_protocol": _score_protocol(), "device": device_name, "training_performed": False,
        "acceptance_status": "diagnostic_only_no_quality_pass_claim", "cases": records,
    }
    with (output_dir / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def rescore_cached_probes(prepared_dir: Path, source_dir: Path, output_dir: Path) -> dict:
    """Rescore immutable cached predictions on CPU, preserving original outputs."""
    source_path = source_dir / "evaluation.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest_path = prepared_dir / "manifest.json"
    if sha256_file(manifest_path) != source["prepared_manifest_sha256"]:
        raise ValueError("prepared manifest differs from the inference source")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_cases = {record["name"]: record for record in source["cases"]}
    if set(source_cases) != {record["name"] for record in manifest["cases"]}:
        raise ValueError("cached predictions do not cover exactly the frozen cases")
    if "baseline" in source:
        baseline = source["baseline"]
    else:
        if source["weights"] != "ema" or "a3_ema" not in source["cases"][0]:
            raise ValueError("legacy score source is not the expected A3 EMA baseline")
        baseline = {"name": "a3_ema", "format": "ema", "checkpoint": source["checkpoint"],
                    "checkpoint_sha256": source["checkpoint_sha256"]}
    records = []
    for record in manifest["cases"]:
        prepared_path = prepared_dir / record["file"]
        if sha256_file(prepared_path) != record["sha256"]:
            raise ValueError(f"prepared case SHA-256 mismatch: {record['name']}")
        old_record = source_cases[record["name"]]
        prediction_path = source_dir / old_record["prediction"]["file"]
        if sha256_file(prediction_path) != old_record["prediction"]["sha256"]:
            raise ValueError(f"cached prediction SHA-256 mismatch: {record['name']}")
        with np.load(prepared_path, allow_pickle=False) as item:
            noisy = item["noisy_dn"].astype(np.float32)
            black = item["black_dn"]
            probe = MotionProbe(item["target_dn"], item["valid"], item["motion"], item["static"])
            prediction = np.load(prediction_path, allow_pickle=False)
            records.append({
                "name": record["name"], "mode": record["mode"], "gain": record["gain"],
                "prediction": {"file": str(prediction_path.resolve()), "sha256": old_record["prediction"]["sha256"]},
                **_score_variants(noisy, black, prediction, probe, baseline["name"]),
            })
    result = {
        "protocol": manifest["config"]["protocol"], "scope": manifest["config"]["scope"],
        "prepared_manifest": str(manifest_path), "prepared_manifest_sha256": sha256_file(manifest_path),
        "source_evaluation": str(source_path), "source_evaluation_sha256": sha256_file(source_path),
        "checkpoint": source["checkpoint"], "checkpoint_sha256": source["checkpoint_sha256"],
        "baseline": baseline, "weights": source["weights"], "metric_protocol": _score_protocol(),
        "operation": "CPU metric rescore of existing predictions; no new inference",
        "training_performed": False, "acceptance_status": "diagnostic_only_no_quality_pass_claim", "cases": records,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result

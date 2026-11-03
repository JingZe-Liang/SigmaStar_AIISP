"""Fixed probes and independent native windows for the paired A12 experiment."""

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Literal

import numpy as np
import torch
from torch import nn

from raw_fusion.finetune.a3 import _channel_vector, _normalise_dn, build_rvideformer
from raw_fusion.finetune.a4 import infer_video
from raw_fusion.finetune.a5_native import _load_native_reference, evaluate_native_outputs
from raw_fusion.finetune.a12_dynamic_probe import evaluate_probes
from raw_fusion.finetune.a12_train import sha256_file
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.stream_io import open_stream, pack_frame


REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_STARTS = (0, 96, 192)
NATIVE_FRAMES = 8
SUPERVISION_DIR = REPO_ROOT / "artifacts/finetune_mis20s1/supervision"
REFERENCE_DIR = REPO_ROOT / "artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe"
PREPARED_DIR = REPO_ROOT / "artifacts/a12_dynamic_probe_20261029/prepared"


def read_native_window(
    supervision_dir: Path, start_frame: int, frames: int = NATIVE_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read exact original frame indices; no padding, concatenation, or alignment."""

    if start_frame < 0 or frames <= 0:
        raise ValueError("native window requires a nonnegative start and positive frame count")
    manifest_path = supervision_dir / "stream_manifests/04_shdarkroom_1000x.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    end_frame = start_frame + frames
    available = min(int(reader.frames), int(manifest["frames_used"]))
    if end_frame > available:
        raise ValueError(f"native window [{start_frame}, {end_frame}) exceeds {available} frames")
    black = _channel_vector(manifest["black_levels"])
    video = np.stack([
        _normalise_dn(pack_frame(reader.read_frame(index), adapter).astype(np.float32), black)
        for index in range(start_frame, end_frame)
    ])
    mask_path = REPO_ROOT / manifest["motion"]["mask_path"]
    full_motion = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if full_motion.ndim != 4 or full_motion.shape[1] != 1 or full_motion.shape[0] < end_frame:
        raise ValueError("motion mask does not cover the requested native window")
    motion = full_motion[start_frame:end_frame, 0, ::2, ::2].astype(np.float32, copy=False)
    if video.shape[1] != 4 or motion.shape != (frames, *video.shape[2:]):
        raise ValueError("native video and motion mask shapes do not match")
    if not np.isfinite(video).all() or not np.isfinite(motion).all():
        raise ValueError("native window contains non-finite values")
    return video, motion, black


def load_selected_model(checkpoint: Path, weights: str, device: torch.device) -> nn.Module:
    """Select the exact requested state, without model/EMA substitution."""

    if weights not in {"model", "ema"}:
        raise ValueError("weights must be model or ema")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or weights not in payload:
        raise ValueError(f"checkpoint lacks requested {weights} state")
    if not all(bool(torch.isfinite(value).all()) for value in payload[weights].values()):
        raise ValueError(f"checkpoint {weights} state contains non-finite values")
    model = build_rvideformer()
    model.load_state_dict(payload[weights], strict=True)
    return model.to(device).eval()


def evaluate_native_windows(
    checkpoint: Path, weights: str, label: str, output_dir: Path, device_name: str,
    checkpoint_sha256: str,
) -> dict:
    """Score three disjoint native clips separately against the existing reference."""

    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 changed before native evaluation")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = SUPERVISION_DIR / "stream_manifests/04_shdarkroom_1000x.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mask_path = REPO_ROOT / manifest["motion"]["mask_path"]
    raw_path = Path(manifest["path"])
    reference, valid = _load_native_reference(REFERENCE_DIR)
    device = torch.device(device_name)
    model = load_selected_model(checkpoint, weights, device)
    result = {
        "protocol": "A12 paired native1000x three independent eight-frame windows v1",
        "scope": "Development same-scene gain holdout; native reference includes evaluated frames; not independent clean truth.",
        "label": label, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "weights": weights, "device": device_name, "prediction_dtype": "float32",
        "tile_batch": 1, "spatial_tile": 64, "spatial_stride": 56, "temporal_frames": NATIVE_FRAMES,
        "temporal_policy": "Each window is inferred independently; no joining sparse clips.",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "raw": {"path": str(raw_path), "size_bytes": raw_path.stat().st_size,
                "cfa": manifest["cfa"], "bit_shift": manifest["bit_shift"],
                "identity_note": "Selected normalized input bytes are hashed per window; complete RAW is not rehashed."},
        "motion_mask": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
        "reference": {"path": str(REFERENCE_DIR / "temporal_mean_native_dn.npy"),
                      "sha256": sha256_file(REFERENCE_DIR / "temporal_mean_native_dn.npy")},
        "valid_region": {"path": str(REFERENCE_DIR / "valid_region.npy"),
                         "sha256": sha256_file(REFERENCE_DIR / "valid_region.npy")},
        "training_performed": False, "acceptance_status": "diagnostic_only_no_quality_pass_claim",
        "windows": [],
    }
    for start in NATIVE_STARTS:
        video, motion, black = read_native_window(SUPERVISION_DIR, start)
        print(f"[a12-eval] {label} native frames {start}..{start + NATIVE_FRAMES - 1}", flush=True)
        prediction = infer_video(model, video, device, tile_batch=1)
        if prediction.dtype != np.float32 or prediction.shape != video.shape or not np.isfinite(prediction).all():
            raise ValueError("native prediction is not a finite FP32 array matching the input")
        prediction_path = output_dir / f"frames_{start:06d}_{start + NATIVE_FRAMES - 1:06d}.npy"
        with prediction_path.open("xb") as stream:
            np.save(stream, prediction, allow_pickle=False)
        record = {
            "frame_indices": list(range(start, start + NATIVE_FRAMES)), "input_shape": list(video.shape),
            "input_sha256": hashlib.sha256(video.tobytes()).hexdigest(),
            "motion_window_sha256": hashlib.sha256(motion.tobytes()).hexdigest(), "black_dn": black.tolist(),
            "prediction": {"file": prediction_path.name, "sha256": sha256_file(prediction_path)},
            "metrics": evaluate_native_outputs(prediction, video, reference, valid, motion, black),
        }
        with (output_dir / f"frames_{start:06d}_metrics.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        result["windows"].append(record)
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 changed during native evaluation")
    with (output_dir / "evaluation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def evaluate_paired_checkpoint(
    checkpoint: Path, weights: Literal["model", "ema"], label: str, output_dir: Path,
    device_name: str, phase: Literal["probes", "native", "all"] = "all",
    expected_checkpoint_sha256: str | None = None,
) -> dict:
    """Run requested evaluations once, preserving prior phases and all old outputs."""

    checkpoint, output_dir = checkpoint.resolve(), output_dir.resolve()
    if weights not in {"model", "ema"} or phase not in {"probes", "native", "all"}:
        raise ValueError("unsupported weights or evaluation phase")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
        raise ValueError("label must contain only letters, digits, underscores, or hyphens")
    if output_dir.parent != (REPO_ROOT / "artifacts").resolve() or not output_dir.name.startswith("a12_"):
        raise ValueError("evaluation output must be a repository artifacts/a12_* directory")
    checkpoint_sha256 = sha256_file(checkpoint)
    if expected_checkpoint_sha256 is not None and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match the requested identity")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no automatic device substitution is allowed")
    identity = {
        "schema": "a12_paired_evaluation_v1", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256, "weights": weights, "label": label,
        "device": device_name, "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "prepared_manifest_sha256": sha256_file(PREPARED_DIR / "manifest.json"),
        "native_starts": list(NATIVE_STARTS), "native_frames": NATIVE_FRAMES, "tile_batch": 1,
    }
    phases = ("probes", "native") if phase == "all" else (phase,)
    identity_path = output_dir / "identity.json"
    if output_dir.exists():
        if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("existing evaluation directory has a different or missing identity")
    else:
        output_dir.mkdir(parents=True)
        with identity_path.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, indent=2)
    if any((output_dir / item).exists() for item in phases):
        raise FileExistsError("requested evaluation phase already has output; use a new directory")
    status_path = output_dir / f"status_{phase}.json"
    status = {"status": "running", "phase": phase, "oom_policy": "stop_without_retry", "outputs": {}}
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
    started = time.monotonic()
    try:
        if "probes" in phases:
            baseline = {"name": label, "format": weights, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_sha256}
            evaluate_probes(PREPARED_DIR, output_dir / "probes", REPO_ROOT, device_name, baseline)
            status["outputs"]["probes"] = str(output_dir / "probes/evaluation.json")
        if "native" in phases:
            evaluate_native_windows(checkpoint, weights, label, output_dir / "native", device_name, checkpoint_sha256)
            status["outputs"]["native"] = str(output_dir / "native/evaluation.json")
        status["status"] = "evaluation_completed"
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        status.update(status="stopped_oom" if is_oom else "stopped_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status

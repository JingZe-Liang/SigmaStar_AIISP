"""A4 leave-one-gain-out evaluation for the Phase A RViDeformer model."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from raw_fusion.finetune.a3 import (
    _CHANNEL_NAMES,
    _PACKED_HEIGHT,
    _PACKED_WIDTH,
    _WHITE_LEVEL,
    _channel_vector,
    _load_json,
    _normalise_dn,
    build_rvideformer,
)
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.stream_io import open_stream, pack_frame


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.no_grad()
def infer_video(model: torch.nn.Module, video: np.ndarray, device: torch.device, tile_batch: int = 4) -> np.ndarray:
    """Run overlapping 8-frame/64-pixel windows and blend their outputs."""

    if video.ndim != 4 or video.shape[1:] != (4, _PACKED_HEIGHT, _PACKED_WIDTH):
        raise ValueError(f"expected [T,4,540,960], got {video.shape}")
    temporal = 8
    out = np.zeros_like(video, dtype=np.float32)
    weights = np.zeros((video.shape[0], 1, video.shape[2], video.shape[3]), dtype=np.float32)
    time_starts = _starts(video.shape[0], temporal, 4)
    y_starts = _starts(video.shape[2], 64, 56)
    x_starts = _starts(video.shape[3], 64, 56)
    model.eval()
    for t in time_starts:
        clip = video[t : t + temporal]
        if clip.shape[0] < temporal:
            clip = np.concatenate([clip, np.repeat(clip[-1:], temporal - clip.shape[0], axis=0)], axis=0)
        positions = [(y, x) for y in y_starts for x in x_starts]
        for start in range(0, len(positions), tile_batch):
            batch_positions = positions[start : start + tile_batch]
            patches = np.stack([clip[:, :, y : y + 64, x : x + 64] for y, x in batch_positions])
            prediction = model(torch.from_numpy(patches).to(device)).clamp(0.0, 1.0).detach().cpu().numpy()
            count = min(temporal, video.shape[0] - t)
            for patch_prediction, (y, x) in zip(prediction, batch_positions, strict=True):
                out[t : t + count, :, y : y + 64, x : x + 64] += patch_prediction[:count]
                weights[t : t + count, :, y : y + 64, x : x + 64] += 1.0
    return out / np.maximum(weights, 1e-6)


def load_checkpoint_model(checkpoint: Path, device: torch.device, ema: bool = True) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_rvideformer()
    state = payload["ema" if ema and "ema" in payload else "model"] if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_official_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from raw_fusion.finetune.a3 import build_loaded_model

    return build_loaded_model(checkpoint, device).eval()


def _read_aligned_video(
    supervision_dir: Path,
    gain: str,
    max_frames: int | None,
    *,
    start_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json"
    manifest = _load_json(manifest_path)
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    reference = np.load(supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy", mmap_mode="r")
    ref_report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    ref_black = _channel_vector(ref_report["black_levels"])
    fit = manifest["exposure_fit"]
    slopes = _channel_vector(fit["slopes"])
    intercepts = _channel_vector(fit["intercepts"])
    shift_x = int(manifest["registration"]["integer_shift_x"])
    shift_y = int(manifest["registration"]["integer_shift_y"])
    frames_used = int(manifest["frames_used"])
    if start_frame < 0 or start_frame >= frames_used:
        raise ValueError(
            f"start_frame must be in [0, {frames_used}), got {start_frame}"
        )
    available = frames_used - start_frame
    frames = available if max_frames is None else min(max_frames, available)
    if frames <= 0:
        raise ValueError(f"max_frames must select at least one frame, got {max_frames}")
    video = np.zeros((frames, 4, _PACKED_HEIGHT, _PACKED_WIDTH), dtype=np.float32)
    overlap = np.zeros((frames, 1, _PACKED_HEIGHT, _PACKED_WIDTH), dtype=bool)
    for index in range(frames):
        packed = pack_frame(reader.read_frame(start_frame + index), adapter).astype(np.float32)
        mapped = slopes[:, None, None] * packed + intercepts[:, None, None]
        normalized = _normalise_dn(mapped, ref_black)
        y0, y1 = max(0, -shift_y), min(_PACKED_HEIGHT, _PACKED_HEIGHT - shift_y)
        x0, x1 = max(0, -shift_x), min(_PACKED_WIDTH, _PACKED_WIDTH - shift_x)
        video[index, :, y0:y1, x0:x1] = normalized[:, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
        overlap[index, 0, y0:y1, x0:x1] = True
    return video, overlap


def _metric(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = (pred - target)[mask]
    if values.size == 0:
        return {"rmse_dn": float("nan"), "psnr_db": float("nan"), "pixels": 0}
    rmse = float(np.sqrt(np.mean(values * values)))
    psnr = float(20.0 * np.log10(_WHITE_LEVEL / max(rmse, 1e-8)))
    return {"rmse_dn": rmse, "psnr_db": psnr, "pixels": int(values.size)}


def evaluate_outputs(
    prediction: np.ndarray,
    input_video: np.ndarray,
    reference: np.ndarray,
    overlap: np.ndarray,
    motion_mask: np.ndarray,
    black: np.ndarray,
) -> dict:
    target = _normalise_dn(reference, black)
    static = overlap & (motion_mask[:, None] < 0.5)
    static &= np.all(input_video < ((3800.0 - black) / (_WHITE_LEVEL - black))[None, :, None, None], axis=1, keepdims=True)
    static = np.broadcast_to(static, prediction.shape)
    prediction_dn = prediction * (_WHITE_LEVEL - black[None, :, None, None]) + black[None, :, None, None]
    input_dn = input_video * (_WHITE_LEVEL - black[None, :, None, None]) + black[None, :, None, None]
    target_dn = reference[None].repeat(prediction.shape[0], axis=0).astype(np.float32, copy=False)
    result = {
        "finetuned": _metric(prediction_dn, target_dn, static),
        "input": _metric(input_dn, target_dn, static),
        "temporal_mad_dn": float(np.mean(np.abs(np.diff(prediction_dn, axis=0))[static[1:] & static[:-1]])) if prediction.shape[0] > 1 else float("nan"),
        "input_temporal_mad_dn": float(np.mean(np.abs(np.diff(input_dn, axis=0))[static[1:] & static[:-1]])) if prediction.shape[0] > 1 else float("nan"),
        "legal_fraction": float(np.mean((prediction_dn >= 0.0) & (prediction_dn <= _WHITE_LEVEL))),
    }
    # A1's sensor chart ROI is reported in sensor pixels; convert to packed coordinates.
    sy0, sy1, sx0, sx1 = (400, 500, 160, 290)
    # Per-frame/channel high-pass RMS, with an explicit ratio to the clean reference.
    ref_roi = target_dn[:, :, sy0:sy1, sx0:sx1]
    pred_roi = prediction_dn[:, :, sy0:sy1, sx0:sx1]
    ratios = []
    for channel in range(4):
        ref_hp = ref_roi[:, channel] - np.stack([cv2.blur(frame, (3, 3)) for frame in ref_roi[:, channel]])
        pred_hp = pred_roi[:, channel] - np.stack([cv2.blur(frame, (3, 3)) for frame in pred_roi[:, channel]])
        ref_rms = float(np.sqrt(np.mean(ref_hp * ref_hp)))
        pred_rms = float(np.sqrt(np.mean(pred_hp * pred_hp)))
        ratios.append(pred_rms / max(ref_rms, 1e-6))
    gray = target_dn[0].mean(axis=0)
    grad_y = np.pad(np.abs(np.diff(gray, axis=0)), ((0, 1), (0, 0)), mode="edge")
    grad_x = np.pad(np.abs(np.diff(gray, axis=1)), ((0, 0), (0, 1)), mode="edge")
    flat = (grad_x < 8.0) & (grad_y < 8.0)
    flat = np.broadcast_to(flat[None, None], prediction_dn.shape) & static
    residual = (prediction_dn - target_dn)[flat]
    result["high_frequency_ratio_mean"] = float(np.mean(ratios))
    result["high_frequency_ratio_per_channel"] = ratios
    result["flat_residual_sigma_dn"] = float(np.std(residual)) if residual.size else float("nan")
    result["static_pixels"] = int(np.count_nonzero(static))
    return result


def save_four_panel(path: Path, input_frame: np.ndarray, teacher: np.ndarray, prediction: np.ndarray, reference: np.ndarray, black: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    def render(frame: np.ndarray) -> np.ndarray:
        dn = frame * (_WHITE_LEVEL - black[:, None, None]) + black[:, None, None]
        mosaic = np.zeros((frame.shape[1] * 2, frame.shape[2] * 2), dtype=np.float32)
        mosaic[0::2, 0::2], mosaic[0::2, 1::2] = dn[0], dn[1]
        mosaic[1::2, 0::2], mosaic[1::2, 1::2] = dn[2], dn[3]
        raw16 = np.rint(np.clip(mosaic / _WHITE_LEVEL, 0, 1) * 65535.0).astype(np.uint16)
        rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535.0
        return np.clip(rgb, 0, 1) ** (1 / 2.2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, image, title in zip(axes, (input_frame, teacher, prediction, reference), ("input", "old teacher", "A3 EMA", "A1 reference"), strict=True):
        axis.imshow(render(image)); axis.set_title(title); axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def run_evaluation(*, supervision_dir: Path, checkpoint: Path, teacher_checkpoint: Path, output_dir: Path, gains: list[str], max_frames: int, device: torch.device, tile_batch: int = 8) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = np.load(supervision_dir / "references" / "04_shdarkroom_10x_clean_reference.npy")
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    black = _channel_vector(report["black_levels"])
    motion_dir = supervision_dir / "motion_masks"
    results = {"protocol": "A4 leave-one-gain-out", "max_frames": max_frames, "tile_batch": tile_batch, "gains": {}}
    for gain in gains:
        prediction_path = output_dir / f"prediction_{gain}.npy"
        teacher_path = output_dir / f"old_teacher_{gain}.npy"
        metrics_path = output_dir / f"metrics_{gain}.json"
        panel_path = output_dir / f"four_panel_{gain}.png"
        if metrics_path.is_file() and prediction_path.is_file() and teacher_path.is_file() and panel_path.is_file():
            results["gains"][gain] = json.loads(metrics_path.read_text(encoding="utf-8"))
            (output_dir / "a4_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            continue

        video, overlap = _read_aligned_video(supervision_dir, gain, max_frames)
        manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
        shift_x = int(manifest["registration"]["integer_shift_x"])
        shift_y = int(manifest["registration"]["integer_shift_y"])
        source_motion = np.load(motion_dir / f"04_shdarkroom_{gain}.npy", mmap_mode="r")[: len(video), 0, ::2, ::2]
        motion = np.ones_like(source_motion, dtype=np.float32)
        y0, y1 = max(0, -shift_y), min(_PACKED_HEIGHT, _PACKED_HEIGHT - shift_y)
        x0, x1 = max(0, -shift_x), min(_PACKED_WIDTH, _PACKED_WIDTH - shift_x)
        motion[:, y0:y1, x0:x1] = source_motion[:, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
        if prediction_path.is_file():
            prediction = np.load(prediction_path).astype(np.float32, copy=False)
        else:
            model = load_checkpoint_model(checkpoint, device, ema=True)
            try:
                prediction = infer_video(model, video, device, tile_batch=tile_batch)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            np.save(prediction_path, prediction.astype(np.float16))
        if teacher_path.is_file():
            teacher_output = np.load(teacher_path).astype(np.float32, copy=False)
        else:
            teacher = load_official_model(teacher_checkpoint, device)
            try:
                teacher_output = infer_video(teacher, video, device, tile_batch=tile_batch)
            finally:
                del teacher
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            np.save(teacher_path, teacher_output.astype(np.float16))
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = evaluate_outputs(prediction, video, ref, overlap, motion, black)
            metrics["old_teacher"] = evaluate_outputs(teacher_output, video, ref, overlap, motion, black)["finetuned"]
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        if not panel_path.is_file():
            save_four_panel(panel_path, video[0], teacher_output[0], prediction[0], _normalise_dn(ref, black), black)
        results["gains"][gain] = metrics
        (output_dir / "a4_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        del video, prediction, teacher_output
    return results

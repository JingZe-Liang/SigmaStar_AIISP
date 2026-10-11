"""Native-domain evaluation for a gain-specific temporal reference."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from raw_fusion.finetune.a3 import (
    _PACKED_HEIGHT,
    _PACKED_WIDTH,
    _WHITE_LEVEL,
    _channel_vector,
    _normalise_dn,
    _load_json,
)
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.a4 import _metric, infer_video, load_checkpoint_model
from raw_fusion.finetune.stream_io import open_stream, pack_frame


def read_native_video(
    supervision_dir: Path,
    gain: str,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a gain without cross-gain exposure mapping."""

    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames = int(manifest["frames_used"] if max_frames is None else min(max_frames, manifest["frames_used"]))
    frames = min(frames, int(reader.frames))
    black = _channel_vector(manifest["black_levels"])
    video = np.zeros((frames, 4, _PACKED_HEIGHT, _PACKED_WIDTH), dtype=np.float32)
    for index in range(frames):
        packed = pack_frame(reader.read_frame(index), adapter).astype(np.float32)
        video[index] = _normalise_dn(packed, black)
    motion = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")[:frames, 0, ::2, ::2].astype(np.float32, copy=False)
    return video, motion, black


def _load_native_reference(probe_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    reference = np.load(probe_dir / "temporal_mean_native_dn.npy", mmap_mode="r").astype(np.float32, copy=False)
    valid = np.load(probe_dir / "valid_region.npy", mmap_mode="r").astype(bool, copy=False)
    if reference.shape != (4, _PACKED_HEIGHT, _PACKED_WIDTH):
        raise ValueError(f"unexpected native reference shape {reference.shape}")
    if valid.shape != (_PACKED_HEIGHT, _PACKED_WIDTH):
        raise ValueError(f"unexpected native valid-region shape {valid.shape}")
    return reference, valid


def evaluate_native_outputs(
    prediction: np.ndarray,
    input_video: np.ndarray,
    reference: np.ndarray,
    valid_region: np.ndarray,
    motion: np.ndarray,
    black: np.ndarray,
) -> dict[str, object]:
    """Evaluate predictions against a same-gain native-DN reference."""

    if prediction.shape != input_video.shape:
        raise ValueError(f"prediction/input shape mismatch: {prediction.shape} vs {input_video.shape}")
    if prediction.shape[1:] != reference.shape:
        raise ValueError(f"prediction/reference shape mismatch: {prediction.shape[1:]} vs {reference.shape}")
    if motion.shape != (prediction.shape[0], _PACKED_HEIGHT, _PACKED_WIDTH):
        raise ValueError(f"unexpected motion shape {motion.shape}")

    target_dn = np.broadcast_to(reference, prediction.shape).astype(np.float32, copy=False)
    prediction_dn = prediction * (_WHITE_LEVEL - black[None, :, None, None]) + black[None, :, None, None]
    input_dn = input_video * (_WHITE_LEVEL - black[None, :, None, None]) + black[None, :, None, None]
    static = valid_region[None] & (motion < 0.5)
    static &= np.all(input_video < ((3800.0 - black) / (_WHITE_LEVEL - black))[None, :, None, None], axis=1)
    static = np.broadcast_to(static[:, None], prediction.shape)
    temporal_static = static[1:] & static[:-1]
    result: dict[str, object] = {
        "finetuned": _metric(prediction_dn, target_dn, static),
        "input": _metric(input_dn, target_dn, static),
        "temporal_mad_dn": float(np.mean(np.abs(np.diff(prediction_dn, axis=0))[temporal_static])) if temporal_static.any() else float("nan"),
        "input_temporal_mad_dn": float(np.mean(np.abs(np.diff(input_dn, axis=0))[temporal_static])) if temporal_static.any() else float("nan"),
        "legal_fraction": float(np.mean((prediction_dn >= 0.0) & (prediction_dn <= _WHITE_LEVEL))),
    }

    sy0, sy1, sx0, sx1 = (400, 500, 160, 290)
    ref_roi = target_dn[:, :, sy0:sy1, sx0:sx1]
    pred_roi = prediction_dn[:, :, sy0:sy1, sx0:sx1]
    ratios: list[float] = []
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
    flat = np.broadcast_to(flat[None, None], prediction.shape) & static
    residual = (prediction_dn - target_dn)[flat]
    result["high_frequency_ratio_mean"] = float(np.mean(ratios))
    result["high_frequency_ratio_per_channel"] = ratios
    result["flat_residual_sigma_dn"] = float(np.std(residual)) if residual.size else float("nan")
    result["static_pixels"] = int(np.count_nonzero(static))
    result["static_fraction"] = float(np.mean(static))
    return result


def save_native_panel(
    path: Path,
    input_frame: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    black: np.ndarray,
) -> None:
    """Save input, prediction, and native reference for visual inspection."""

    import matplotlib.pyplot as plt

    def render(frame: np.ndarray, *, frame_is_dn: bool) -> np.ndarray:
        # Input and prediction are normalized, while the native reference is already in DN.
        dn = frame if frame_is_dn else frame * (_WHITE_LEVEL - black[:, None, None]) + black[:, None, None]
        mosaic = np.zeros((frame.shape[1] * 2, frame.shape[2] * 2), dtype=np.float32)
        mosaic[0::2, 0::2], mosaic[0::2, 1::2] = dn[0], dn[1]
        mosaic[1::2, 0::2], mosaic[1::2, 1::2] = dn[2], dn[3]
        raw16 = np.rint(np.clip(mosaic / _WHITE_LEVEL, 0, 1) * 65535.0).astype(np.uint16)
        rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535.0
        return np.clip(rgb, 0, 1) ** (1 / 2.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, image, title, frame_is_dn in zip(
        axes,
        (input_frame, prediction, reference),
        ("native input", "A3 EMA", "1000x native reference"),
        (False, False, True),
        strict=True,
    ):
        axis.imshow(render(image, frame_is_dn=frame_is_dn))
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_native_evaluation(
    *,
    supervision_dir: Path,
    probe_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    max_frames: int,
    device: torch.device,
    tile_batch: int = 8,
) -> dict[str, object]:
    """Run the independent 1000x native-reference evaluation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "native_prediction_1000x.npy"
    metrics_path = output_dir / "native_metrics_1000x.json"
    panel_path = output_dir / "native_four_panel_1000x.png"
    input_video, motion, black = read_native_video(supervision_dir, "1000x", max_frames)
    reference, valid_region = _load_native_reference(probe_dir)
    if prediction_path.is_file():
        prediction = np.load(prediction_path).astype(np.float32, copy=False)
        if prediction.shape != input_video.shape:
            raise ValueError(f"cached prediction shape {prediction.shape} does not match input {input_video.shape}")
    else:
        model = load_checkpoint_model(checkpoint, device, ema=True)
        prediction = infer_video(model, input_video, device, tile_batch=tile_batch)
        np.save(prediction_path, prediction.astype(np.float16))
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = evaluate_native_outputs(prediction, input_video, reference, valid_region, motion, black)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if not panel_path.is_file():
        save_native_panel(panel_path, input_video[0], prediction[0], reference, black)
    result = {
        "protocol": "A5 native-gain reference",
        "gain": "1000x",
        "reference_domain": "native packed DN",
        "max_frames": max_frames,
        "tile_batch": tile_batch,
        "cross_gain_holdout": {
            "gain": "645x",
            "protocol": "A4 leave-one-gain-out against 10x reference",
            "metrics_path": str(output_dir.parent / "metrics_645x.json"),
            "available": (output_dir.parent / "metrics_645x.json").is_file(),
        },
        "metrics": metrics,
        "outputs": {
            "prediction": str(prediction_path),
            "metrics": str(metrics_path),
            "panel": str(panel_path),
        },
    }
    (output_dir / "native_1000x_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

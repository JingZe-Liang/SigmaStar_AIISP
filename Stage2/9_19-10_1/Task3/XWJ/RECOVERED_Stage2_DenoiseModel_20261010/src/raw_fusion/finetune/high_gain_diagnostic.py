"""Inference-only diagnostics for high-gain A3 temporal and tile artifacts."""

from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
import torch

from raw_fusion.finetune.all_dataset_preview import (
    PreviewSequence,
    denormalize_packed,
    render_traditional_isp,
    save_json,
    sha256_file,
    video_encoder,
)


def spatial_starts(length: int, tile: int, stride: int) -> tuple[int, ...]:
    """Return deterministic tile starts with complete edge coverage."""

    if tile <= 0 or stride <= 0 or stride > tile:
        raise ValueError("tile and stride must satisfy 0 < stride <= tile")
    if length < tile:
        raise ValueError(f"length {length} is smaller than tile {tile}")
    if length == tile:
        return (0,)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return tuple(starts)


def spatial_blend_window(tile: int, mode: str, floor: float) -> np.ndarray:
    """Build a positive 2D tile window for normalized overlap-add."""

    if tile <= 1:
        raise ValueError("tile must be greater than one")
    if not 0.0 < floor <= 1.0:
        raise ValueError("window floor must be in (0, 1]")
    if mode == "uniform":
        axis = np.ones(tile, dtype=np.float32)
    elif mode == "hann":
        axis = np.hanning(tile).astype(np.float32)
        axis = floor + (1.0 - floor) * axis
    else:
        raise ValueError(f"unsupported blend window {mode!r}")
    window = np.outer(axis, axis).astype(np.float32)
    return window / float(window.max())


@torch.no_grad()
def infer_tiled_clip(
    model: torch.nn.Module,
    clip: np.ndarray,
    device: torch.device,
    *,
    tile: int,
    stride: int,
    tile_batch: int,
    blend_mode: str,
    blend_floor: float,
) -> np.ndarray:
    """Infer one temporal clip using weighted spatial overlap-add."""

    if clip.ndim != 4 or clip.shape[1] != 4:
        raise ValueError(f"expected [T,4,H,W], got {clip.shape}")
    if clip.shape[0] != 8:
        raise ValueError(f"A3 requires exactly eight frames, got {clip.shape[0]}")
    if tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    y_starts = spatial_starts(clip.shape[2], tile, stride)
    x_starts = spatial_starts(clip.shape[3], tile, stride)
    positions = tuple((y, x) for y in y_starts for x in x_starts)
    window = spatial_blend_window(tile, blend_mode, blend_floor)
    weighted_window = window[None, None]
    output = np.zeros_like(clip, dtype=np.float32)
    weights = np.zeros((1, 1, clip.shape[2], clip.shape[3]), dtype=np.float32)

    model.eval()
    for batch_start in range(0, len(positions), tile_batch):
        batch_positions = positions[batch_start : batch_start + tile_batch]
        patches = np.stack(
            [clip[:, :, y : y + tile, x : x + tile] for y, x in batch_positions]
        )
        prediction = (
            model(torch.from_numpy(patches).to(device))
            .clamp(0.0, 1.0)
            .detach()
            .cpu()
            .numpy()
        )
        for patch_prediction, (y, x) in zip(
            prediction, batch_positions, strict=True
        ):
            output[:, :, y : y + tile, x : x + tile] += (
                patch_prediction * weighted_window
            )
            weights[:, :, y : y + tile, x : x + tile] += weighted_window
    if float(weights.min()) <= 0.0:
        raise RuntimeError("tile blending left uncovered pixels")
    return output / weights


def infer_frame_repeated(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
    *,
    temporal_output_index: int,
    tile: int,
    stride: int,
    tile_batch: int,
    blend_mode: str,
    blend_floor: float,
) -> np.ndarray:
    """Denoise one frame without cross-frame evidence by repeating it eight times."""

    if frame.ndim != 3 or frame.shape[0] != 4:
        raise ValueError(f"expected [4,H,W], got {frame.shape}")
    if not 0 <= temporal_output_index < 8:
        raise ValueError("temporal_output_index must be in [0, 8)")
    repeated = np.repeat(frame[None], 8, axis=0).astype(np.float32, copy=False)
    prediction = infer_tiled_clip(
        model,
        repeated,
        device,
        tile=tile,
        stride=stride,
        tile_batch=tile_batch,
        blend_mode=blend_mode,
        blend_floor=blend_floor,
    )
    return prediction[temporal_output_index]


def _gradient_mean(video: np.ndarray) -> float:
    luma = video.mean(axis=1)
    horizontal = np.abs(np.diff(luma, axis=2)).mean()
    vertical = np.abs(np.diff(luma, axis=1)).mean()
    return float((horizontal + vertical) * 0.5)


def _temporal_mean_gradient(video: np.ndarray) -> float:
    return _gradient_mean(video.mean(axis=0, keepdims=True))


def _blur_video(video: np.ndarray, kernel: int) -> np.ndarray:
    blurred = np.empty_like(video, dtype=np.float32)
    for frame_index in range(video.shape[0]):
        for channel_index in range(video.shape[1]):
            blurred[frame_index, channel_index] = cv2.blur(
                video[frame_index, channel_index], (kernel, kernel)
            )
    return blurred


def _highpass_rms(video: np.ndarray) -> float:
    residual = video - _blur_video(video, 3)
    return float(np.sqrt(np.mean(residual * residual)))


def _tile_boundaries(length: int, tile: int, stride: int) -> tuple[int, ...]:
    starts = spatial_starts(length, tile, stride)
    boundaries = set(starts[1:])
    boundaries.update(start + tile for start in starts[:-1] if start + tile < length)
    return tuple(sorted(boundaries))


def tile_discontinuity(video: np.ndarray, tile: int, stride: int) -> dict[str, float]:
    """Measure discontinuities on tile start/end lines against ordinary neighbors."""

    luma = video.mean(axis=1)
    dx = np.abs(np.diff(luma, axis=2))
    dy = np.abs(np.diff(luma, axis=1))
    x_boundaries = _tile_boundaries(video.shape[3], tile, stride)
    y_boundaries = _tile_boundaries(video.shape[2], tile, stride)
    boundary_values: list[np.ndarray] = []
    ordinary_x = np.ones(dx.shape[2], dtype=bool)
    ordinary_y = np.ones(dy.shape[1], dtype=bool)
    for boundary in x_boundaries:
        index = boundary - 1
        boundary_values.append(dx[:, :, index].reshape(-1))
        ordinary_x[max(0, index - 1) : min(dx.shape[2], index + 2)] = False
    for boundary in y_boundaries:
        index = boundary - 1
        boundary_values.append(dy[:, index, :].reshape(-1))
        ordinary_y[max(0, index - 1) : min(dy.shape[1], index + 2)] = False
    if not boundary_values or not ordinary_x.any() or not ordinary_y.any():
        raise ValueError("tile grid does not contain enough boundary and ordinary pixels")
    boundary_mean = float(np.concatenate(boundary_values).mean())
    ordinary_mean = float(
        np.concatenate((dx[:, :, ordinary_x].reshape(-1), dy[:, ordinary_y].reshape(-1))).mean()
    )
    return {
        "boundary_mean": boundary_mean,
        "ordinary_mean": ordinary_mean,
        "boundary_to_ordinary_ratio": boundary_mean / max(ordinary_mean, 1e-12),
    }


def evaluate_variants(
    variants: dict[str, np.ndarray],
    tile_grids: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Evaluate detail, temporal change, legality, and tile discontinuity."""

    if "input" not in variants:
        raise ValueError("variants must contain input")
    shape = variants["input"].shape
    if any(video.shape != shape for video in variants.values()):
        raise ValueError("all variants must have the same shape")
    input_gradient = _gradient_mean(variants["input"])
    input_temporal_mean_gradient = _temporal_mean_gradient(variants["input"])
    input_temporal_mad = float(np.abs(np.diff(variants["input"], axis=0)).mean())
    input_lowpass_gradient = _gradient_mean(_blur_video(variants["input"], 5))
    input_highpass_rms = _highpass_rms(variants["input"])
    result: dict[str, Any] = {
        "shape": list(shape),
        "tile_grids": {
            name: {"tile": tile, "stride": stride}
            for name, (tile, stride) in tile_grids.items()
        },
        "variants": {},
    }
    for name, video in variants.items():
        gradient = _gradient_mean(video)
        temporal_mean_gradient = _temporal_mean_gradient(video)
        temporal_mad = float(np.abs(np.diff(video, axis=0)).mean())
        lowpass_gradient = _gradient_mean(_blur_video(video, 5))
        highpass_rms = _highpass_rms(video)
        metrics: dict[str, Any] = {
            "gradient_mean": gradient,
            "gradient_ratio_to_input": gradient / max(input_gradient, 1e-12),
            "temporal_mean_gradient": temporal_mean_gradient,
            "temporal_mean_gradient_ratio_to_input": temporal_mean_gradient
            / max(input_temporal_mean_gradient, 1e-12),
            "temporal_mad": temporal_mad,
            "temporal_mad_ratio_to_input": temporal_mad
            / max(input_temporal_mad, 1e-12),
            "lowpass_gradient_mean": lowpass_gradient,
            "lowpass_gradient_ratio_to_input": lowpass_gradient
            / max(input_lowpass_gradient, 1e-12),
            "highpass_rms": highpass_rms,
            "highpass_rms_ratio_to_input": highpass_rms
            / max(input_highpass_rms, 1e-12),
            "legal_raw_fraction": float(np.mean((video >= 0.0) & (video <= 1.0))),
        }
        metrics["tile_discontinuity"] = {
            grid_name: tile_discontinuity(video, tile, stride)
            for grid_name, (tile, stride) in tile_grids.items()
        }
        result["variants"][name] = metrics
    if "a3_uniform" in variants:
        baseline = variants["a3_uniform"]
        for name, video in variants.items():
            result["variants"][name]["mae_vs_a3_uniform"] = float(
                np.abs(video - baseline).mean()
            )
    for grid_name in tile_grids:
        input_ratio = result["variants"]["input"]["tile_discontinuity"][grid_name][
            "boundary_to_ordinary_ratio"
        ]
        for metrics in result["variants"].values():
            ratio = metrics["tile_discontinuity"][grid_name][
                "boundary_to_ordinary_ratio"
            ]
            metrics["tile_discontinuity"][grid_name][
                "amplification_vs_input"
            ] = ratio / max(input_ratio, 1e-12)
    return result


def _resize_cell(image: np.ndarray, label: str, detail: bool) -> np.ndarray:
    if detail:
        height, width = image.shape[:2]
        image = image[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    cell = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    cell = np.ascontiguousarray(cell)
    cv2.rectangle(cell, (0, 0), (960, 44), (10, 10, 10), -1)
    cv2.putText(
        cell,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return cell


def _open_encoder(
    output_path: Path, fps: int, preferred_encoder: str, crf: int, pixel_format: str
) -> tuple[subprocess.Popen[bytes], list[str], str]:
    encoder = video_encoder(preferred_encoder)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        encoder,
    ]
    if encoder == "libx264":
        command.extend(["-crf", str(crf), "-pix_fmt", pixel_format])
    elif encoder == "libx265":
        command.extend(["-x265-params", f"crf={crf}", "-pix_fmt", pixel_format])
    else:
        command.extend(["-q:v", "2", "-pix_fmt", "yuv420p"])
    command.append(str(output_path))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    return process, command, encoder


def render_comparison_videos(
    output_dir: Path,
    sequence: PreviewSequence,
    source_start: int,
    black: np.ndarray,
    exposure_ev: float,
    variants: dict[str, np.ndarray],
    labels: dict[str, str],
    isp_config: dict[str, Any],
    video_config: dict[str, Any],
) -> dict[str, Any]:
    """Render synchronized 2x2 overview and center-detail comparison videos."""

    order = ("input", "a3_uniform", "a3_weighted", "a3_frame_repeated")
    if tuple(variants) != order:
        raise ValueError(f"variant order must be {order}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = {
        "overview": output_dir / f"{sequence.name}_high_gain_4way_overview.mp4",
        "detail": output_dir / f"{sequence.name}_high_gain_4way_detail.mp4",
    }
    encoders = {
        name: _open_encoder(
            path,
            int(video_config["fps"]),
            str(video_config["preferred_encoder"]),
            int(video_config["crf"]),
            str(video_config["pixel_format"]),
        )
        for name, path in video_paths.items()
    }
    first_frames: dict[str, np.ndarray] = {}
    try:
        for frame_index in range(variants["input"].shape[0]):
            rendered: dict[str, np.ndarray] = {}
            for name in order:
                frame_dn = denormalize_packed(variants[name][frame_index], black)
                rendered[name] = render_traditional_isp(
                    frame_dn,
                    black,
                    sequence.wb_rgb,
                    exposure_ev,
                    sequence.cfa,
                    highlight_recovery=bool(isp_config["highlight_recovery"]),
                    tone_map=str(isp_config["tone_map"]),
                )
            marker = f"source frame {source_start + frame_index}"
            for kind, detail in (("overview", False), ("detail", True)):
                cells = [
                    _resize_cell(rendered[name], f"{labels[name]} | {marker}", detail)
                    for name in order
                ]
                panel = np.concatenate(
                    (np.concatenate(cells[:2], axis=1), np.concatenate(cells[2:], axis=1)),
                    axis=0,
                )
                if panel.shape != (1080, 1920, 3):
                    raise ValueError(f"unexpected panel shape {panel.shape}")
                if kind not in first_frames:
                    first_frames[kind] = panel.copy()
                process = encoders[kind][0]
                if process.stdin is None:
                    raise RuntimeError("ffmpeg stdin disappeared")
                process.stdin.write(panel.tobytes())
        for process, _, _ in encoders.values():
            if process.stdin is None or process.stderr is None:
                raise RuntimeError("ffmpeg pipes are absent")
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr[-3000:]}")
    except BaseException:
        for kind, (process, _, _) in encoders.items():
            process.kill()
            process.wait()
            video_paths[kind].unlink(missing_ok=True)
        raise

    previews: dict[str, Path] = {}
    for kind, frame in first_frames.items():
        preview = output_dir / f"{sequence.name}_high_gain_4way_{kind}_frame0000.png"
        if not cv2.imwrite(str(preview), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {preview}")
        previews[kind] = preview
    manifest = {
        "sequence": sequence.to_mapping(),
        "source_start": source_start,
        "frames": int(variants["input"].shape[0]),
        "layout": "2x2: Input, A3 uniform 64/56, A3 weighted 64/48, A3 frame-repeated weighted 64/48",
        "videos": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "ffmpeg_command": encoders[name][1],
                "encoder": encoders[name][2],
            }
            for name, path in video_paths.items()
        },
        "previews": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in previews.items()
        },
        "isp": {
            **isp_config,
            "fixed_exposure_ev": exposure_ev,
            "black_levels_R_Gr_Gb_B": [float(value) for value in black],
            "wb_rgb": list(sequence.wb_rgb),
        },
    }
    save_json(output_dir / "video_manifest.json", manifest)
    return manifest

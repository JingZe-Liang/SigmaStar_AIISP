"""Deterministic motion-region fusion between A3 and the official teacher."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FlowMaskParams:
    temporal_gap: int = 4
    min_flow_px: float = 1.0
    max_flow_px: float = 30.0
    max_forward_backward_error_px: float = 1.5
    max_photometric_error: float = 0.08
    seed_dilation_radius: int = 3
    feather_sigma: float = 1.0


def build_flow_motion_weights(
    video: np.ndarray,
    dynamic_masks: np.ndarray,
    saturation_limits: np.ndarray,
    params: FlowMaskParams,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Build inward-feathered flow-confidence weights constrained to A0 motion."""

    if video.ndim != 4 or video.shape[1] != 4:
        raise ValueError(f"expected video [T,4,H,W], got {video.shape}")
    if dynamic_masks.shape != (video.shape[0], video.shape[2], video.shape[3]):
        raise ValueError(
            f"dynamic mask shape {dynamic_masks.shape} does not match video {video.shape}"
        )
    if saturation_limits.shape != (4,):
        raise ValueError(f"expected four saturation limits, got {saturation_limits.shape}")
    if params.temporal_gap < 1 or params.temporal_gap >= video.shape[0]:
        raise ValueError("temporal_gap must be positive and shorter than the video")
    if params.seed_dilation_radius < 0 or params.feather_sigma < 0.0:
        raise ValueError("mask dilation and feathering parameters must be non-negative")

    frames, _, height, width = video.shape
    weights = np.zeros((frames, height, width), dtype=np.float32)
    records: list[dict[str, float]] = []
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setVariationalRefinementIterations(2)
    kernel_size = 2 * params.seed_dilation_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for frame_index in range(frames):
        if frame_index + params.temporal_gap < frames:
            target_index = frame_index + params.temporal_gap
        else:
            target_index = frame_index - params.temporal_gap
        source = video[frame_index]
        target = video[target_index]
        source_gray = np.clip(np.mean(source[1:3], axis=0), 0.0, 1.0).astype(np.float32)
        target_gray = np.clip(np.mean(target[1:3], axis=0), 0.0, 1.0).astype(np.float32)
        source_u8 = np.rint(source_gray * 255.0).astype(np.uint8)
        target_u8 = np.rint(target_gray * 255.0).astype(np.uint8)
        forward = dis.calc(source_u8, target_u8, None)
        backward = dis.calc(target_u8, source_u8, None)

        map_x = grid_x + forward[..., 0]
        map_y = grid_y + forward[..., 1]
        warped_backward = np.stack(
            [
                cv2.remap(
                    backward[..., channel],
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                for channel in range(2)
            ],
            axis=-1,
        )
        warped_target_green = cv2.remap(
            target_gray,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_target_motion = cv2.remap(
            dynamic_masks[target_index].astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )
        flow_error = np.linalg.norm(forward + warped_backward, axis=-1)
        photometric_error = np.abs(source_gray - warped_target_green)
        flow_magnitude = np.linalg.norm(forward, axis=-1)
        current_dynamic = dynamic_masks[frame_index] >= 0.5
        paired_dynamic = current_dynamic | (warped_target_motion >= 0.5)
        source_valid = np.all(source < saturation_limits[:, None, None], axis=0)
        target_valid = np.all(target < saturation_limits[:, None, None], axis=0)
        inside = (map_x >= 1.0) & (map_x < width - 1.0)
        inside &= (map_y >= 1.0) & (map_y < height - 1.0)
        confidence = paired_dynamic & source_valid & target_valid & inside
        confidence &= flow_error < params.max_forward_backward_error_px
        confidence &= photometric_error < params.max_photometric_error
        confidence &= flow_magnitude >= params.min_flow_px
        confidence &= flow_magnitude < params.max_flow_px

        expanded = confidence.astype(np.uint8)
        if params.seed_dilation_radius > 0:
            expanded = cv2.dilate(expanded, kernel)
        expanded = expanded.astype(np.float32) * current_dynamic.astype(np.float32)
        if params.feather_sigma > 0.0:
            expanded = cv2.GaussianBlur(expanded, (0, 0), params.feather_sigma)
        expanded *= current_dynamic.astype(np.float32)
        weights[frame_index] = np.clip(expanded, 0.0, 1.0)

        dynamic_count = int(paired_dynamic.sum())
        records.append(
            {
                "frame": float(frame_index),
                "target_frame": float(target_index),
                "current_dynamic_fraction": float(current_dynamic.mean()),
                "paired_dynamic_fraction": float(paired_dynamic.mean()),
                "strict_confidence_fraction": float(confidence.mean()),
                "strict_confidence_fraction_of_dynamic": float(
                    confidence.sum() / max(dynamic_count, 1)
                ),
                "fusion_weight_fraction": float((weights[frame_index] > 0.0).mean()),
                "fusion_weight_mean": float(weights[frame_index].mean()),
            }
        )
    return weights, records


def fuse_motion_regions(
    baseline: np.ndarray,
    teacher: np.ndarray,
    weights: np.ndarray,
    black: np.ndarray,
    alpha: float,
    residual_clip_dn: float,
    white_level: float = 4095.0,
) -> np.ndarray:
    """Inject a bounded teacher residual only where motion weights are non-zero."""

    if baseline.shape != teacher.shape or baseline.ndim != 4 or baseline.shape[1] != 4:
        raise ValueError(
            f"baseline and teacher must match [T,4,H,W], got {baseline.shape} and {teacher.shape}"
        )
    if weights.shape != (baseline.shape[0], baseline.shape[2], baseline.shape[3]):
        raise ValueError(f"weight shape {weights.shape} does not match {baseline.shape}")
    if black.shape != (4,):
        raise ValueError(f"expected four black levels, got {black.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if residual_clip_dn <= 0.0:
        raise ValueError("residual_clip_dn must be positive")

    scale = (white_level - black)[None, :, None, None]
    baseline_dn = baseline.astype(np.float32) * scale + black[None, :, None, None]
    teacher_dn = teacher.astype(np.float32) * scale + black[None, :, None, None]
    residual_dn = np.clip(teacher_dn - baseline_dn, -residual_clip_dn, residual_clip_dn)
    candidate_dn = baseline_dn + alpha * weights[:, None] * residual_dn
    return np.clip(
        (candidate_dn - black[None, :, None, None]) / scale,
        0.0,
        1.0,
    ).astype(np.float32)


def motion_temporal_summary(
    prediction: np.ndarray,
    input_video: np.ndarray,
    dynamic_masks: np.ndarray,
) -> dict[str, float]:
    """Temporal-change diagnostic over A0 motion transitions."""

    if prediction.shape != input_video.shape:
        raise ValueError("prediction and input_video must have identical shapes")
    transition = (dynamic_masks[1:] >= 0.5) | (dynamic_masks[:-1] >= 0.5)
    static = ~transition
    prediction_diff = np.abs(np.diff(prediction, axis=0)).transpose(0, 2, 3, 1)
    input_diff = np.abs(np.diff(input_video, axis=0)).transpose(0, 2, 3, 1)
    moving_prediction = prediction_diff[transition]
    moving_input = input_diff[transition]
    static_prediction = prediction_diff[static]
    return {
        "motion_mean_abs_normalized": float(np.mean(moving_prediction)),
        "input_motion_mean_abs_normalized": float(np.mean(moving_input)),
        "motion_to_input_ratio": float(
            np.mean(moving_prediction) / max(float(np.mean(moving_input)), 1e-8)
        ),
        "static_mean_abs_normalized": float(np.mean(static_prediction)),
    }


def fusion_shift_summary(
    candidate: np.ndarray,
    baseline: np.ndarray,
    dynamic_masks: np.ndarray,
    black: np.ndarray,
    white_level: float = 4095.0,
) -> dict[str, float]:
    """Report how much the fusion moved A3 inside and outside A0 motion."""

    scale = (white_level - black)[None, :, None, None]
    shift_dn = np.abs(candidate - baseline) * scale
    dynamic = np.broadcast_to((dynamic_masks >= 0.5)[:, None], shift_dn.shape)
    static = ~dynamic
    return {
        "motion_mean_abs_dn": float(np.mean(shift_dn[dynamic])) if dynamic.any() else 0.0,
        "motion_p95_abs_dn": float(np.percentile(shift_dn[dynamic], 95)) if dynamic.any() else 0.0,
        "static_mean_abs_dn": float(np.mean(shift_dn[static])) if static.any() else 0.0,
        "static_max_abs_dn": float(np.max(shift_dn[static])) if static.any() else 0.0,
    }

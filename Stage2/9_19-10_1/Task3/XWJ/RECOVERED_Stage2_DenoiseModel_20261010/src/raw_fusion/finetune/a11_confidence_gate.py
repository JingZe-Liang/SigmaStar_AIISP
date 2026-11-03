"""Deterministic input-evidence confidence gating for high-gain A3 outputs."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from raw_fusion.finetune.all_dataset_preview import (
    PreviewSequence,
    denormalize_packed,
    render_traditional_isp,
    save_json,
    sha256_file,
    video_encoder,
)
from raw_fusion.finetune.high_gain_diagnostic import evaluate_variants


@dataclass(frozen=True, slots=True)
class ConfidenceGateParams:
    temporal_gap: int
    residual_low_z: float
    residual_high_z: float
    gradient_low: float
    gradient_high: float
    structure_low: float
    structure_high: float
    flow_low_px: float
    flow_high_px: float
    fb_low_px: float
    fb_high_px: float
    photometric_low_z: float
    photometric_high_z: float
    seed_threshold: float
    min_component_area: int
    dilation_radius: int
    feather_sigma: float
    min_weight: float
    max_weight_residual: float
    max_weight_gradient: float
    max_weight_full: float

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ConfidenceGateParams":
        return cls(**payload)


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not high > low:
        raise ValueError("smoothstep high must be greater than low")
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _green_luma(video: np.ndarray) -> np.ndarray:
    if video.ndim != 4 or video.shape[1] != 4:
        raise ValueError(f"expected [T,4,H,W], got {video.shape}")
    return np.mean(video[:, 1:3], axis=1, dtype=np.float32)


def _gradient_components(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, cv2.magnitude(gx, gy)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    return _gradient_components(image)[2]


def _local_noise(image: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(image, (0, 0), 1.0)
    residual = np.abs(image - smooth)
    return cv2.GaussianBlur(residual, (0, 0), 2.0) + np.float32(0.0015)


def _flow_image(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate((source.reshape(-1), target.reshape(-1)))
    low, high = np.quantile(combined, (0.01, 0.995))
    scale = max(float(high - low), 1e-4)

    def convert(image: np.ndarray) -> np.ndarray:
        normalized = np.clip((image - low) / scale, 0.0, 1.0)
        return np.rint(normalized * 255.0).astype(np.uint8)

    return convert(source), convert(target)


def _pair_evidence(
    source: np.ndarray,
    target: np.ndarray,
    params: ConfidenceGateParams,
) -> dict[str, np.ndarray]:
    source_noise = _local_noise(source)
    target_noise = _local_noise(target)
    source_evidence = cv2.GaussianBlur(source, (0, 0), 1.2)
    target_evidence = cv2.GaussianBlur(target, (0, 0), 1.2)
    noise_scale = 0.35 * (source_noise + target_noise) + np.float32(0.001)
    raw_residual_z = np.abs(source_evidence - target_evidence) / noise_scale
    residual_score = _smoothstep(
        raw_residual_z, params.residual_low_z, params.residual_high_z
    )

    source_gx, source_gy, source_gradient = _gradient_components(source_evidence)
    target_gx, target_gy, target_gradient = _gradient_components(target_evidence)
    structure = np.maximum(source_gradient, target_gradient)
    structure_score = _smoothstep(
        structure, params.structure_low, params.structure_high
    )
    magnitude_disagreement = np.abs(source_gradient - target_gradient) / (
        source_gradient + target_gradient + np.float32(0.002)
    )
    direction_cosine = (source_gx * target_gx + source_gy * target_gy) / (
        source_gradient * target_gradient + np.float32(1e-6)
    )
    direction_disagreement = np.clip((1.0 - direction_cosine) * 0.5, 0.0, 1.0)
    gradient_disagreement = np.maximum(magnitude_disagreement, direction_disagreement)
    gradient_score = _smoothstep(
        gradient_disagreement, params.gradient_low, params.gradient_high
    ) * structure_score

    source_u8, target_u8 = _flow_image(source_evidence, target_evidence)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setVariationalRefinementIterations(2)
    forward = dis.calc(source_u8, target_u8, None)
    backward = dis.calc(target_u8, source_u8, None)
    height, width = source.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
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
    warped_target = cv2.remap(
        target_evidence,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    flow_magnitude = np.linalg.norm(forward, axis=-1)
    fb_error = np.linalg.norm(forward + warped_backward, axis=-1)
    warped_residual_z = np.abs(source_evidence - warped_target) / noise_scale
    flow_score = _smoothstep(flow_magnitude, params.flow_low_px, params.flow_high_px)
    fb_score = _smoothstep(fb_error, params.fb_low_px, params.fb_high_px)
    photometric_score = _smoothstep(
        warped_residual_z,
        params.photometric_low_z,
        params.photometric_high_z,
    )
    flow_reliability = (1.0 - fb_score) * (1.0 - photometric_score)
    matched_motion = flow_score * flow_reliability * np.maximum(structure_score, 0.35)
    mismatch = np.maximum(fb_score, photometric_score) * np.maximum(
        gradient_score, 0.35 * structure_score
    )
    full_score = residual_score * np.maximum(matched_motion, mismatch)

    return {
        "residual": residual_score.astype(np.float32),
        "gradient": (residual_score * gradient_score).astype(np.float32),
        "full": full_score.astype(np.float32),
        "flow": flow_score.astype(np.float32),
        "mismatch": mismatch.astype(np.float32),
    }


def _remove_small_components(seed: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        raise ValueError("min_component_area must be positive")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed.astype(np.uint8), connectivity=8
    )
    kept = np.zeros(seed.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == label] = 1
    return kept


def _postprocess_score(
    score: np.ndarray,
    static_core: np.ndarray,
    max_weight: float,
    params: ConfidenceGateParams,
) -> tuple[np.ndarray, np.ndarray]:
    seed = _remove_small_components(score >= params.seed_threshold, params.min_component_area)
    if params.dilation_radius > 0:
        size = params.dilation_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        support = cv2.dilate(seed, kernel).astype(np.float32)
    else:
        support = seed.astype(np.float32)
    if params.feather_sigma > 0.0:
        support = cv2.GaussianBlur(support, (0, 0), params.feather_sigma)
        continuous = cv2.GaussianBlur(score, (0, 0), 1.0)
    else:
        continuous = score
    weight = max_weight * support * (0.35 + 0.65 * continuous)
    weight[weight < params.min_weight] = 0.0
    weight[static_core] = 0.0
    return np.clip(weight, 0.0, max_weight).astype(np.float32), seed.astype(bool)


def build_confidence_weights(
    input_video: np.ndarray,
    params: ConfidenceGateParams,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, float]]]:
    """Build conservative residual, gradient, and matching-confidence weights."""

    if params.temporal_gap <= 0 or params.temporal_gap >= input_video.shape[0]:
        raise ValueError("temporal_gap must be positive and shorter than the clip")
    luma = _green_luma(input_video)
    frames, height, width = luma.shape
    scores = {
        name: np.zeros((frames, height, width), dtype=np.float32)
        for name in ("residual_only", "residual_gradient", "full")
    }
    flow_scores = np.zeros_like(scores["full"])
    mismatch_scores = np.zeros_like(scores["full"])
    records: list[dict[str, float]] = []

    cv2.setRNGSeed(0)
    for frame_index in range(frames):
        targets = [
            target
            for target in (
                frame_index - params.temporal_gap,
                frame_index + params.temporal_gap,
            )
            if 0 <= target < frames
        ]
        if not targets:
            raise RuntimeError(f"frame {frame_index} has no temporal comparison target")
        for target_index in targets:
            evidence = _pair_evidence(luma[frame_index], luma[target_index], params)
            scores["residual_only"][frame_index] = np.maximum(
                scores["residual_only"][frame_index], evidence["residual"]
            )
            scores["residual_gradient"][frame_index] = np.maximum(
                scores["residual_gradient"][frame_index], evidence["gradient"]
            )
            scores["full"][frame_index] = np.maximum(
                scores["full"][frame_index], evidence["full"]
            )
            flow_scores[frame_index] = np.maximum(
                flow_scores[frame_index], evidence["flow"]
            )
            mismatch_scores[frame_index] = np.maximum(
                mismatch_scores[frame_index], evidence["mismatch"]
            )
        records.append(
            {
                "frame": float(frame_index),
                "comparison_count": float(len(targets)),
                "residual_score_mean": float(scores["residual_only"][frame_index].mean()),
                "residual_score_p99": float(
                    np.percentile(scores["residual_only"][frame_index], 99)
                ),
                "residual_score_max": float(scores["residual_only"][frame_index].max()),
                "gradient_score_mean": float(scores["residual_gradient"][frame_index].mean()),
                "gradient_score_p99": float(
                    np.percentile(scores["residual_gradient"][frame_index], 99)
                ),
                "gradient_score_max": float(
                    scores["residual_gradient"][frame_index].max()
                ),
                "full_score_mean": float(scores["full"][frame_index].mean()),
                "full_score_p99": float(np.percentile(scores["full"][frame_index], 99)),
                "full_score_max": float(scores["full"][frame_index].max()),
                "flow_score_mean": float(flow_scores[frame_index].mean()),
                "flow_score_p99": float(np.percentile(flow_scores[frame_index], 99)),
                "mismatch_score_mean": float(mismatch_scores[frame_index].mean()),
                "mismatch_score_p99": float(
                    np.percentile(mismatch_scores[frame_index], 99)
                ),
            }
        )

    static_core = (
        (scores["residual_only"] < 0.05)
        & (flow_scores < 0.05)
        & (mismatch_scores < 0.05)
    )
    static_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for frame_index in range(frames):
        static_core[frame_index] = cv2.erode(
            static_core[frame_index].astype(np.uint8), static_kernel
        ).astype(bool)

    maxima = {
        "residual_only": params.max_weight_residual,
        "residual_gradient": params.max_weight_gradient,
        "full": params.max_weight_full,
    }
    weights: dict[str, np.ndarray] = {}
    seeds: dict[str, np.ndarray] = {}
    for name, score_video in scores.items():
        weight_video = np.zeros_like(score_video)
        seed_video = np.zeros_like(score_video, dtype=bool)
        for frame_index in range(frames):
            weight_video[frame_index], seed_video[frame_index] = _postprocess_score(
                score_video[frame_index],
                static_core[frame_index],
                maxima[name],
                params,
            )
        weights[name] = weight_video
        seeds[name] = seed_video
    masks = {
        "static_core": static_core,
        "motion_core": seeds["full"],
        "residual_seed": seeds["residual_only"],
        "gradient_seed": seeds["residual_gradient"],
    }
    return weights, masks, records


def fuse_branches(
    weighted_temporal: np.ndarray,
    frame_repeated: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Blend two normalized RAW branches with one scalar weight per packed pixel."""

    if weighted_temporal.shape != frame_repeated.shape:
        raise ValueError("A3 branches must have identical shapes")
    if weights.shape != (
        weighted_temporal.shape[0],
        weighted_temporal.shape[2],
        weighted_temporal.shape[3],
    ):
        raise ValueError(f"weight shape {weights.shape} does not match RAW branches")
    output = (
        (1.0 - weights[:, None]) * weighted_temporal.astype(np.float32)
        + weights[:, None] * frame_repeated.astype(np.float32)
    )
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _regional_temporal_summary(
    video: np.ndarray,
    input_video: np.ndarray,
    motion_core: np.ndarray,
    static_core: np.ndarray,
) -> dict[str, float]:
    video_diff = np.abs(np.diff(video, axis=0)).mean(axis=1)
    input_diff = np.abs(np.diff(input_video, axis=0)).mean(axis=1)
    motion = motion_core[1:] | motion_core[:-1]
    static = static_core[1:] & static_core[:-1]

    def summarize(region: np.ndarray, prefix: str) -> dict[str, float]:
        if not region.any():
            return {
                f"{prefix}_fraction": 0.0,
                f"{prefix}_temporal_mad": 0.0,
                f"{prefix}_input_temporal_mad": 0.0,
                f"{prefix}_to_input_ratio": 0.0,
            }
        candidate_value = float(video_diff[region].mean())
        input_value = float(input_diff[region].mean())
        return {
            f"{prefix}_fraction": float(region.mean()),
            f"{prefix}_temporal_mad": candidate_value,
            f"{prefix}_input_temporal_mad": input_value,
            f"{prefix}_to_input_ratio": candidate_value / max(input_value, 1e-12),
        }

    return {**summarize(motion, "motion"), **summarize(static, "static")}


def _boundary_summary(
    candidate: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    candidate_luma = candidate.mean(axis=1)
    baseline_luma = baseline.mean(axis=1)
    boundary_values: list[np.ndarray] = []
    baseline_values: list[np.ndarray] = []
    shift_gradient_values: list[np.ndarray] = []
    boundary_fraction = 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for frame_index in range(candidate.shape[0]):
        support = (weights[frame_index] > 0.0).astype(np.uint8)
        boundary = cv2.morphologyEx(support, cv2.MORPH_GRADIENT, kernel).astype(bool)
        boundary_fraction += float(boundary.mean())
        if not boundary.any():
            continue
        candidate_gradient = _gradient_magnitude(candidate_luma[frame_index])
        baseline_gradient = _gradient_magnitude(baseline_luma[frame_index])
        shift_gradient = _gradient_magnitude(
            candidate_luma[frame_index] - baseline_luma[frame_index]
        )
        boundary_values.append(candidate_gradient[boundary])
        baseline_values.append(baseline_gradient[boundary])
        shift_gradient_values.append(shift_gradient[boundary])
    if not boundary_values:
        return {
            "boundary_fraction": 0.0,
            "candidate_gradient_mean": 0.0,
            "baseline_gradient_mean": 0.0,
            "candidate_to_baseline_gradient_ratio": 0.0,
            "shift_gradient_mean": 0.0,
            "shift_gradient_p99": 0.0,
        }
    candidate_gradient = np.concatenate(boundary_values)
    baseline_gradient = np.concatenate(baseline_values)
    shift_gradient = np.concatenate(shift_gradient_values)
    return {
        "boundary_fraction": boundary_fraction / candidate.shape[0],
        "candidate_gradient_mean": float(candidate_gradient.mean()),
        "baseline_gradient_mean": float(baseline_gradient.mean()),
        "candidate_to_baseline_gradient_ratio": float(candidate_gradient.mean())
        / max(float(baseline_gradient.mean()), 1e-12),
        "shift_gradient_mean": float(shift_gradient.mean()),
        "shift_gradient_p99": float(np.percentile(shift_gradient, 99)),
    }


def evaluate_gate(
    variants: dict[str, np.ndarray],
    ablation_weights: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    black: np.ndarray,
) -> dict[str, Any]:
    """Evaluate branch fidelity, contamination, temporal behavior, and boundaries."""

    result = evaluate_variants(
        variants,
        {"baseline_64_56": (64, 56), "weighted_64_48": (64, 48)},
    )
    baseline = variants["a3_weighted"]
    scale = (4095.0 - black)[None, :, None, None]
    result["mask_ablations"] = {}
    for name, weights in ablation_weights.items():
        candidate = fuse_branches(baseline, variants["a3_frame_repeated"], weights)
        shift_dn = np.abs(candidate - baseline) * scale
        static = np.broadcast_to(masks["static_core"][:, None], shift_dn.shape)
        result["mask_ablations"][name] = {
            "weight_mean": float(weights.mean()),
            "coverage_w_gt_0": float((weights > 0.0).mean()),
            "coverage_w_ge_0_5": float((weights >= 0.5).mean()),
            "static_nonzero_weight_fraction": float(
                (weights[masks["static_core"]] > 0.0).mean()
            )
            if masks["static_core"].any()
            else 0.0,
            "shift_dn_all": _distribution(shift_dn.reshape(-1)),
            "shift_dn_static": _distribution(shift_dn[static]),
            "temporal_regions": _regional_temporal_summary(
                candidate,
                variants["input"],
                masks["motion_core"],
                masks["static_core"],
            ),
            "boundary": _boundary_summary(candidate, baseline, weights),
        }
    result["mask_regions"] = {
        "motion_core_fraction": float(masks["motion_core"].mean()),
        "static_core_fraction": float(masks["static_core"].mean()),
    }
    return result


def _resize_cell(image: np.ndarray, label: str, detail: bool) -> np.ndarray:
    if detail:
        height, width = image.shape[:2]
        image = image[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    cell = cv2.resize(image, (640, 540), interpolation=cv2.INTER_AREA)
    cell = np.ascontiguousarray(cell)
    cv2.rectangle(cell, (0, 0), (640, 40), (10, 10, 10), -1)
    cv2.putText(
        cell,
        label,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return cell


def _weight_image(weight: np.ndarray) -> np.ndarray:
    normalized = np.rint(np.clip(weight / 0.65, 0.0, 1.0) * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _open_encoder(
    output_path: Path,
    fps: int,
    preferred_encoder: str,
    crf: int,
    pixel_format: str,
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


def render_gate_videos(
    output_dir: Path,
    sequence: PreviewSequence,
    source_start: int,
    black: np.ndarray,
    exposure_ev: float,
    variants: dict[str, np.ndarray],
    weights: np.ndarray,
    isp_config: dict[str, Any],
    video_config: dict[str, Any],
) -> dict[str, Any]:
    """Render five fixed-ISP branches plus the A11 confidence weight."""

    order = ("input", "a3_uniform", "a3_weighted", "a3_frame_repeated", "a11_gated")
    if any(name not in variants for name in order):
        raise ValueError("five fixed-ISP variants are required")
    labels = {
        "input": "Input",
        "a3_uniform": "A3 uniform 64/56",
        "a3_weighted": "A3 weighted 64/48",
        "a3_frame_repeated": "A3 frame-repeat 64/48",
        "a11_gated": "A11 confidence-gated",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection_dir = output_dir / "inspection_frames"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    video_paths = {
        kind: output_dir / f"{sequence.name}_a11_5way_{kind}.mp4"
        for kind in ("overview", "detail")
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
    inspection_paths: dict[str, list[Path]] = {"overview": [], "detail": []}
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
            rendered["weight"] = _weight_image(weights[frame_index])
            marker = f"source frame {source_start + frame_index}"
            for kind, detail in (("overview", False), ("detail", True)):
                cells = [
                    _resize_cell(rendered[name], f"{labels[name]} | {marker}", detail)
                    for name in order
                ]
                cells.append(
                    _resize_cell(rendered["weight"], f"A11 weight | {marker}", detail)
                )
                panel = np.concatenate(
                    (
                        np.concatenate(cells[:3], axis=1),
                        np.concatenate(cells[3:], axis=1),
                    ),
                    axis=0,
                )
                if panel.shape != (1080, 1920, 3):
                    raise ValueError(f"unexpected panel shape {panel.shape}")
                inspection_path = inspection_dir / f"{kind}_frame{frame_index:04d}.png"
                if not cv2.imwrite(
                    str(inspection_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                ):
                    raise OSError(f"failed to write {inspection_path}")
                inspection_paths[kind].append(inspection_path)
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
        for process, _, _ in encoders.values():
            process.kill()
            process.wait()
        raise
    manifest = {
        "sequence": sequence.to_mapping(),
        "source_start": source_start,
        "frames": int(variants["input"].shape[0]),
        "layout": "2x3: five fixed-ISP branches plus A11 confidence weight",
        "videos": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "ffmpeg_command": encoders[name][1],
                "encoder": encoders[name][2],
            }
            for name, path in video_paths.items()
        },
        "inspection_frames": {
            name: [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in paths
            ]
            for name, paths in inspection_paths.items()
        },
        "isp": {
            **isp_config,
            "fixed_exposure_ev": exposure_ev,
            "black_levels_R_Gr_Gb_B": [float(value) for value in black],
            "wb_rgb": list(sequence.wb_rgb),
            "group03_2dnr_enabled": False,
            "group03_3dnr_enabled": False,
        },
    }
    save_json(output_dir / "video_manifest.json", manifest)
    return manifest

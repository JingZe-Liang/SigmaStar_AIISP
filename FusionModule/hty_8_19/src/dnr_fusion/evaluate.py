from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .confidence import safety_confidence
from .config import load_config, project_root, validate_scene
from .dataset_fast import load_packed_frame, open_scene_streams, scene_safety_params
from .raw_io import RawSpec, RawStream, pack_rggb


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def _masked_sum(values: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    count = int(mask.sum())
    return (float(values[mask].sum()) if count else 0.0, count)


def _ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


@torch.inference_mode()
def evaluate_scene(
    config: dict[str, Any],
    scene_name: str,
    output_raw: Path,
    gate_raw: Path,
    report_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    validate_scene(config, scene_name)
    streams = open_scene_streams(config, scene_name)
    raw = config["raw"]
    spec = RawSpec(width=raw["width"], height=raw["height"])
    output_stream = RawStream(output_raw, spec)
    if output_stream.frame_count != streams.frame_count:
        raise ValueError("Fused output frame count does not match input streams")

    gate_shape = (streams.frame_count, raw["height"] // 2, raw["width"] // 2)
    expected_gate_bytes = int(np.prod(gate_shape))
    if gate_raw.stat().st_size != expected_gate_bytes:
        raise ValueError(
            f"Gate stream has {gate_raw.stat().st_size} bytes; expected {expected_gate_bytes}"
        )
    gates = np.memmap(gate_raw, dtype=np.uint8, mode="r", shape=gate_shape)
    safety = scene_safety_params(config, scene_name)

    temporal_sums = {"2dnr": 0.0, "3dnr": 0.0, "learned_fusion": 0.0}
    temporal_count = 0
    motion_output_deviation_sum = 0.0
    motion_3dnr_deviation_sum = 0.0
    motion_count = 0
    motion_exact_count = 0
    static_to_3dnr_sum = 0.0
    static_to_2dnr_sum = 0.0
    static_count = 0
    static_gate_sum = 0.0
    motion_gate_sum = 0.0
    previous = None

    for frame_index in tqdm(range(streams.frame_count), desc=f"evaluate {scene_name}"):
        _, denoised_norm, fused_norm = load_packed_frame(streams, frame_index, raw)
        denoised_dn = pack_rggb(streams.denoised.frame(frame_index)).astype(np.float32)
        fused_dn = pack_rggb(streams.fused.frame(frame_index)).astype(np.float32)
        output_dn = pack_rggb(output_stream.frame(frame_index)).astype(np.float32)
        gate = np.asarray(gates[frame_index], dtype=np.float32) / 255.0

        if previous is None:
            previous_norm = denoised_norm
        else:
            previous_norm = previous["denoised_norm"]
        confidence, diagnostics = safety_confidence(
            torch.from_numpy(denoised_norm).unsqueeze(0).to(device),
            torch.from_numpy(previous_norm).unsqueeze(0).to(device),
            torch.from_numpy(fused_norm).unsqueeze(0).to(device),
            safety,
        )
        confidence_np = confidence.squeeze().cpu().numpy()
        hard_mask = diagnostics["hard_mask"].squeeze().bool().cpu().numpy()
        static_mask = confidence_np >= 0.8

        if frame_index > 0 and previous is not None:
            static_temporal_mask = np.logical_and(static_mask, previous["static_mask"])
            count = int(static_temporal_mask.sum())
            if count:
                for name, current, prior in (
                    ("2dnr", denoised_dn, previous["denoised_dn"]),
                    ("3dnr", fused_dn, previous["fused_dn"]),
                    ("learned_fusion", output_dn, previous["output_dn"]),
                ):
                    difference = np.mean(np.abs(current - prior), axis=0)
                    temporal_sums[name] += float(difference[static_temporal_mask].sum())
                temporal_count += count

        output_deviation = np.mean(np.abs(output_dn - denoised_dn), axis=0)
        candidate_deviation = np.mean(np.abs(fused_dn - denoised_dn), axis=0)
        current_motion_sum, current_motion_count = _masked_sum(
            output_deviation, hard_mask
        )
        motion_output_deviation_sum += current_motion_sum
        candidate_motion_sum, _ = _masked_sum(candidate_deviation, hard_mask)
        motion_3dnr_deviation_sum += candidate_motion_sum
        motion_count += current_motion_count
        if current_motion_count:
            exact_map = np.all(output_dn == denoised_dn, axis=0)
            motion_exact_count += int(np.logical_and(exact_map, hard_mask).sum())
            motion_gate_sum += float(gate[hard_mask].sum())

        static_3dnr_sum, current_static_count = _masked_sum(
            np.mean(np.abs(output_dn - fused_dn), axis=0), static_mask
        )
        static_2dnr_sum, _ = _masked_sum(output_deviation, static_mask)
        static_to_3dnr_sum += static_3dnr_sum
        static_to_2dnr_sum += static_2dnr_sum
        static_count += current_static_count
        if current_static_count:
            static_gate_sum += float(gate[static_mask].sum())

        previous = {
            "denoised_norm": denoised_norm,
            "denoised_dn": denoised_dn,
            "fused_dn": fused_dn,
            "output_dn": output_dn,
            "static_mask": static_mask,
        }

    gate_sample = np.asarray(gates[:, ::4, ::4], dtype=np.float32) / 255.0
    metrics = {
        "methodology": {
            "has_clean_ground_truth": False,
            "psnr_ssim_reported": False,
            "reason": "The company sequences contain source/2DNR/3DNR only; no clean reference is available.",
        },
        "scene": scene_name,
        "frames": streams.frame_count,
        "static_temporal_difference_mae_dn": {
            key: _ratio(value, temporal_count) for key, value in temporal_sums.items()
        },
        "hard_motion_region": {
            "pixels": motion_count,
            "learned_fusion_deviation_from_2dnr_dn": _ratio(
                motion_output_deviation_sum, motion_count
            ),
            "3dnr_deviation_from_2dnr_dn": _ratio(
                motion_3dnr_deviation_sum, motion_count
            ),
            "exact_2dnr_fallback_fraction": _ratio(
                motion_exact_count, motion_count
            ),
            "mean_gate": _ratio(motion_gate_sum, motion_count),
        },
        "static_high_confidence_region": {
            "pixels": static_count,
            "learned_fusion_deviation_from_3dnr_dn": _ratio(
                static_to_3dnr_sum, static_count
            ),
            "learned_fusion_deviation_from_2dnr_dn": _ratio(
                static_to_2dnr_sum, static_count
            ),
            "mean_gate": _ratio(static_gate_sum, static_count),
        },
        "gate_quantiles": {
            str(q): float(np.quantile(gate_sample, q))
            for q in (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--scene", choices=("645x", "128x"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root(config) / "outputs"
    input_path = args.input or root / "raw" / f"{args.scene}_learned_fusion.raw"
    gate_path = args.gate or root / "gates" / f"{args.scene}_gate_u8.raw"
    output_path = args.output or root / "metrics" / f"{args.scene}_evaluation.json"
    metrics = evaluate_scene(
        config,
        args.scene,
        input_path.resolve(),
        gate_path.resolve(),
        output_path.resolve(),
        torch.device(args.device),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


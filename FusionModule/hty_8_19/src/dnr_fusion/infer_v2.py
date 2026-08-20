from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .confidence import conservative_gate, safety_confidence
from .config import load_config, project_root, validate_scene
from .dataset_fast import load_packed_frame, open_scene_streams, scene_safety_params
from .features_v2 import build_threshold_normalized_features
from .infer import _prepare_output, load_model
from .raw_io import pack_rggb, unpack_rggb


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def stabilize_gate(
    instantaneous: torch.Tensor,
    previous_gate: torch.Tensor | None,
    hard_mask: torch.Tensor,
    rise_alpha: float,
    fall_alpha: float,
) -> torch.Tensor:
    if previous_gate is None:
        stable = instantaneous
    else:
        alpha = torch.where(
            instantaneous < previous_gate,
            torch.full_like(instantaneous, fall_alpha),
            torch.full_like(instantaneous, rise_alpha),
        )
        stable = previous_gate + alpha * (instantaneous - previous_gate)
    return torch.where(hard_mask > 0, torch.zeros_like(stable), stable).clamp_(0.0, 1.0)


@torch.inference_mode()
def infer_scene_v2(
    config: dict[str, Any],
    scene_name: str,
    checkpoint_path: Path,
    output_path: Path,
    gate_path: Path,
    stats_path: Path,
    device: torch.device,
    overwrite: bool,
    rise_alpha: float,
    fall_alpha: float,
) -> dict[str, Any]:
    validate_scene(config, scene_name)
    streams = open_scene_streams(config, scene_name)
    model, checkpoint = load_model(checkpoint_path, device)
    safety = scene_safety_params(config, scene_name)
    raw = config["raw"]
    use_amp = bool(config["inference"].get("use_amp", True)) and device.type == "cuda"
    output_temporary = _prepare_output(output_path, overwrite)
    gate_temporary = _prepare_output(gate_path, overwrite)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    output_hash = hashlib.sha256()
    gate_hash = hashlib.sha256()
    frame_stats: list[dict[str, float | int]] = []
    previous_2dnr_np: np.ndarray | None = None
    previous_gate: torch.Tensor | None = None
    started = time.perf_counter()

    with output_temporary.open("wb") as output_handle, gate_temporary.open("wb") as gate_handle:
        for frame_index in tqdm(range(streams.frame_count), desc=f"infer v2 {scene_name}"):
            source_np, denoised_np, fused_np = load_packed_frame(streams, frame_index, raw)
            if previous_2dnr_np is None:
                previous_2dnr_np = denoised_np
            current_source = torch.from_numpy(source_np).unsqueeze(0).to(device)
            current_2dnr = torch.from_numpy(denoised_np).unsqueeze(0).to(device)
            current_3dnr = torch.from_numpy(fused_np).unsqueeze(0).to(device)
            prior_2dnr = torch.from_numpy(previous_2dnr_np).unsqueeze(0).to(device)
            features = build_threshold_normalized_features(
                current_source, current_2dnr, current_3dnr, prior_2dnr, safety
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                predicted_gate = model.gate(features)
            confidence, diagnostics = safety_confidence(
                current_2dnr.float(), prior_2dnr.float(), current_3dnr.float(), safety
            )
            instantaneous = conservative_gate(predicted_gate.float(), confidence)
            if frame_index == 0:
                gate = torch.zeros_like(instantaneous)
            else:
                gate = stabilize_gate(
                    instantaneous,
                    previous_gate,
                    diagnostics["hard_mask"],
                    rise_alpha,
                    fall_alpha,
                )
                previous_gate = gate

            denoised_dn = pack_rggb(streams.denoised.frame(frame_index)).astype(np.float32)
            fused_dn = pack_rggb(streams.fused.frame(frame_index)).astype(np.float32)
            gate_np = gate.squeeze(0).cpu().numpy()
            output_dn = np.rint(denoised_dn + gate_np * (fused_dn - denoised_dn))
            output_packed = np.clip(output_dn, 0, int(raw["white"])).astype(np.uint16)
            output_mosaic = unpack_rggb(output_packed).astype("<u2", copy=False)
            output_bytes = output_mosaic.tobytes(order="C")
            output_handle.write(output_bytes)
            output_hash.update(output_bytes)

            gate_u8 = np.rint(gate_np.squeeze(0) * 255.0).astype(np.uint8)
            gate_bytes = gate_u8.tobytes(order="C")
            gate_handle.write(gate_bytes)
            gate_hash.update(gate_bytes)
            frame_stats.append(
                {
                    "frame": frame_index,
                    "predicted_gate_mean": float(predicted_gate.float().mean()),
                    "instantaneous_gate_mean": float(instantaneous.mean()),
                    "final_gate_mean": float(gate.mean()),
                    "hard_fallback_fraction": float((diagnostics["hard_mask"] > 0).float().mean()),
                }
            )
            previous_2dnr_np = denoised_np

    output_temporary.replace(output_path)
    gate_temporary.replace(gate_path)
    elapsed = time.perf_counter() - started
    summary = {
        "version": 2,
        "scene": scene_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_fold": checkpoint.get("fold"),
        "frames": streams.frame_count,
        "output_raw": str(output_path.resolve()),
        "output_sha256": output_hash.hexdigest(),
        "gate_raw": str(gate_path.resolve()),
        "gate_sha256": gate_hash.hexdigest(),
        "fusion_domain": "native candidate DN",
        "gate_stabilization": {"rise_alpha": rise_alpha, "fall_alpha": fall_alpha},
        "elapsed_seconds": elapsed,
        "frames_per_second": streams.frame_count / elapsed,
        "device": str(device),
        "frame_statistics": frame_stats,
    }
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--scene", choices=("645x", "128x"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rise-alpha", type=float, default=0.08)
    parser.add_argument("--fall-alpha", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root(config) / "outputs"
    output = args.output or root / "raw" / f"{args.scene}_learned_fusion.raw"
    gate_output = args.gate_output or root / "gates" / f"{args.scene}_gate_u8.raw"
    stats_output = args.stats_output or root / "metrics" / f"{args.scene}_inference.json"
    summary = infer_scene_v2(
        config,
        args.scene,
        args.checkpoint.resolve(),
        output.resolve(),
        gate_output.resolve(),
        stats_output.resolve(),
        torch.device(args.device),
        args.overwrite,
        args.rise_alpha,
        args.fall_alpha,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "frame_statistics"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

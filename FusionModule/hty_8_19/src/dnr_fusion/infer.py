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

from .confidence import conservative_gate, fuse_candidates, safety_confidence
from .config import load_config, project_root, validate_scene
from .dataset_fast import (
    build_features,
    load_packed_frame,
    open_scene_streams,
    scene_safety_params,
)
from .model import SafeGateUNet
from .raw_io import unpack_rggb


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SafeGateUNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_info = checkpoint["model"]
    model = SafeGateUNet(
        input_channels=int(model_info["input_channels"]),
        width=int(model_info["width"]),
    ).to(device)
    model.load_state_dict(model_info["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    if temporary.exists():
        temporary.unlink()
    return temporary


@torch.inference_mode()
def infer_scene(
    config: dict[str, Any],
    scene_name: str,
    checkpoint_path: Path,
    output_path: Path,
    gate_path: Path,
    stats_path: Path,
    device: torch.device,
    overwrite: bool,
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

    frame_stats: list[dict[str, float | int]] = []
    output_hash = hashlib.sha256()
    gate_hash = hashlib.sha256()
    started = time.perf_counter()
    previous_2dnr_np: np.ndarray | None = None

    try:
        with output_temporary.open("wb") as output_handle, gate_temporary.open("wb") as gate_handle:
            for frame_index in tqdm(range(streams.frame_count), desc=f"infer {scene_name}"):
                source_np, denoised_np, fused_np = load_packed_frame(
                    streams, frame_index, raw
                )
                if previous_2dnr_np is None:
                    previous_2dnr_np = denoised_np

                current_source = torch.from_numpy(source_np).unsqueeze(0).to(device)
                current_2dnr = torch.from_numpy(denoised_np).unsqueeze(0).to(device)
                current_3dnr = torch.from_numpy(fused_np).unsqueeze(0).to(device)
                previous_2dnr = (
                    torch.from_numpy(previous_2dnr_np).unsqueeze(0).to(device)
                )
                features = build_features(
                    current_source, current_2dnr, current_3dnr, previous_2dnr
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    predicted_gate = model.gate(features)
                confidence, diagnostics = safety_confidence(
                    current_2dnr.float(),
                    previous_2dnr.float(),
                    current_3dnr.float(),
                    safety,
                )
                gate = conservative_gate(predicted_gate.float(), confidence)
                if frame_index == 0:
                    gate.zero_()
                output = fuse_candidates(current_2dnr.float(), current_3dnr.float(), gate)

                output_dn = torch.round(
                    output * float(raw["white"] - raw["candidate_black"])
                    + float(raw["candidate_black"])
                ).clamp_(0, raw["white"])
                output_packed = output_dn.squeeze(0).to(torch.uint16).cpu().numpy()
                output_mosaic = unpack_rggb(output_packed).astype("<u2", copy=False)
                output_bytes = output_mosaic.tobytes(order="C")
                output_handle.write(output_bytes)
                output_hash.update(output_bytes)

                gate_u8 = (
                    torch.round(gate.squeeze(0).squeeze(0) * 255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                gate_bytes = gate_u8.tobytes(order="C")
                gate_handle.write(gate_bytes)
                gate_hash.update(gate_bytes)

                hard_mask = diagnostics["hard_mask"] > 0
                static_mask = confidence >= 0.8
                frame_stats.append(
                    {
                        "frame": frame_index,
                        "predicted_gate_mean": float(predicted_gate.float().mean()),
                        "final_gate_mean": float(gate.mean()),
                        "confidence_mean": float(confidence.mean()),
                        "hard_fallback_fraction": float(hard_mask.float().mean()),
                        "static_fraction": float(static_mask.float().mean()),
                    }
                )
                previous_2dnr_np = denoised_np

        output_temporary.replace(output_path)
        gate_temporary.replace(gate_path)
    except Exception:
        raise

    elapsed = time.perf_counter() - started
    summary = {
        "scene": scene_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_fold": checkpoint.get("fold"),
        "frames": streams.frame_count,
        "width": raw["width"],
        "height": raw["height"],
        "packed_gate_width": raw["width"] // 2,
        "packed_gate_height": raw["height"] // 2,
        "output_raw": str(output_path.resolve()),
        "output_sha256": output_hash.hexdigest(),
        "gate_raw": str(gate_path.resolve()),
        "gate_sha256": gate_hash.hexdigest(),
        "elapsed_seconds": elapsed,
        "frames_per_second": streams.frame_count / elapsed,
        "device": str(device),
        "frame_statistics": frame_stats,
    }
    stats_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--scene", choices=("645x", "128x"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = project_root(config) / "outputs"
    output = args.output or output_root / "raw" / f"{args.scene}_learned_fusion.raw"
    gate_output = args.gate_output or output_root / "gates" / f"{args.scene}_gate_u8.raw"
    stats_output = args.stats_output or output_root / "metrics" / f"{args.scene}_inference.json"
    summary = infer_scene(
        config,
        args.scene,
        args.checkpoint.resolve(),
        output.resolve(),
        gate_output.resolve(),
        stats_output.resolve(),
        torch.device(args.device),
        args.overwrite,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frame_statistics"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


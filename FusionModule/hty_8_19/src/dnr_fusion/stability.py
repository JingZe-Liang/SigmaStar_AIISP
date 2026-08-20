from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .confidence import safety_confidence
from .config import load_config, project_root, validate_scene
from .dataset_fast import load_packed_frame, open_scene_streams, scene_safety_params


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {str(q): float("nan") for q in (0.5, 0.9, 0.95, 0.99, 1.0)}
    array = np.asarray(values, dtype=np.float32)
    return {str(q): float(np.quantile(array, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)}


@torch.inference_mode()
def analyze_scene(
    config: dict[str, Any], scene_name: str, device: torch.device
) -> dict[str, Any]:
    validate_scene(config, scene_name)
    streams = open_scene_streams(config, scene_name)
    raw = config["raw"]
    safety = scene_safety_params(config, scene_name)
    gate_path = project_root(config) / "outputs" / "gates" / f"{scene_name}_gate_u8.raw"
    shape = (streams.frame_count, raw["height"] // 2, raw["width"] // 2)
    gates = np.memmap(gate_path, dtype=np.uint8, mode="r", shape=shape).astype(np.float32) / 255.0

    previous_gate: np.ndarray | None = None
    previous_static: np.ndarray | None = None
    gate_diffs: list[float] = []
    static_gate_diffs: list[float] = []
    static_pair_pixels = 0
    all_pair_pixels = 0
    frame_gate_means: list[float] = []
    for frame_index in range(streams.frame_count):
        _, current_2dnr, current_3dnr = load_packed_frame(streams, frame_index, raw)
        if frame_index == 0:
            previous_2dnr = current_2dnr
        else:
            _, previous_2dnr, _ = load_packed_frame(streams, frame_index - 1, raw)
        confidence, diagnostics = safety_confidence(
            torch.from_numpy(current_2dnr).unsqueeze(0).to(device),
            torch.from_numpy(previous_2dnr).unsqueeze(0).to(device),
            torch.from_numpy(current_3dnr).unsqueeze(0).to(device),
            safety,
        )
        static = (confidence.squeeze().cpu().numpy() >= 0.8)
        hard = diagnostics["hard_mask"].squeeze().cpu().numpy() > 0
        current_gate = gates[frame_index]
        frame_gate_means.append(float(current_gate.mean()))
        if previous_gate is not None:
            difference = np.abs(current_gate - previous_gate)
            gate_diffs.extend(difference.ravel().tolist())
            pair = static & previous_static & ~hard
            # Require a 5x5 packed neighborhood to remain static, avoiding edge leakage.
            pair_tensor = torch.from_numpy(pair.astype(np.float32))[None, None]
            pair = (F.avg_pool2d(pair_tensor, 5, stride=1, padding=2).squeeze() >= 0.999).numpy()
            static_gate_diffs.extend(difference[pair].ravel().tolist())
            static_pair_pixels += int(pair.sum())
            all_pair_pixels += int(difference.size)
        previous_gate = current_gate.copy()
        previous_static = static

    return {
        "scene": scene_name,
        "frames": streams.frame_count,
        "gate_mean_over_frames": float(np.mean(frame_gate_means)),
        "gate_frame_mean_quantiles": _quantiles(frame_gate_means),
        "all_pair_gate_abs_difference": {
            "pixels": all_pair_pixels,
            "mean": float(np.mean(gate_diffs)) if gate_diffs else float("nan"),
            "quantiles": _quantiles(gate_diffs),
        },
        "persistent_static_pair_gate_abs_difference": {
            "pixels": static_pair_pixels,
            "mean": float(np.mean(static_gate_diffs)) if static_gate_diffs else float("nan"),
            "quantiles": _quantiles(static_gate_diffs),
            "mask_definition": "confidence >= 0.8 in both adjacent frames, hard risk excluded, 5x5 packed erosion",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/company.yaml"))
    parser.add_argument("--scene", choices=("645x", "128x"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    result = analyze_scene(config, args.scene, torch.device(args.device))
    output = project_root(config) / "outputs" / "metrics" / f"{args.scene}_stability.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

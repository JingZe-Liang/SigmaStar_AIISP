"""Inference for v2.0. It needs only current noisy RAW, 2DNR, and 3DNR."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import TextureGateNet, fuse
from raw_io import RawSequence, read_packed_normalized, unpack_rggb


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_tiled(
    model: TextureGateNet,
    noisy: np.ndarray,
    base: np.ndarray,
    temporal: np.ndarray,
    device: torch.device,
    tile_size: int,
    halo: int,
) -> np.ndarray:
    height, width = base.shape[-2:]
    alpha = np.empty((1, height, width), dtype=np.float32)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        ext_top, ext_bottom = max(0, top - halo), min(height, bottom + halo)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            ext_left, ext_right = max(0, left - halo), min(width, right + halo)
            arrays = (noisy, base, temporal)
            tensors = [
                torch.from_numpy(value[:, ext_top:ext_bottom, ext_left:ext_right].copy()).unsqueeze(0).to(device)
                for value in arrays
            ]
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                value = model(*tensors)[0].float().cpu().numpy()
            alpha[:, top:bottom, left:right] = value[:, top - ext_top:bottom - ext_top, left - ext_left:right - ext_left]
    return alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--halo", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-raw", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TextureGateNet(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    height, width = int(config["height"]), int(config["width"])

    def stream(name: str) -> tuple[RawSequence, dict]:
        spec = config[name]
        return RawSequence(spec["path"], height, width, spec.get("dtype", "uint16")), spec

    noisy_stream, noisy_spec = stream("noisy")
    base_stream, base_spec = stream("base")
    temporal_stream, temporal_spec = stream("temporal")
    if len({noisy_stream.frame_count, base_stream.frame_count, temporal_stream.frame_count}) != 1:
        raise ValueError("noisy/2DNR/3DNR frame counts differ")
    frame_count = min(base_stream.frame_count, args.max_frames) if args.max_frames else base_stream.frame_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "fused_raw_frames"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(frame_count):
        def read(sequence: RawSequence, spec: dict) -> np.ndarray:
            return read_packed_normalized(sequence, index, float(spec["black_level"]), float(spec["white_level"]), int(spec.get("right_shift", 0)))
        noisy = read(noisy_stream, noisy_spec)
        base = read(base_stream, base_spec)
        temporal = read(temporal_stream, temporal_spec)
        alpha = predict_tiled(model, noisy, base, temporal, device, args.tile_size, args.halo)
        output = fuse(torch.from_numpy(base)[None], torch.from_numpy(temporal)[None], torch.from_numpy(alpha)[None])[0].numpy().clip(0.0, 1.0)
        injection = output - base
        np.save(args.output_dir / f"fused_packed_{index:04d}.npy", output)
        np.save(args.output_dir / f"alpha_{index:04d}.npy", alpha)
        if args.save_raw:
            raw12 = np.rint(output * (float(base_spec["white_level"]) - float(base_spec["black_level"])) + float(base_spec["black_level"]))
            unpack_rggb(np.clip(raw12, 0.0, 65535.0).astype(np.uint16)).tofile(raw_dir / f"out_{index:04d}.raw")
        row = {
            "frame_index": index,
            "injected_l1_ratio": float(np.abs(injection).sum() / max(np.abs(base).sum(), 1e-8)),
            "active_pixel_ratio": float((np.abs(injection).mean(axis=0) > 0.002).mean()),
            "alpha_mean": float(alpha.mean()),
        }
        rows.append(row)
        print(f"frame={index:04d} alpha={row['alpha_mean']:.4f} injection_l1={row['injected_l1_ratio']:.5f}", flush=True)
    with (args.output_dir / "inference_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

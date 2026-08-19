from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.nn import functional as F

from lite_unet_fusion.model import LiteFusionUNet
from lite_unet_fusion.raw import FusionStream, StreamPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled 2DNR-prior U-Net fusion inference to a 12-bit RAW stream.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--two-d", required=True, type=Path)
    parser.add_argument("--three-d", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if tile % 8 or not 0 <= overlap < tile:
        raise ValueError("tile must be divisible by 8 and overlap must be in [0, tile)")
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def predict_beta(model: LiteFusionUNet, inputs: np.ndarray, device: torch.device, tile: int, overlap: int) -> np.ndarray:
    _, height, width = inputs.shape
    result = np.zeros((4, height, width), dtype=np.float32)
    weights = np.zeros((1, height, width), dtype=np.float32)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    window = np.maximum(window, 0.05)[None]
    model.eval()
    with torch.no_grad():
        for top in positions(height, tile, overlap):
            for left in positions(width, tile, overlap):
                crop = inputs[:, top : top + tile, left : left + tile]
                crop_height, crop_width = crop.shape[-2:]
                if crop_height != tile or crop_width != tile:
                    crop = np.pad(crop, ((0, 0), (0, tile - crop_height), (0, tile - crop_width)), mode="reflect")
                tensor = torch.from_numpy(crop[None]).to(device)
                beta = model(tensor)[0].cpu().numpy()[:, :crop_height, :crop_width]
                local_window = window[:, :crop_height, :crop_width]
                result[:, top : top + crop_height, left : left + crop_width] += beta * local_window
                weights[:, top : top + crop_height, left : left + crop_width] += local_window
    return result / np.maximum(weights, 1e-6)


def main() -> None:
    args = parse_args()
    device = device_for(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LiteFusionUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    stream = FusionStream(StreamPaths(args.source, args.two_d, args.three_d))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(args.output, dtype="<u2", mode="w+", shape=(stream.frame_count, stream.height, stream.width))
    summary = {"frames": stream.frame_count, "checkpoint": str(args.checkpoint), "output": str(args.output), "mode": "two_d_prior_unet", "mean_2d_weight": []}
    for index in range(stream.frame_count):
        inputs, values = stream.network_input(index)
        beta = predict_beta(model, inputs, device, args.tile, args.overlap)
        output[index] = stream.output_code(index, beta)
        effective = 0.35 * beta * (1.0 - values["motion"]) * values["flatness"]
        summary["mean_2d_weight"].append(float(1.0 - effective.mean()))
        print(f"Fused {index + 1}/{stream.frame_count}", flush=True)
    output.flush()
    summary["mean_2d_weight"] = float(np.mean(summary["mean_2d_weight"]))
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Run tiled 2DNR-prior ST-Mamba fusion and write a 12-bit RAW stream."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from Mamba.mamba_2d_prior.model import Mamba2DPriorConfig, Mamba2DPriorFusionNet
from Mamba.mamba_2d_prior.raw_dataset import RawFusionStream, RawStreamConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--two-d", required=True, type=Path)
    parser.add_argument("--three-d", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=4,
        help="Number of spatial tiles fused in one model call; use 2 or 1 if CUDA memory is insufficient",
    )
    parser.add_argument("--source-black-level", type=float, default=252.0)
    parser.add_argument("--denoised-black-level", type=float, default=300.0)
    parser.add_argument("--source-container-scale", type=float, default=16.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scan-path-mode", choices=("temporal4", "temporal4_grouped", "multiscale_grouped", "8path"), default=None)
    return parser.parse_args()


def device_for(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or tile % 8 or not 0 <= overlap < tile:
        raise ValueError("tile must be positive and divisible by 8; overlap must be in [0, tile)")
    if length <= tile:
        return [0]
    stride = tile - overlap
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def tiled_prediction(
    model: Mamba2DPriorFusionNet,
    values: dict[str, np.ndarray],
    device: torch.device,
    tile: int,
    overlap: int,
    tile_batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if tile_batch_size <= 0:
        raise ValueError(f"tile_batch_size must be positive, got {tile_batch_size}")
    _, height, width = values["curr4"].shape
    # Keep accumulation on the device.  This avoids a device synchronization and
    # a GPU->CPU copy for every tile; only the completed frame is copied back.
    prediction_sum = torch.zeros((4, height, width), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32), 0.05)[None]
    window_device = torch.from_numpy(window).to(device=device)
    model.eval()
    correction_sum = torch.zeros(4, dtype=torch.float32, device=device)
    correction_count = 0
    tile_items: list[tuple[int, int, int, int, dict[str, np.ndarray]]] = []

    def flush_batch(batch: list[tuple[int, int, int, int, dict[str, np.ndarray]]]) -> None:
        nonlocal correction_count
        if not batch:
            return
        # Pack all seven aligned inputs into one contiguous host buffer.  This
        # reduces seven small transfer launches to one transfer per tile batch;
        # the slices below are views and do not copy on the device.
        input_keys = tuple(batch[0][4].keys())
        host_batch = np.stack(
            [np.stack([item[4][key] for key in input_keys], axis=0) for item in batch],
            axis=0,
        )
        device_batch = torch.from_numpy(host_batch).to(device)
        tensor = {key: device_batch[:, key_index] for key_index, key in enumerate(input_keys)}
        output = model(**tensor)
        for batch_index, (top, left, crop_height, crop_width, _) in enumerate(batch):
            local_weight = window_device[:, :crop_height, :crop_width]
            local = output.prediction[batch_index, :, :crop_height, :crop_width]
            prediction_sum[:, top : top + crop_height, left : left + crop_width] += local * local_weight
            weight_sum[:, top : top + crop_height, left : left + crop_width] += local_weight
        # Average spatially for each sample/plane, then sum samples.  Do not
        # reduce the Bayer-plane dimension, which is reported separately.
        correction_sum.add_(output.weight_3d.mean(dim=(2, 3)).sum(dim=0))
        correction_count += len(batch)

    with torch.inference_mode():
        for top in starts(height, tile, overlap):
            for left in starts(width, tile, overlap):
                crop = {
                    key: value[:, top : top + tile, left : left + tile]
                    for key, value in values.items()
                    if key != "teacher_w3"
                }
                crop_height, crop_width = crop["curr4"].shape[-2:]
                if crop_height != tile or crop_width != tile:
                    crop = {
                        key: np.pad(
                            value,
                            ((0, 0), (0, tile - crop_height), (0, tile - crop_width)),
                            mode="reflect",
                        )
                        for key, value in crop.items()
                    }
                tile_items.append((top, left, crop_height, crop_width, crop))
                if len(tile_items) >= tile_batch_size:
                    flush_batch(tile_items)
                    tile_items.clear()
        flush_batch(tile_items)

    prediction = (prediction_sum / weight_sum.clamp_min(1e-6)).cpu().numpy()
    mean_weight = (correction_sum / max(1, correction_count)).cpu().numpy()
    return prediction, mean_weight


def main() -> None:
    args = parse_args()
    device = device_for(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = dict(checkpoint["model_config"])
    # Checkpoints created before scan_path_mode was introduced used the
    # original eight-path backbone.
    model_config.setdefault("scan_path_mode", "8path")
    if args.scan_path_mode is not None:
        model_config["scan_path_mode"] = args.scan_path_mode
    config = Mamba2DPriorConfig(**model_config)
    model = Mamba2DPriorFusionNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    stream = RawFusionStream(
        RawStreamConfig(
            args.source,
            args.two_d,
            args.three_d,
            source_black_level=args.source_black_level,
            denoised_black_level=args.denoised_black_level,
            source_container_scale=args.source_container_scale,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(args.output, dtype="<u2", mode="w+", shape=(stream.frame_count, stream.config.height, stream.config.width))
    correction_weights: list[np.ndarray] = []
    for index in range(stream.frame_count):
        prediction, mean_weight = tiled_prediction(
            model,
            stream.sample(index),
            device,
            args.tile,
            args.overlap,
            args.tile_batch_size,
        )
        output[index] = stream.unpack_to_codes(prediction)
        correction_weights.append(mean_weight)
        print(f"Fused {index + 1}/{stream.frame_count}", flush=True)
    output.flush()
    mean_weights = np.mean(correction_weights, axis=0)
    report = {
        "output": str(args.output),
        "frames": stream.frame_count,
        "tile": args.tile,
        "overlap": args.overlap,
        "tile_batch_size": args.tile_batch_size,
        "mean_3dnr_correction_weight": float(mean_weights.mean()),
        "mean_3dnr_correction_weight_by_plane": {"R": float(mean_weights[0]), "G1": float(mean_weights[1]), "G2": float(mean_weights[2]), "B": float(mean_weights[3])},
        "mean_2dnr_weight": float(1.0 - mean_weights.mean()),
        "max_3dnr_weight": config.max_3dnr_weight,
        "checkpoint": str(args.checkpoint),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

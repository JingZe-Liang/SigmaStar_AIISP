"""Run FGRF-Net and report the fraction of the frame changed by injection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import build_dataset
from model import FGRFNet
from raw_io import unpack_rggb


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_frame_raw(
    packed: torch.Tensor,
    path: Path,
    black_level: float,
    white_level: float,
) -> None:
    packed_np = packed.detach().cpu().numpy()
    raw12 = np.rint(packed_np * (white_level - black_level) + black_level)
    raw16 = np.clip(raw12, 0.0, 65535.0).astype(np.uint16)
    unpack_rggb(raw16).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_example_645.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="inference_output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--active-threshold", type=float, default=0.002)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    dataset = build_dataset(config, training=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FGRFNet(
        input_channels=12,
        base_channels=int(config.get("model", {}).get("base_channels", 24)),
        threshold_floor=float(config.get("model", {}).get("threshold_floor", 0.008)),
        initial_threshold=float(config.get("model", {}).get("initial_threshold", 0.01)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "fused_raw_frames"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        frame_limit = len(dataset) if args.max_frames <= 0 else min(len(dataset), args.max_frames)
        for item in range(frame_limit):
            sample = dataset[item]
            index = int(sample["frame_index"])
            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in sample.items()
                if torch.is_tensor(value)
            }
            prediction = model(
                batch["base"],
                batch["temporal_residual"],
                batch["noisy_residual"],
                batch["static_mask"],
            )
            output = prediction["output"]
            base = batch["base"]
            injected = prediction["injected_residual"]
            base_abs = base.abs().sum().item()
            injected_abs = injected.abs().sum().item()
            base_l2 = torch.sqrt((base * base).sum()).item()
            injected_l2 = torch.sqrt((injected * injected).sum()).item()
            active = (injected.abs().mean(dim=1) > args.active_threshold).float().mean().item()
            row = {
                "frame_index": index,
                "injected_l1_ratio": injected_abs / max(base_abs, 1e-8),
                "injected_l2_ratio": injected_l2 / max(base_l2, 1e-8),
                "active_pixel_ratio": active,
                "mean_gate": prediction["gates"].mean().item(),
                "static_pixel_ratio": batch["static_mask"].mean().item(),
            }
            rows.append(row)
            np.save(output_dir / f"fused_packed_{index:04d}.npy", output[0].cpu().numpy())
            if args.save_raw:
                save_frame_raw(
                    output[0],
                    raw_dir / f"out_{index:04d}.raw",
                    black_level=float(config["base"].get("black_level", 300.0)),
                    white_level=float(config["base"].get("white_level", 4095.0)),
                )
            print(
                f"frame={index:04d} injected_l1={row['injected_l1_ratio']:.6f} "
                f"active_pixels={row['active_pixel_ratio']:.6f}"
            )

    with (output_dir / "injection_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "injection_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        print(
            "mean "
            f"injected_l1={np.mean([row['injected_l1_ratio'] for row in rows]):.6f} "
            f"active_pixels={np.mean([row['active_pixel_ratio'] for row in rows]):.6f}"
        )


if __name__ == "__main__":
    main()

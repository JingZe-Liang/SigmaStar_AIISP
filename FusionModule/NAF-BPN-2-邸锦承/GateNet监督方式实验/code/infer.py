from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from data import (
    FRAME_COUNT,
    HEIGHT,
    WIDTH,
    NR_BLACK_LEVEL,
    _expand_cfa_values,
    discover_sequences,
    estimate_sequence_statistics,
    linear_to_nr_raw,
    raw_to_linear,
    read_candidate,
    read_source,
)
from model import NAFBPNMotionFusionNet, extract_model_state, forward_padded


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无外部 MD 的 NAF-BPN Bayer 推理")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", choices=("128x", "645x", "all"), default="all")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=512, help="Bayer tile size; 0 means full frame")
    parser.add_argument("--halo", type=int, default=32, help="Bayer halo around each tile")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _predict(model: torch.nn.Module, arrays: tuple[np.ndarray, ...], device: torch.device, amp: bool) -> np.ndarray:
    tensors = [
        torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).unsqueeze(0).to(device)
        for array in arrays
    ]
    with torch.inference_mode(), _autocast(device, amp):
        return forward_padded(model, *tensors)[0, 0].float().cpu().numpy()


def infer_tiled(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    tile_size: int,
    halo: int,
    amp: bool,
) -> np.ndarray:
    if tile_size == 0:
        return _predict(model, arrays, device, amp)
    if tile_size <= 0 or halo < 1:
        raise ValueError("tile-size 必须为 0 或正数，halo 必须至少为 1")
    height, width = arrays[0].shape
    if any(array.shape != (height, width) for array in arrays):
        raise ValueError("所有推理输入尺寸必须一致")
    output = np.empty((height, width), dtype=np.float32)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        ext_top, ext_bottom = max(0, top - halo), min(height, bottom + halo)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            ext_left, ext_right = max(0, left - halo), min(width, right + halo)
            tile_arrays = tuple(array[ext_top:ext_bottom, ext_left:ext_right] for array in arrays)
            tile_output = _predict(model, tile_arrays, device, amp)
            output[top:bottom, left:right] = tile_output[
                top - ext_top : bottom - ext_top,
                left - ext_left : right - ext_left,
            ]
    return output


def main() -> int:
    args = parse_args()
    if args.tile_size < 0 or args.halo < 1:
        raise ValueError("tile-size 必须为 0 或正数，halo 必须至少为 1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 要求 CUDA，但当前 PyTorch 未检测到 CUDA")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model = NAFBPNMotionFusionNet(
        num_basis=int(model_config.get("num_basis", config.get("num_basis", 15))),
        kernel_size=int(model_config.get("kernel_size", config.get("kernel_size", 7))),
        width=int(model_config.get("model_width", config.get("model_width", 32))),
    ).to(device)
    model.load_state_dict(extract_model_state(payload), strict=True)
    model.eval()

    sequences = discover_sequences(
        Path(config["data_root"]),
        tuple(config.get("sequence_names", ("128x", "645x"))),
        None,
        str(config.get("cfa_pattern", "RGGB")),
        int(config.get("source_black_level", 252)),
        int(config.get("dnr_black_level", 300)),
        int(config.get("white_level", 4095)),
    )
    if args.sequence != "all":
        sequences = tuple(sequence for sequence in sequences if sequence.name == args.sequence)
    count = FRAME_COUNT if args.frame_limit is None else min(FRAME_COUNT, max(args.frame_limit, 0))
    output_root = args.output or ROOT / "outputs_stage2" / args.checkpoint.stem
    amp_enabled = bool(args.amp and device.type == "cuda")
    for sequence in sequences:
        destination = output_root / sequence.name
        destination.mkdir(parents=True, exist_ok=True)
        statistics = estimate_sequence_statistics(sequence)
        offset_map = _expand_cfa_values(statistics.source_to_dnr_offset, sequence.cfa_pattern)
        source_scale = max(sequence.white_level - sequence.source_black_level, 1)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for frame_index in range(count):
            source_current = raw_to_linear(read_source(sequence, frame_index), sequence.source_black_level, sequence.white_level)
            source_previous = raw_to_linear(read_source(sequence, max(0, frame_index - 1)), sequence.source_black_level, sequence.white_level)
            source_current = np.clip(source_current + offset_map / source_scale, 0.0, 1.0)
            source_previous = np.clip(source_previous + offset_map / source_scale, 0.0, 1.0)
            image_2dnr = raw_to_linear(read_candidate(sequence.dnr2_paths[frame_index]), sequence.dnr_black_level, sequence.white_level)
            image_3dnr = raw_to_linear(read_candidate(sequence.dnr3_paths[frame_index]), sequence.dnr_black_level, sequence.white_level)
            prediction = infer_tiled(
                model,
                (image_2dnr, image_3dnr, source_current, source_previous),
                device,
                args.tile_size,
                args.halo,
                amp_enabled,
            )
            output = linear_to_nr_raw(
                prediction,
                int(config.get("dnr_black_level", NR_BLACK_LEVEL)),
                int(config.get("white_level", 4095)),
            )
            target = destination / f"out_{frame_index:04d}.raw"
            output.tofile(target)
            if target.stat().st_size != WIDTH * HEIGHT * 2:
                raise ValueError(f"输出大小错误: {target}")
        elapsed = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        manifest = {
            "variant": "naf_bpn_weak_stage2",
            "checkpoint": str(args.checkpoint),
            "frames": count,
            "cfa_pattern": sequence.cfa_pattern,
            "input_domain": "single-channel Bayer black-level-corrected linear",
            "output_domain": {"black_level": int(config.get("dnr_black_level", NR_BLACK_LEVEL)), "white_level": int(config.get("white_level", 4095))},
            "external_motion_detector": False,
            "motion_cache_used": False,
            "tile_size": args.tile_size,
            "halo": args.halo,
            "tile_basis_is_per_tile": args.tile_size > 0,
            "elapsed_seconds": elapsed,
            "frames_per_second": count / max(elapsed, 1e-6),
            "peak_memory_bytes": peak_memory,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    print(f"RAW 输出目录: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

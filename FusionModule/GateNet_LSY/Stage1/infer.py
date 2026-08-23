from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from dataset_io import RAW_DTYPE, RawStreamReader, discover_dataset, pack_bayer
from gatenet import GateNet, build_gate_features


def unpack_bayer(packed: np.ndarray, cfa_pattern: str) -> np.ndarray:
    """Invert pack_bayer for a full frame with origin (0, 0)."""
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected packed Bayer shape (4, H, W), got {packed.shape}")
    labels: list[str] = []
    green_index = 0
    for color in cfa_pattern:
        if color == "G":
            green_index += 1
            labels.append(f"G{green_index}")
        else:
            labels.append(color)
    planes = dict(zip(("R", "G1", "G2", "B"), packed, strict=True))
    height, width = packed.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=packed.dtype)
    for index, label in enumerate(labels):
        y, x = divmod(index, 2)
        mosaic[y::2, x::2] = planes[label]
    return mosaic


def infer_alpha_tiled(
    model: GateNet,
    arrays: tuple[np.ndarray, ...],
    motion: np.ndarray,
    noise_sigma: np.ndarray,
    *,
    device: torch.device,
    tile_size: int,
    halo: int,
    amp_enabled: bool,
) -> np.ndarray:
    height, width = motion.shape
    alpha = np.empty((height, width), dtype=np.float32)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        ext_top = max(0, top - halo)
        ext_bottom = min(height, bottom + halo)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            ext_left = max(0, left - halo)
            ext_right = min(width, right + halo)
            tensors = [
                torch.from_numpy(
                    np.ascontiguousarray(
                        array[:, ext_top:ext_bottom, ext_left:ext_right],
                        dtype=np.float32,
                    )
                ).unsqueeze(0).to(device)
                for array in arrays
            ]
            motion_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    motion[ext_top:ext_bottom, ext_left:ext_right],
                    dtype=np.float32,
                )
            ).view(1, 1, ext_bottom - ext_top, ext_right - ext_left).to(device)
            sigma_tensor = torch.from_numpy(noise_sigma).view(1, 4, 1, 1).to(device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                features = build_gate_features(*tensors, motion_tensor, sigma_tensor)
                tile_alpha = model(features)[0, 0]
            core = tile_alpha[
                top - ext_top : bottom - ext_top,
                left - ext_left : right - ext_left,
            ]
            alpha[top:bottom, left:right] = core.float().cpu().numpy()
    return alpha


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run GateNet on paired 2DNR/3DNR RAW streams")
    parser.add_argument("--dataset-root", type=Path, default=script_root / "DATASET")
    parser.add_argument("--md-root", type=Path, default=script_root / "DERIVED" / "md_mog2")
    parser.add_argument("--checkpoint", type=Path, default=script_root / "runs" / "final_all" / "best.pt")
    parser.add_argument("--output", type=Path, default=script_root / "DERIVED" / "inference_final_all")
    parser.add_argument("--sequences", nargs="+", default=["128x", "645x"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=512, help="Packed-pixel tile size")
    parser.add_argument("--halo", type=int, default=8, help="Packed-pixel context around each tile")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tile_size <= 0 or args.halo < 5:
        raise ValueError("tile-size must be positive and halo must be at least 5")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    catalog = discover_dataset(args.dataset_root)
    selected = [s for s in catalog.fusion_sequences if s.sequence_id in set(args.sequences)]
    found = {s.sequence_id for s in selected}
    if found != set(args.sequences):
        raise ValueError(f"Unknown sequence IDs: {sorted(set(args.sequences) - found)}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GateNet(**checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    config = checkpoint["config"]
    statistics = {**config.get("val_statistics", {}), **config.get("train_statistics", {})}
    black_source = float(config["black_source"])
    black_dnr = float(config["black_dnr"])
    supported_start = int(config["warmup_frames"]) + 3

    args.output.mkdir(parents=True, exist_ok=True)
    for sequence in selected:
        frame_start = args.frame_start
        frame_stop = sequence.frame_count if args.frame_stop is None else args.frame_stop
        if not 0 <= frame_start < frame_stop <= sequence.frame_count:
            raise ValueError(
                f"Invalid frame interval [{frame_start}, {frame_stop}) for {sequence.sequence_id}"
            )
        supported_stop = sequence.frame_count - 3
        stats = statistics.get(sequence.sequence_id)
        if stats is None:
            raise KeyError(f"Checkpoint has no statistics for {sequence.sequence_id}")
        offset = np.asarray(stats["source_to_dnr_offset"], dtype=np.float32).reshape(4, 1, 1)
        sigma = np.asarray(stats["noise_sigma"], dtype=np.float32)

        sequence_output = args.output / sequence.sequence_id
        sequence_output.mkdir(parents=True, exist_ok=True)
        fusion_path = sequence_output / "fusion.raw"
        alpha_path = sequence_output / "alpha_u8.raw"
        fusion_partial = fusion_path.with_suffix(".raw.partial")
        alpha_partial = alpha_path.with_suffix(".raw.partial")
        source_reader = RawStreamReader(sequence.source)
        denoised_reader = RawStreamReader(sequence.denoised)
        fused_reader = RawStreamReader(sequence.fused)
        md_path = args.md_root / sequence.sequence_id / "md_mog2.raw"
        expected_md_bytes = sequence.frame_count * 540 * 960
        if not md_path.is_file() or md_path.stat().st_size != expected_md_bytes:
            raise FileNotFoundError(f"Missing or invalid MD stream: {md_path}")
        md_stream = np.memmap(md_path, dtype=np.uint8, mode="r", shape=(sequence.frame_count, 540, 960))

        fusion_hash = hashlib.sha256()
        alpha_hash = hashlib.sha256()
        alpha_sum = 0.0
        alpha_static_sum = 0.0
        alpha_motion_sum = 0.0
        modeled_pixels = 0
        static_pixels = 0
        motion_pixels = 0
        modeled_frames = 0
        started = time.perf_counter()
        print(f"{sequence.sequence_id}: frames [{frame_start}, {frame_stop}) on {device}", flush=True)
        with fusion_partial.open("wb") as fusion_file, alpha_partial.open("wb") as alpha_file:
            for frame_index in range(frame_start, frame_stop):
                denoised_12bit = denoised_reader.read_frame(frame_index)
                motion = (md_stream[frame_index] > 0).astype(np.float32)
                if supported_start <= frame_index < supported_stop:
                    denoised = pack_bayer(denoised_12bit, sequence.denoised.cfa_pattern).astype(np.float32) - black_dnr
                    fused = pack_bayer(fused_reader.read_frame(frame_index), sequence.fused.cfa_pattern).astype(np.float32) - black_dnr

                    def source_at(index: int) -> np.ndarray:
                        packed = pack_bayer(source_reader.read_frame(index), sequence.source.cfa_pattern)
                        return packed.astype(np.float32) - black_source + offset

                    source = source_at(frame_index)
                    source_prev = source_at(frame_index - 1)
                    source_next = source_at(frame_index + 1)
                    alpha = infer_alpha_tiled(
                        model,
                        (denoised, fused, source, source_prev, source_next),
                        motion,
                        sigma,
                        device=device,
                        tile_size=args.tile_size,
                        halo=args.halo,
                        amp_enabled=device.type == "cuda",
                    )
                    output_packed = denoised + alpha[None] * (fused - denoised) + black_dnr
                    output_packed = np.clip(np.rint(output_packed), 0, 4095).astype(RAW_DTYPE)
                    output = unpack_bayer(output_packed, sequence.denoised.cfa_pattern)
                    modeled_frames += 1
                else:
                    alpha = np.zeros((540, 960), dtype=np.float32)
                    output = denoised_12bit.astype(RAW_DTYPE, copy=False)

                alpha_u8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
                output_bytes = output.astype(RAW_DTYPE, copy=False).tobytes(order="C")
                alpha_bytes = alpha_u8.tobytes(order="C")
                fusion_file.write(output_bytes)
                alpha_file.write(alpha_bytes)
                fusion_hash.update(output_bytes)
                alpha_hash.update(alpha_bytes)
                alpha_sum += float(alpha.sum())
                if supported_start <= frame_index < supported_stop:
                    static = motion < 0.5
                    moving = ~static
                    alpha_static_sum += float(alpha[static].sum())
                    alpha_motion_sum += float(alpha[moving].sum())
                    modeled_pixels += alpha.size
                    static_pixels += int(static.sum())
                    motion_pixels += int(moving.sum())
                completed = frame_index - frame_start + 1
                if completed == 1 or completed % 10 == 0 or frame_index + 1 == frame_stop:
                    elapsed = time.perf_counter() - started
                    rate = completed / max(elapsed, 1e-6)
                    print(f"  {completed}/{frame_stop-frame_start} ({rate:.2f} frames/s)", flush=True)

        fusion_partial.replace(fusion_path)
        alpha_partial.replace(alpha_path)
        frame_count = frame_stop - frame_start
        packed_pixels = frame_count * 540 * 960
        summary = {
            "sequence_id": sequence.sequence_id,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "frame_interval": [frame_start, frame_stop],
            "model_supported_interval": [supported_start, supported_stop],
            "modeled_frames": modeled_frames,
            "passthrough_d2_frames": frame_count - modeled_frames,
            "fusion": {
                "path": fusion_path.name,
                "format": (
                    "little-endian uint16, right-aligned 12-bit "
                    f"{sequence.denoised.cfa_pattern}"
                ),
                "shape": [frame_count, 1080, 1920],
                "bytes": fusion_path.stat().st_size,
                "sha256": fusion_hash.hexdigest(),
            },
            "alpha": {
                "path": alpha_path.name,
                "format": "uint8, 0=D2 and 255=D3, one value per Bayer 2x2 block",
                "shape": [frame_count, 540, 960],
                "bytes": alpha_path.stat().st_size,
                "sha256": alpha_hash.hexdigest(),
                "mean": alpha_sum / packed_pixels,
                "modeled_mean": alpha_sum / max(modeled_pixels, 1),
                "modeled_static_mean": alpha_static_sum / max(static_pixels, 1),
                "modeled_motion_mean": alpha_motion_sum / max(motion_pixels, 1),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        (sequence_output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

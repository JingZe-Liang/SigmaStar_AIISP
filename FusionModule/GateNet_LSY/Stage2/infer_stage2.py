from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


STAGE2_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STAGE2_ROOT.parents[2]
PHASE2_ROOT = WORKSPACE_ROOT / "Phase2"
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from dataset_io import RAW_DTYPE, RawStreamReader, discover_dataset, pack_bayer  # noqa: E402

from gatenet_stage2 import GateNetStage2, build_gate_features


def unpack_bayer(packed: np.ndarray, cfa_pattern: str) -> np.ndarray:
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


def infer_tiled(
    model: GateNetStage2,
    arrays: tuple[np.ndarray, ...],
    noise_sigma: np.ndarray,
    *,
    device: torch.device,
    tile_size: int,
    halo: int,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = arrays[0].shape[1:]
    alpha = np.empty((height, width), dtype=np.float32)
    predicted_motion = np.empty((height, width), dtype=np.float32)
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
                )
                .unsqueeze(0)
                .to(device)
                for array in arrays
            ]
            sigma_tensor = torch.from_numpy(noise_sigma).view(1, 4, 1, 1).to(device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                features = build_gate_features(*tensors, sigma_tensor)
                tile_alpha, motion_logit = model(features, return_motion=True)
                tile_alpha = tile_alpha[0, 0]
                tile_motion = torch.sigmoid(motion_logit[0, 0])
            core = (
                slice(top - ext_top, bottom - ext_top),
                slice(left - ext_left, right - ext_left),
            )
            alpha[top:bottom, left:right] = tile_alpha[core].float().cpu().numpy()
            predicted_motion[top:bottom, left:right] = (
                tile_motion[core].float().cpu().numpy()
            )
    return alpha, predicted_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage2 GateNet without an external MD input"
    )
    parser.add_argument("--dataset-root", type=Path, default=PHASE2_ROOT / "DATASET")
    parser.add_argument(
        "--checkpoint", type=Path, default=STAGE2_ROOT / "runs" / "final_all" / "best.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=STAGE2_ROOT / "outputs" / "inference_final_all"
    )
    parser.add_argument("--sequences", nargs="+", default=["128x", "645x"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=6)
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
    requested = set(args.sequences)
    selected = [s for s in catalog.fusion_sequences if s.sequence_id in requested]
    if {s.sequence_id for s in selected} != requested:
        raise ValueError(
            f"Unknown sequence IDs: {sorted(requested - {s.sequence_id for s in selected})}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GateNetStage2(**checkpoint.get("model_config", {}))
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
        paths = {
            "fusion": sequence_output / "fusion.raw",
            "alpha": sequence_output / "alpha_u8.raw",
            "motion": sequence_output / "predicted_motion_u8.raw",
        }
        partials = {key: path.with_suffix(path.suffix + ".partial") for key, path in paths.items()}
        source_reader = RawStreamReader(sequence.source)
        denoised_reader = RawStreamReader(sequence.denoised)
        fused_reader = RawStreamReader(sequence.fused)
        hashes = {key: hashlib.sha256() for key in paths}
        alpha_sum = 0.0
        motion_sum = 0.0
        modeled_pixels = 0
        modeled_frames = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        print(f"{sequence.sequence_id}: frames [{frame_start}, {frame_stop}) on {device}", flush=True)
        with (
            partials["fusion"].open("wb") as fusion_file,
            partials["alpha"].open("wb") as alpha_file,
            partials["motion"].open("wb") as motion_file,
        ):
            for frame_index in range(frame_start, frame_stop):
                denoised_12bit = denoised_reader.read_frame(frame_index)
                if supported_start <= frame_index < supported_stop:
                    denoised = (
                        pack_bayer(denoised_12bit, sequence.denoised.cfa_pattern).astype(np.float32)
                        - black_dnr
                    )
                    fused = (
                        pack_bayer(
                            fused_reader.read_frame(frame_index), sequence.fused.cfa_pattern
                        ).astype(np.float32)
                        - black_dnr
                    )

                    def source_at(index: int) -> np.ndarray:
                        packed = pack_bayer(
                            source_reader.read_frame(index), sequence.source.cfa_pattern
                        )
                        return packed.astype(np.float32) - black_source + offset

                    source = source_at(frame_index)
                    source_prev = source_at(frame_index - 1)
                    source_next = source_at(frame_index + 1)
                    alpha, predicted_motion = infer_tiled(
                        model,
                        (denoised, fused, source, source_prev, source_next),
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
                    modeled_pixels += alpha.size
                    alpha_sum += float(alpha.sum())
                    motion_sum += float(predicted_motion.sum())
                else:
                    alpha = np.zeros((540, 960), dtype=np.float32)
                    predicted_motion = np.zeros((540, 960), dtype=np.float32)
                    output = denoised_12bit.astype(RAW_DTYPE, copy=False)

                encoded = {
                    "fusion": output.astype(RAW_DTYPE, copy=False).tobytes(order="C"),
                    "alpha": np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8).tobytes(),
                    "motion": np.clip(np.rint(predicted_motion * 255.0), 0, 255).astype(np.uint8).tobytes(),
                }
                for key, payload in encoded.items():
                    {"fusion": fusion_file, "alpha": alpha_file, "motion": motion_file}[key].write(payload)
                    hashes[key].update(payload)

                completed = frame_index - frame_start + 1
                if completed == 1 or completed % 10 == 0 or frame_index + 1 == frame_stop:
                    rate = completed / max(time.perf_counter() - started, 1e-6)
                    print(f"  {completed}/{frame_stop-frame_start} ({rate:.2f} frames/s)", flush=True)

        for key in paths:
            partials[key].replace(paths[key])
        elapsed = time.perf_counter() - started
        frame_count = frame_stop - frame_start
        summary = {
            "sequence_id": sequence.sequence_id,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "external_md_used": False,
            "inference_inputs": ["2DNR", "3DNR", "noisy_t", "noisy_t-1", "noisy_t+1"],
            "frame_interval": [frame_start, frame_stop],
            "model_supported_interval": [supported_start, supported_stop],
            "modeled_frames": modeled_frames,
            "passthrough_d2_frames": frame_count - modeled_frames,
            "fusion": {
                "path": paths["fusion"].name,
                "shape": [frame_count, 1080, 1920],
                "bytes": paths["fusion"].stat().st_size,
                "sha256": hashes["fusion"].hexdigest(),
            },
            "alpha": {
                "path": paths["alpha"].name,
                "mean_modeled": alpha_sum / max(modeled_pixels, 1),
                "sha256": hashes["alpha"].hexdigest(),
            },
            "predicted_motion": {
                "path": paths["motion"].name,
                "mean_modeled": motion_sum / max(modeled_pixels, 1),
                "sha256": hashes["motion"].hexdigest(),
            },
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": elapsed,
            "frames_per_second": frame_count / max(elapsed, 1e-6),
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        }
        (sequence_output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

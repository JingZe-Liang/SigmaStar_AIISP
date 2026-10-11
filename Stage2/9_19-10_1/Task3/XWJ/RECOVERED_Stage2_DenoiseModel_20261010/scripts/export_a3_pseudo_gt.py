#!/usr/bin/env python3
"""Export A3 EMA predictions as downstream pseudo-ground-truth RAW streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from raw_fusion.finetune.a3 import _WHITE_LEVEL, _channel_vector, _load_json
from raw_fusion.finetune.a4 import _read_aligned_video, infer_video, load_checkpoint_model


_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
_PACKED_SHAPE = (4, 540, 960)
_RAW_SHAPE = (1080, 1920)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unpack_rggb(packed: np.ndarray) -> np.ndarray:
    if packed.shape != _PACKED_SHAPE:
        raise ValueError(f"expected packed shape {_PACKED_SHAPE}, got {packed.shape}")
    height, width = packed.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=packed.dtype)
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 0::2] = packed[2]
    mosaic[1::2, 1::2] = packed[3]
    return mosaic


def _quantize_frame(normalized: np.ndarray, black: np.ndarray) -> np.ndarray:
    if normalized.shape != _PACKED_SHAPE:
        raise ValueError(f"expected normalized shape {_PACKED_SHAPE}, got {normalized.shape}")
    dn = normalized * (_WHITE_LEVEL - black[:, None, None]) + black[:, None, None]
    packed = np.rint(np.clip(dn, 0.0, _WHITE_LEVEL)).astype("<u2")
    return _unpack_rggb(packed)


def _write_raw(path: Path, prediction: np.ndarray, black: np.ndarray) -> tuple[int, int]:
    if prediction.ndim != 4 or prediction.shape[1:] != _PACKED_SHAPE:
        raise ValueError(f"unexpected prediction shape {prediction.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    minimum = int(_WHITE_LEVEL)
    maximum = 0
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            for frame in prediction:
                raw = _quantize_frame(frame, black)
                if raw.shape != _RAW_SHAPE:
                    raise ValueError(f"unexpected raw frame shape {raw.shape}")
                minimum = min(minimum, int(raw.min()))
                maximum = max(maximum, int(raw.max()))
                handle.write(np.ascontiguousarray(raw).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return minimum, maximum


def _split_raw_stream(path: Path, frame_dir: Path, frame_count: int) -> None:
    """Materialize the downstream per-frame RAW contract from a stream."""

    frame_bytes = _RAW_SHAPE[0] * _RAW_SHAPE[1] * np.dtype("<u2").itemsize
    frame_dir.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source:
        for index in range(frame_count):
            payload = source.read(frame_bytes)
            if len(payload) != frame_bytes:
                raise ValueError(f"{path}: truncated frame {index}")
            frame_path = frame_dir / f"out_{index:04d}.raw"
            if frame_path.exists():
                raise FileExistsError(frame_path)
            frame_path.write_bytes(payload)
        if source.read(1):
            raise ValueError(f"{path}: contains bytes after frame {frame_count - 1}")


def _export_gain(
    *,
    supervision_dir: Path,
    gain: str,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    max_frames: int,
    tile_batch: int,
    black: np.ndarray,
) -> dict[str, object]:
    video, _ = _read_aligned_video(supervision_dir, gain, max_frames)
    prediction = infer_video(model, video, device, tile_batch=tile_batch)
    gain_dir = output_dir / gain
    raw_path = gain_dir / f"{gain}_0000_{len(prediction) - 1:04d}.raw"
    minimum, maximum = _write_raw(raw_path, prediction, black)
    _split_raw_stream(raw_path, gain_dir, len(prediction))
    manifest = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "sequence": gain,
        "source_manifest": str((supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json").resolve()),
        "frame_range": [0, len(prediction) - 1],
        "frame_count": len(prediction),
        "width": _RAW_SHAPE[1],
        "height": _RAW_SHAPE[0],
        "packed_shape": list(_PACKED_SHAPE),
        "raw_dtype": "<u2",
        "black_levels_dn": {name: float(value) for name, value in zip(_CHANNEL_NAMES, black, strict=True)},
        "inference_protocol": "A3 EMA; temporal 8-frame overlap-4; spatial 64-pixel overlap-8; A4 aligned 10x domain",
        "label_semantics": "pseudo_ground_truth",
        "output_raw_path": str(raw_path.resolve()),
        "output_raw_bytes": raw_path.stat().st_size,
        "output_raw_sha256": _sha256_file(raw_path),
        "per_frame_raw_dir": str(gain_dir.resolve()),
        "per_frame_raw_pattern": "out_{index:04d}.raw",
        "per_frame_raw_bytes": _RAW_SHAPE[0] * _RAW_SHAPE[1] * np.dtype("<u2").itemsize,
        "minimum": minimum,
        "maximum": maximum,
    }
    (gain_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    del prediction, video
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/pseudo_gt/a3_step005000"))
    parser.add_argument("--gains", default="128x,645x")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for downstream A3 inference")

    report = _load_json(args.supervision_dir / "references" / "a1_reference_report.json")
    black = _channel_vector(report["black_levels"])
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    device = torch.device(args.device)
    model = load_checkpoint_model(args.checkpoint, device, ema=True)
    gains = [value.strip() for value in args.gains.split(",") if value.strip()]
    if not gains:
        raise ValueError("at least one gain is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        gain: _export_gain(
            supervision_dir=args.supervision_dir,
            gain=gain,
            output_dir=args.output_dir,
            model=model,
            device=device,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            max_frames=args.max_frames,
            tile_batch=args.tile_batch,
            black=black,
        )
        for gain in gains
    }
    summary = {
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "label_semantics": "pseudo_ground_truth",
        "gains": manifests,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

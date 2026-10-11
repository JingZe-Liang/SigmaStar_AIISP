#!/usr/bin/env python3
"""Project A3's A1-domain pseudo-GT back into each native gain domain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from raw_fusion.finetune.a3 import _channel_vector, _load_json
from raw_fusion.finetune.stream_io import CHANNEL_NAMES


RAW_SHAPE = (1080, 1920)
PACKED_SHAPE = (4, 540, 960)
FRAME_BYTES = RAW_SHAPE[0] * RAW_SHAPE[1] * np.dtype("<u2").itemsize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pack(raw: np.ndarray) -> np.ndarray:
    return np.stack(
        (raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 1::2], raw[1::2, 0::2]),
        axis=0,
    )


def _unpack(packed: np.ndarray) -> np.ndarray:
    raw = np.empty(RAW_SHAPE, dtype="<u2")
    raw[0::2, 0::2], raw[0::2, 1::2] = packed[0], packed[1]
    raw[1::2, 1::2], raw[1::2, 0::2] = packed[2], packed[3]
    return raw


def _project_gain(a3_dir: Path, supervision_dir: Path, gain: str, output_dir: Path, frame_count: int) -> dict[str, object]:
    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    slopes = _channel_vector(manifest["exposure_fit"]["slopes"])
    intercepts = _channel_vector(manifest["exposure_fit"]["intercepts"])
    source = a3_dir / gain / f"{gain}_0000_{frame_count - 1:04d}.raw"
    if source.stat().st_size != frame_count * FRAME_BYTES:
        raise ValueError(f"{gain}: unexpected A3 stream size")
    gain_dir = output_dir / gain
    gain_dir.mkdir(parents=True, exist_ok=True)
    stream_path = gain_dir / source.name
    if stream_path.exists():
        raise FileExistsError(stream_path)
    minimum, maximum = 4095, 0
    with source.open("rb") as src, stream_path.open("wb") as dst:
        for index in range(frame_count):
            raw = np.frombuffer(src.read(FRAME_BYTES), dtype="<u2").reshape(RAW_SHAPE)
            native = (_pack(raw).astype(np.float32) - intercepts[:, None, None]) / slopes[:, None, None]
            projected = np.rint(np.clip(_unpack(native), 0.0, 4095.0)).astype("<u2")
            dst.write(projected.tobytes())
            (gain_dir / f"out_{index:04d}.raw").write_bytes(projected.tobytes())
            minimum = min(minimum, int(projected.min()))
            maximum = max(maximum, int(projected.max()))
    result = {
        "sequence": gain,
        "frame_count": frame_count,
        "source_a3_stream": str(source.resolve()),
        "source_a3_sha256": _sha256(source),
        "native_stream": str(stream_path.resolve()),
        "native_stream_sha256": _sha256(stream_path),
        "frame_bytes": FRAME_BYTES,
        "minimum": minimum,
        "maximum": maximum,
        "projection": "inverse A4 per-channel exposure fit: native=(A1-intercept)/slope",
        "slopes": {name: float(value) for name, value in zip(CHANNEL_NAMES, slopes, strict=True)},
        "intercepts": {name: float(value) for name, value in zip(CHANNEL_NAMES, intercepts, strict=True)},
        "label_semantics": "downstream_target_native_domain",
    }
    (gain_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-dir", type=Path, default=Path("artifacts/finetune_mis20s1/pseudo_gt/a3_step005000"))
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_mis20s1/pseudo_gt/a3_step005000_native"))
    parser.add_argument("--gains", default="128x,645x")
    parser.add_argument("--frame-count", type=int, default=200)
    args = parser.parse_args()
    results = {
        gain: _project_gain(args.a3_dir, args.supervision_dir, gain, args.output_dir, args.frame_count)
        for gain in (value.strip() for value in args.gains.split(",")) if gain
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"protocol": "A3 pseudo-GT inverse exposure projection", "gains": results}, indent=2) + "\n",
        encoding="ascii",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

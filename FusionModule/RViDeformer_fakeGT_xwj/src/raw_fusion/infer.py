"""Tiled full-frame RAW fusion inference with atomic stream output."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import operator
import os
from pathlib import Path
import tempfile
from typing import Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .checkpoint import load_checkpoint_strict
from .config import config_fingerprint, load_dataset_config, load_experiment_config
from .model import CausalRawFusionNet
from .raw import RawStreamReader, normalize_raw, pack_rggb, quantize_normalized, unpack_rggb


_INPUT_NAMES = ("prev_noisy", "curr_noisy", "denoised", "fused")


@dataclass(frozen=True, slots=True)
class TiledFusionOutput:
    prediction: Tensor
    gate: Tensor
    correction: Tensor
    weight_sum: Tensor


@dataclass(frozen=True, slots=True)
class InferenceManifest:
    checkpoint_sha256: str
    config_fingerprint: str
    sequence: str
    frame_range: tuple[int, int]
    frame_count: int
    output_raw_path: Path
    output_raw_bytes: int
    minimum: int
    maximum: int


def tile_starts(length: int, *, tile: int, overlap: int) -> tuple[int, ...]:
    """Return tile starts which include the final edge-aligned tile."""
    image_length = operator.index(length)
    tile_length = operator.index(tile)
    overlap_length = operator.index(overlap)
    if image_length <= 0 or tile_length <= 0 or tile_length > image_length:
        raise ValueError("tile must satisfy 0 < tile <= length")
    if overlap_length < 0 or overlap_length >= tile_length:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    stride = tile_length - overlap_length
    starts = list(range(0, image_length - tile_length + 1, stride))
    final_start = image_length - tile_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def infer_tiled(
    model: nn.Module,
    inputs: Mapping[str, Tensor],
    *,
    tile_size: int,
    overlap: int,
) -> TiledFusionOutput:
    """Infer a packed RAW frame by overlapping tiles and float32 window blending."""
    tensors = _validate_tiled_inputs(inputs)
    batch, _, height, width = tensors["curr_noisy"].shape
    tile = operator.index(tile_size)
    rows = tile_starts(height, tile=tile, overlap=overlap)
    columns = tile_starts(width, tile=tile, overlap=overlap)
    device = tensors["curr_noisy"].device
    axis = torch.hann_window(tile + 2, periodic=False, device=device, dtype=torch.float32)[1:-1]
    window = torch.outer(axis, axis).clamp_min(1e-3).reshape(1, 1, tile, tile)
    prediction_sum = torch.zeros((batch, 4, height, width), device=device, dtype=torch.float32)
    gate_sum = torch.zeros((batch, 1, height, width), device=device, dtype=torch.float32)
    correction_sum = torch.zeros((batch, 4, height, width), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((batch, 1, height, width), device=device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for top in rows:
                for left in columns:
                    tile_inputs = {
                        name: value[..., top : top + tile, left : left + tile]
                        for name, value in tensors.items()
                    }
                    output = model(**tile_inputs)
                    _validate_tile_output(
                        output,
                        batch=batch,
                        tile=tile,
                        expected_device=device,
                    )
                    prediction_sum[..., top : top + tile, left : left + tile] += (
                        output.prediction.float() * window
                    )
                    gate_sum[..., top : top + tile, left : left + tile] += (
                        output.gate.float() * window
                    )
                    correction_sum[..., top : top + tile, left : left + tile] += (
                        output.correction.float() * window
                    )
                    weight_sum[..., top : top + tile, left : left + tile] += window
    finally:
        model.train(was_training)

    if not bool(torch.all(weight_sum > 0)):
        raise RuntimeError("tile blending left uncovered pixels")
    return TiledFusionOutput(
        prediction=prediction_sum / weight_sum,
        gate=gate_sum / weight_sum,
        correction=correction_sum / weight_sum,
        weight_sum=weight_sum,
    )


class AtomicRawStreamWriter:
    """Write fixed-size RAW frames without exposing partial final output."""

    def __init__(self, path: Path, width: int, height: int, frame_count: int) -> None:
        self.path = Path(path)
        self.width = operator.index(width)
        self.height = operator.index(height)
        self.frame_count = operator.index(frame_count)
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise ValueError("width, height, and frame_count must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._file = temporary
        self._temporary_path = Path(temporary.name)
        self._written = 0
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._written >= self.frame_count:
            self.abort()
            raise RuntimeError("writer cannot accept more frames")
        try:
            array = np.asarray(frame)
            if array.shape != (self.height, self.width):
                raise ValueError("frame shape does not match configured height and width")
            if array.dtype != np.dtype("<u2"):
                raise TypeError("frame dtype must be little-endian uint16")
            self._file.write(np.ascontiguousarray(array).tobytes())
            self._written += 1
        except BaseException:
            self.abort()
            raise

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._written != self.frame_count:
            self.abort()
            raise RuntimeError("writer received fewer frames than configured")
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            os.replace(self._temporary_path, self.path)
            self._closed = True
            _fsync_directory(self.path.parent)
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._file.close()
        finally:
            try:
                self._temporary_path.unlink()
            except FileNotFoundError:
                pass
            self._closed = True

    def __enter__(self) -> AtomicRawStreamWriter:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        try:
            self.close()
        except BaseException:
            self.abort()
            raise
        return False


def infer_sequence(
    config_path: Path,
    checkpoint_path: Path,
    sequence: str,
    frame_range: tuple[int, int],
    output_dir: Path,
    device: torch.device | str,
) -> InferenceManifest:
    """Run causal full-frame inference for an inclusive sequence frame range."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _invalidate_manifest(destination / "manifest.json")
    experiment = load_experiment_config(Path(config_path))
    dataset = load_dataset_config(experiment.dataset_path)
    _validate_inference_layout(dataset.layout)
    if sequence not in dataset.sequences:
        raise ValueError(f"unknown sequence: {sequence}")
    start, end = _validate_frame_range(frame_range, dataset.layout.frame_count)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA inference without an available GPU")
    fingerprint = config_fingerprint(dataset, experiment)
    with _checkpoint_snapshot(Path(checkpoint_path), destination) as (
        snapshot_path,
        checkpoint_sha256,
    ):
        model = CausalRawFusionNet(experiment.model).to(target_device)
        load_checkpoint_strict(
            snapshot_path,
            expected_fingerprint=fingerprint,
            model=model,
            device=target_device,
        )
        model.eval()
        selected = dataset.sequences[sequence]
        layout = dataset.layout
        noisy_reader = RawStreamReader(
            selected.noisy_stream,
            layout.width,
            layout.height,
            layout.frame_count,
            shift=4,
        )
        denoised_reader = RawStreamReader(
            selected.denoised_stream, layout.width, layout.height, layout.frame_count, shift=0
        )
        fused_reader = RawStreamReader(
            selected.fused_stream, layout.width, layout.height, layout.frame_count, shift=0
        )
        output_raw_path = destination / f"{sequence}_{start:04d}_{end:04d}.raw"
        minimum = 4095
        maximum = 0
        with AtomicRawStreamWriter(
            output_raw_path,
            width=layout.width,
            height=layout.height,
            frame_count=end - start + 1,
        ) as writer:
            for index in range(start, end + 1):
                previous_index = index if index == 0 else index - 1
                inputs = {
                    "prev_noisy": _packed_tensor(
                        noisy_reader.read_frame(previous_index), black_level=252, device=target_device
                    ),
                    "curr_noisy": _packed_tensor(
                        noisy_reader.read_frame(index), black_level=252, device=target_device
                    ),
                    "denoised": _packed_tensor(
                        denoised_reader.read_frame(index), black_level=300, device=target_device
                    ),
                    "fused": _packed_tensor(
                        fused_reader.read_frame(index), black_level=300, device=target_device
                    ),
                }
                output = infer_tiled(
                    model,
                    inputs,
                    tile_size=experiment.inference.tile_size_packed,
                    overlap=experiment.inference.overlap_packed,
                )
                normalized = output.prediction[0].detach().cpu().numpy()
                quantized = quantize_normalized(unpack_rggb(normalized), 252, 4095)
                writer.write(quantized)
                minimum = min(minimum, int(quantized.min()))
                maximum = max(maximum, int(quantized.max()))

        manifest = InferenceManifest(
            checkpoint_sha256=checkpoint_sha256,
            config_fingerprint=fingerprint,
            sequence=sequence,
            frame_range=(start, end),
            frame_count=end - start + 1,
            output_raw_path=output_raw_path,
            output_raw_bytes=output_raw_path.stat().st_size,
            minimum=minimum,
            maximum=maximum,
        )
        _write_manifest_atomic(destination / "manifest.json", manifest)
        return manifest


def _validate_tiled_inputs(inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if set(inputs) != set(_INPUT_NAMES):
        raise ValueError("inputs must contain exactly the four fusion tensors")
    tensors = {name: inputs[name] for name in _INPUT_NAMES}
    if any(not isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("all four inputs must be tensors")
    reference = tensors["prev_noisy"]
    if reference.ndim != 4 or reference.shape[1] != 4:
        raise ValueError("all four inputs must have shape [B, 4, H, W]")
    if any(value.shape != reference.shape for value in tensors.values()):
        raise ValueError("all four inputs must have the same shape")
    if any(not value.is_floating_point() for value in tensors.values()):
        raise TypeError("all four inputs must be floating point")
    if any(value.dtype != reference.dtype for value in tensors.values()):
        raise ValueError("all four inputs must have the same dtype")
    if any(value.device != reference.device for value in tensors.values()):
        raise ValueError("all four inputs must have the same device")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("all four inputs must be finite")
    return tensors


def _validate_tile_output(
    output: object,
    *,
    batch: int,
    tile: int,
    expected_device: torch.device,
) -> None:
    for name, channels in (("prediction", 4), ("gate", 1), ("correction", 4)):
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or value.shape != (batch, channels, tile, tile):
            raise ValueError(f"model {name} output must match tile geometry")
        if not value.is_floating_point():
            raise TypeError(f"model {name} output must be floating point")
        if value.device != expected_device:
            raise ValueError(f"model {name} output must be on the input device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"model {name} output must be finite")


def _validate_inference_layout(layout: object) -> None:
    required = {
        "white_level": 4095,
        "noisy_black_level": 252,
        "candidate_black_level": 300,
        "target_black_level": 252,
        "noisy_shift": 4,
        "cfa_pattern": "RGGB",
    }
    if any(getattr(layout, name, None) != expected for name, expected in required.items()):
        raise ValueError("inference requires the fixed MIS20S1 RAW layout")


def _validate_frame_range(frame_range: tuple[int, int], frame_count: int) -> tuple[int, int]:
    if not isinstance(frame_range, tuple) or len(frame_range) != 2:
        raise ValueError("frame_range must be an inclusive (start, end) tuple")
    start, end = (operator.index(value) for value in frame_range)
    if start < 0 or start > end or end >= frame_count:
        raise ValueError("frame_range is outside the configured sequence")
    return start, end


def _packed_tensor(frame: np.ndarray, *, black_level: int, device: torch.device) -> Tensor:
    packed = pack_rggb(normalize_raw(frame, black_level, 4095))
    return torch.from_numpy(packed).unsqueeze(0).to(device=device, dtype=torch.float32)


@contextmanager
def _checkpoint_snapshot(path: Path, directory: Path) -> Iterator[tuple[Path, str]]:
    snapshot_path, checkpoint_sha256 = _create_checkpoint_snapshot(path, directory)
    try:
        yield snapshot_path, checkpoint_sha256
    finally:
        _remove_file_and_fsync(snapshot_path)


def _create_checkpoint_snapshot(path: Path, directory: Path) -> tuple[Path, str]:
    source_path = Path(path)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as source, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".checkpoint-snapshot.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                temporary.write(block)
                digest.update(block)
            temporary.flush()
            os.fsync(temporary.fileno())
    except OSError as error:
        if temporary_path is not None:
            _remove_file_and_fsync(temporary_path)
        raise ValueError(f"无法读取检查点: {source_path}") from error
    if temporary_path is None:
        raise RuntimeError("checkpoint snapshot was not created")
    return temporary_path, digest.hexdigest()


def _invalidate_manifest(path: Path) -> None:
    _remove_file_and_fsync(path)


def _remove_file_and_fsync(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    _fsync_directory(Path(path).parent)


def _write_manifest_atomic(path: Path, manifest: InferenceManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    payload["frame_range"] = list(manifest.frame_range)
    payload["output_raw_path"] = str(manifest.output_raw_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="分块推理 RAW 融合网络")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)
    manifest = infer_sequence(
        arguments.config,
        arguments.checkpoint,
        arguments.sequence,
        (arguments.start, arguments.end),
        arguments.output_dir,
        arguments.device,
    )
    payload = asdict(manifest)
    payload["frame_range"] = list(manifest.frame_range)
    payload["output_raw_path"] = str(manifest.output_raw_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


WIDTH = 1920
HEIGHT = 1080
BYTES_PER_PIXEL = 2
FRAME_BYTES = WIDTH * HEIGHT * BYTES_PER_PIXEL
RAW_DTYPE = np.dtype("<u2")

LEFT_ALIGNED_12BIT = "left_aligned_12bit_in_u16"
RIGHT_ALIGNED_12BIT = "right_aligned_12bit_in_u16"


class DatasetValidationError(RuntimeError):
    pass


def _frame_index(path: Path) -> int:
    match = re.fullmatch(r"out_(\d+)", path.stem)
    if match is None:
        raise DatasetValidationError(f"Unexpected frame filename: {path}")
    return int(match.group(1))


def _capture_metadata(filename: str) -> dict[str, int]:
    patterns = (
        (
            r"Shutter=(\d+),SenserG=(\d+),IspG=(\d+),R=(\d+),G=(\d+),B=(\d+)",
            ("shutter", "sensor_gain", "isp_gain", "r_gain", "g_gain", "b_gain"),
        ),
        (
            r"FN=(\d+),US=(\d+),AG=(\d+),DG=(\d+),BV=(-?\d+),R=(\d+),G=(\d+),B=(\d+)",
            ("fn", "exposure_us", "analog_gain", "digital_gain", "bv", "r_gain", "g_gain", "b_gain"),
        ),
    )
    for pattern, names in patterns:
        match = re.search(pattern, filename)
        if match:
            return {name: int(value) for name, value in zip(names, match.groups())}
    return {}


def _cfa_pattern(path: Path) -> str:
    name = path.as_posix().upper()
    if "_16_GR_" in name:
        return "GRBG"
    if "@RG" in name or "_16_RG_" in name:
        return "RGGB"
    raise DatasetValidationError(f"Cannot infer CFA pattern from {path}")


def _source_category(path: Path, source_root: Path) -> tuple[str, str]:
    relative = path.relative_to(source_root)
    parts = relative.parts
    if parts[0] == "mis20s1_calibrationdata":
        return f"calibration/{parts[1]}", parts[2]
    return parts[0], parts[1]


@dataclass(frozen=True)
class RawStreamSpec:
    path: Path
    role: str
    encoding: str
    cfa_pattern: str
    frame_count: int
    category: str
    condition: str
    metadata: dict[str, int] = field(default_factory=dict)
    width: int = WIDTH
    height: int = HEIGHT

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        role: str,
        encoding: str,
        category: str,
        condition: str,
        cfa_pattern: str | None = None,
    ) -> "RawStreamSpec":
        path = path.resolve()
        size = path.stat().st_size
        frame_bytes = WIDTH * HEIGHT * BYTES_PER_PIXEL
        frame_count, remainder = divmod(size, frame_bytes)
        if remainder:
            raise DatasetValidationError(
                f"RAW size is not an integer number of frames: {path} ({size} bytes)"
            )
        if frame_count == 0:
            raise DatasetValidationError(f"Empty RAW stream: {path}")
        return cls(
            path=path,
            role=role,
            encoding=encoding,
            cfa_pattern=cfa_pattern or _cfa_pattern(path),
            frame_count=frame_count,
            category=category,
            condition=condition,
            metadata=_capture_metadata(path.name),
        )


class RawStreamReader:
    """Random-access reader for one little-endian uint16 RAW stream."""

    def __init__(self, spec: RawStreamSpec):
        self.spec = spec

    def read_frame(
        self,
        index: int,
        *,
        crop: tuple[int, int, int, int] | None = None,
        convert_to_12bit: bool = True,
    ) -> np.ndarray:
        if not 0 <= index < self.spec.frame_count:
            raise IndexError(
                f"Frame {index} outside [0, {self.spec.frame_count}) for {self.spec.path}"
            )
        offset = index * self.spec.width * self.spec.height * BYTES_PER_PIXEL
        frame = np.memmap(
            self.spec.path,
            dtype=RAW_DTYPE,
            mode="r",
            offset=offset,
            shape=(self.spec.height, self.spec.width),
            order="C",
        )
        if crop is not None:
            top, left, crop_height, crop_width = crop
            if min(top, left, crop_height, crop_width) < 0:
                raise ValueError(f"Crop values must be non-negative: {crop}")
            if crop_height == 0 or crop_width == 0:
                raise ValueError(f"Crop dimensions must be positive: {crop}")
            if top + crop_height > self.spec.height or left + crop_width > self.spec.width:
                raise ValueError(f"Crop {crop} exceeds {self.spec.width}x{self.spec.height}")
            frame = frame[top : top + crop_height, left : left + crop_width]

        array = np.asarray(frame)
        if not convert_to_12bit:
            return array.copy()
        if self.spec.encoding == LEFT_ALIGNED_12BIT:
            return np.right_shift(array, 4).astype(np.uint16, copy=False)
        if self.spec.encoding == RIGHT_ALIGNED_12BIT:
            return array.copy()
        raise DatasetValidationError(f"Unknown RAW encoding: {self.spec.encoding}")


def pack_bayer(
    mosaic: np.ndarray,
    cfa_pattern: str,
    *,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Pack a Bayer mosaic as R, G1, G2, B while preserving global CFA phase."""
    if mosaic.ndim != 2:
        raise ValueError(f"Expected a 2D Bayer mosaic, got shape {mosaic.shape}")
    if mosaic.shape[0] % 2 or mosaic.shape[1] % 2:
        raise ValueError(f"Packed Bayer input must have even dimensions, got {mosaic.shape}")
    if cfa_pattern not in {"RGGB", "GRBG", "GBRG", "BGGR"}:
        raise ValueError(f"Unsupported CFA pattern: {cfa_pattern}")

    labels: list[str] = []
    green_index = 0
    for color in cfa_pattern:
        if color == "G":
            green_index += 1
            labels.append(f"G{green_index}")
        else:
            labels.append(color)
    global_labels = np.asarray(labels, dtype=object).reshape(2, 2)

    top, left = origin
    planes: dict[str, np.ndarray] = {}
    for local_y in range(2):
        for local_x in range(2):
            label = global_labels[(top + local_y) % 2, (left + local_x) % 2]
            planes[str(label)] = mosaic[local_y::2, local_x::2]
    return np.stack([planes["R"], planes["G1"], planes["G2"], planes["B"]], axis=0)


@dataclass(frozen=True)
class FusionSequence:
    sequence_id: str
    source: RawStreamSpec
    denoised: RawStreamSpec
    fused: RawStreamSpec
    denoised_frames: tuple[Path, ...]
    fused_frames: tuple[Path, ...]
    denoised_pngs: tuple[Path, ...]
    fused_pngs: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return self.source.frame_count


@dataclass(frozen=True)
class DatasetCatalog:
    root: Path
    source_streams: tuple[RawStreamSpec, ...]
    fusion_sequences: tuple[FusionSequence, ...]
    physical_files: tuple[Path, ...]
    archive_path: Path | None
    readme_path: Path | None

    @property
    def source_frame_count(self) -> int:
        return sum(stream.frame_count for stream in self.source_streams)

    @property
    def paired_frame_count(self) -> int:
        return sum(sequence.frame_count for sequence in self.fusion_sequences)

    def manifest(self) -> dict[str, Any]:
        def raw_entry(spec: RawStreamSpec) -> dict[str, Any]:
            return {
                "path": spec.path.relative_to(self.root).as_posix(),
                "role": spec.role,
                "encoding": spec.encoding,
                "cfa_pattern": spec.cfa_pattern,
                "width": spec.width,
                "height": spec.height,
                "frame_count": spec.frame_count,
                "category": spec.category,
                "condition": spec.condition,
                "metadata": spec.metadata,
            }

        return {
            "schema_version": 1,
            "root": str(self.root),
            "summary": {
                "physical_file_count": len(self.physical_files),
                "source_stream_count": len(self.source_streams),
                "source_frame_count": self.source_frame_count,
                "fusion_sequence_count": len(self.fusion_sequences),
                "paired_frame_count": self.paired_frame_count,
            },
            "source_streams": [raw_entry(spec) for spec in self.source_streams],
            "fusion_sequences": [
                {
                    "sequence_id": sequence.sequence_id,
                    "frame_count": sequence.frame_count,
                    "source": raw_entry(sequence.source),
                    "denoised": raw_entry(sequence.denoised),
                    "fused": raw_entry(sequence.fused),
                    "denoised_frame_files": [
                        str(path.relative_to(self.root).as_posix())
                        for path in sequence.denoised_frames
                    ],
                    "fused_frame_files": [
                        str(path.relative_to(self.root).as_posix())
                        for path in sequence.fused_frames
                    ],
                    "denoised_png_files": [
                        str(path.relative_to(self.root).as_posix())
                        for path in sequence.denoised_pngs
                    ],
                    "fused_png_files": [
                        str(path.relative_to(self.root).as_posix())
                        for path in sequence.fused_pngs
                    ],
                }
                for sequence in self.fusion_sequences
            ],
            "physical_files": [
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "size": path.stat().st_size,
                    "suffix": path.suffix.lower(),
                }
                for path in self.physical_files
            ],
            "archive": (
                self.archive_path.relative_to(self.root).as_posix()
                if self.archive_path is not None
                else None
            ),
            "readme": (
                self.readme_path.relative_to(self.root).as_posix()
                if self.readme_path is not None
                else None
            ),
        }


def discover_dataset(root: str | Path) -> DatasetCatalog:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    source_root = root / "Sigmastar_7_30"
    if not source_root.is_dir():
        raise DatasetValidationError(f"Missing source root: {source_root}")

    source_streams: list[RawStreamSpec] = []
    for path in sorted(source_root.rglob("*.raw")):
        category, condition = _source_category(path, source_root)
        source_streams.append(
            RawStreamSpec.from_path(
                path,
                role="source",
                encoding=LEFT_ALIGNED_12BIT,
                category=category,
                condition=condition,
            )
        )

    source_by_condition: dict[tuple[str, int, int, int], list[RawStreamSpec]] = {}
    for spec in source_streams:
        metadata = spec.metadata
        if {"r_gain", "g_gain", "b_gain"} <= metadata.keys():
            key = (
                spec.condition,
                metadata["r_gain"],
                metadata["g_gain"],
                metadata["b_gain"],
            )
            source_by_condition.setdefault(key, []).append(spec)

    fusion_sequences: list[FusionSequence] = []
    output_pattern = re.compile(
        r"raw_stream_1920x1080_16bit@RG_R=(\d+),G=(\d+),B=(\d+)_(.+)"
    )
    for output_dir in sorted(path for path in root.glob("raw_stream_*") if path.is_dir()):
        match = output_pattern.fullmatch(output_dir.name)
        if match is None:
            raise DatasetValidationError(f"Unexpected output directory: {output_dir}")
        r_gain, g_gain, b_gain = (int(value) for value in match.groups()[:3])
        condition = match.group(4)
        candidates = source_by_condition.get((condition, r_gain, g_gain, b_gain), [])
        if len(candidates) != 1:
            raise DatasetValidationError(
                f"Expected one source for {output_dir.name}, found {len(candidates)}"
            )
        source = candidates[0]
        denoised = RawStreamSpec.from_path(
            output_dir / "denoised.raw",
            role="2dnr",
            encoding=RIGHT_ALIGNED_12BIT,
            category=source.category,
            condition=condition,
        )
        fused = RawStreamSpec.from_path(
            output_dir / "fused.raw",
            role="3dnr",
            encoding=RIGHT_ALIGNED_12BIT,
            category=source.category,
            condition=condition,
        )
        fusion_sequences.append(
            FusionSequence(
                sequence_id=condition,
                source=source,
                denoised=denoised,
                fused=fused,
                denoised_frames=tuple(
                    sorted((output_dir / "denoised").glob("*.raw"), key=_frame_index)
                ),
                fused_frames=tuple(
                    sorted((output_dir / "fused").glob("*.raw"), key=_frame_index)
                ),
                denoised_pngs=tuple(
                    sorted((output_dir / "denoised").glob("*.png"), key=_frame_index)
                ),
                fused_pngs=tuple(
                    sorted((output_dir / "fused").glob("*.png"), key=_frame_index)
                ),
            )
        )

    archive_path = root / "mis20s1_2D&3D.zip"
    readme_path = root / "readme.txt"
    return DatasetCatalog(
        root=root,
        source_streams=tuple(source_streams),
        fusion_sequences=tuple(fusion_sequences),
        physical_files=tuple(sorted(path for path in root.rglob("*") if path.is_file())),
        archive_path=archive_path if archive_path.is_file() else None,
        readme_path=readme_path if readme_path.is_file() else None,
    )


def _sample_values(spec: RawStreamSpec) -> np.ndarray:
    reader = RawStreamReader(spec)
    samples = []
    for index in sorted({0, spec.frame_count // 2, spec.frame_count - 1}):
        frame = reader.read_frame(index, convert_to_12bit=False)
        samples.append(frame[::64, ::64].reshape(-1))
    return np.concatenate(samples)


def _compare_files(first: Path, second: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(chunk_size)
            right_chunk = right.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(catalog: DatasetCatalog, *, deep: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    allowed_suffixes = {".raw", ".png", ".zip", ".txt"}
    unknown = [path for path in catalog.physical_files if path.suffix.lower() not in allowed_suffixes]
    if unknown:
        errors.extend(f"Unclassified file: {path}" for path in unknown)

    for spec in catalog.source_streams:
        sample = _sample_values(spec)
        if np.any(sample & 0xF):
            errors.append(f"Source is not left-aligned 12-bit: {spec.path}")

    verified_output_frames = 0
    for sequence in catalog.fusion_sequences:
        counts = {
            sequence.source.frame_count,
            sequence.denoised.frame_count,
            sequence.fused.frame_count,
            len(sequence.denoised_frames),
            len(sequence.fused_frames),
            len(sequence.denoised_pngs),
            len(sequence.fused_pngs),
        }
        if len(counts) != 1:
            errors.append(f"Frame counts disagree in sequence {sequence.sequence_id}: {counts}")
            continue
        expected_indices = list(range(sequence.frame_count))
        for label, paths in (
            ("2DNR RAW", sequence.denoised_frames),
            ("3DNR RAW", sequence.fused_frames),
            ("2DNR PNG", sequence.denoised_pngs),
            ("3DNR PNG", sequence.fused_pngs),
        ):
            indices = [_frame_index(path) for path in paths]
            if indices != expected_indices:
                errors.append(f"Non-contiguous {label} indices in {sequence.sequence_id}")
        for spec in (sequence.denoised, sequence.fused):
            sample = _sample_values(spec)
            if int(sample.max()) > 4095:
                errors.append(f"DNR output exceeds 12-bit range: {spec.path}")
        for path in (*sequence.denoised_frames, *sequence.fused_frames):
            if path.stat().st_size != FRAME_BYTES:
                errors.append(f"Single-frame RAW has wrong size: {path}")

        if deep:
            for stream, frames in (
                (sequence.denoised.path, sequence.denoised_frames),
                (sequence.fused.path, sequence.fused_frames),
            ):
                with stream.open("rb") as stream_handle:
                    for frame_path in frames:
                        stream_chunk = stream_handle.read(FRAME_BYTES)
                        with frame_path.open("rb") as frame_handle:
                            frame_chunk = frame_handle.read()
                        if stream_chunk != frame_chunk:
                            errors.append(
                                f"Concatenated stream differs from {frame_path}"
                            )
                            break
                        verified_output_frames += 1
                    if stream_handle.read(1):
                        errors.append(f"Trailing bytes in concatenated stream: {stream}")

    archive_member_count = 0
    if catalog.archive_path is not None:
        with zipfile.ZipFile(catalog.archive_path) as archive:
            archived = {
                info.filename: info.file_size
                for info in archive.infolist()
                if not info.is_dir()
            }
        archive_member_count = len(archived)
        extracted = {
            path.relative_to(catalog.root).as_posix(): path.stat().st_size
            for path in catalog.physical_files
            if path != catalog.archive_path
            and "Sigmastar_7_30" not in path.relative_to(catalog.root).parts
        }
        missing = sorted(set(archived) - set(extracted))
        extra = sorted(set(extracted) - set(archived))
        mismatched = sorted(
            path for path in set(archived) & set(extracted) if archived[path] != extracted[path]
        )
        if missing:
            errors.append(f"Archive members missing after extraction: {missing[:5]}")
        if extra:
            errors.append(f"Extracted files absent from archive: {extra[:5]}")
        if mismatched:
            errors.append(f"Archive/extracted size mismatch: {mismatched[:5]}")

    source_hashes: dict[str, str] = {}
    png_verified = 0
    if deep:
        for spec in catalog.source_streams:
            source_hashes[spec.path.relative_to(catalog.root).as_posix()] = _sha256(spec.path)
        try:
            from PIL import Image
        except ImportError as error:
            errors.append(f"Pillow is required for deep PNG validation: {error}")
        else:
            for sequence in catalog.fusion_sequences:
                for path in (*sequence.denoised_pngs, *sequence.fused_pngs):
                    try:
                        with Image.open(path) as image:
                            if image.size != (WIDTH, HEIGHT) or image.mode != "RGB":
                                errors.append(
                                    f"Unexpected PNG format {image.size}/{image.mode}: {path}"
                                )
                            image.verify()
                        png_verified += 1
                    except Exception as error:
                        errors.append(f"Invalid PNG {path}: {error}")

    report = {
        "ok": not errors,
        "deep": deep,
        "physical_file_count": len(catalog.physical_files),
        "source_stream_count": len(catalog.source_streams),
        "source_frame_count": catalog.source_frame_count,
        "fusion_sequence_count": len(catalog.fusion_sequences),
        "paired_frame_count": catalog.paired_frame_count,
        "archive_member_count": archive_member_count,
        "deep_verified_output_frame_files": verified_output_frames,
        "deep_verified_png_files": png_verified,
        "source_sha256": source_hashes,
        "warnings": warnings,
        "errors": errors,
    }
    if errors:
        raise DatasetValidationError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _prepare_tensor(
    mosaic: np.ndarray,
    *,
    cfa_pattern: str,
    origin: tuple[int, int],
    black_level: float | None,
    normalize: bool,
) -> torch.Tensor:
    packed = pack_bayer(mosaic, cfa_pattern, origin=origin).astype(np.float32)
    if black_level is not None:
        packed = np.maximum(packed - black_level, 0.0)
    if normalize:
        denominator = 4095.0 if black_level is None else 4095.0 - black_level
        packed = packed / denominator
    return torch.from_numpy(np.ascontiguousarray(packed))


class AllSourceFramesDataset(Dataset):
    """Index every frame in all 29 source streams without dropping boundaries."""

    def __init__(
        self,
        catalog: DatasetCatalog,
        *,
        pack: bool = True,
        normalize: bool = False,
        black_level: float | None = None,
    ):
        self.catalog = catalog
        self.pack = pack
        self.normalize = normalize
        self.black_level = black_level
        self._ends: list[int] = []
        total = 0
        for stream in catalog.source_streams:
            total += stream.frame_count
            self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        stream_index = bisect.bisect_right(self._ends, index)
        previous_end = 0 if stream_index == 0 else self._ends[stream_index - 1]
        frame_index = index - previous_end
        spec = self.catalog.source_streams[stream_index]
        mosaic = RawStreamReader(spec).read_frame(frame_index)
        if self.pack:
            data = _prepare_tensor(
                mosaic,
                cfa_pattern=spec.cfa_pattern,
                origin=(0, 0),
                black_level=self.black_level,
                normalize=self.normalize,
            )
        else:
            array = mosaic.astype(np.float32)
            if self.black_level is not None:
                array = np.maximum(array - self.black_level, 0.0)
            if self.normalize:
                denominator = 4095.0 if self.black_level is None else 4095.0 - self.black_level
                array = array / denominator
            data = torch.from_numpy(np.ascontiguousarray(array))
        return {
            "raw": data,
            "stream_index": stream_index,
            "frame_index": frame_index,
            "path": str(spec.path),
            "category": spec.category,
            "condition": spec.condition,
            "cfa_pattern": spec.cfa_pattern,
            "metadata": spec.metadata,
        }


class PairedFusionDataset(Dataset):
    """Read every source/2DNR/3DNR frame from all paired sequences."""

    def __init__(
        self,
        catalog: DatasetCatalog,
        *,
        crop_size: int | None = None,
        temporal_radius: int = 0,
        normalize: bool = False,
        black_source: float | None = None,
        black_dnr: float | None = None,
    ):
        if temporal_radius not in {0, 1}:
            raise ValueError("temporal_radius must be 0 or 1")
        if crop_size is not None and (crop_size <= 0 or crop_size % 2):
            raise ValueError("crop_size must be a positive even integer")
        self.catalog = catalog
        self.crop_size = crop_size
        self.temporal_radius = temporal_radius
        self.normalize = normalize
        self.black_source = black_source
        self.black_dnr = black_dnr
        self.samples: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(catalog.fusion_sequences):
            start = temporal_radius
            stop = sequence.frame_count - temporal_radius
            self.samples.extend((sequence_index, index) for index in range(start, stop))

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self) -> tuple[int, int, int, int] | None:
        if self.crop_size is None:
            return None
        top = int(np.random.randint(0, HEIGHT - self.crop_size + 1))
        left = int(np.random.randint(0, WIDTH - self.crop_size + 1))
        return top, left, self.crop_size, self.crop_size

    def __getitem__(self, item: int) -> dict[str, Any]:
        sequence_index, frame_index = self.samples[item]
        sequence = self.catalog.fusion_sequences[sequence_index]
        crop = self._crop()
        origin = (0, 0) if crop is None else crop[:2]
        readers = {
            "source": RawStreamReader(sequence.source),
            "denoised": RawStreamReader(sequence.denoised),
            "fused": RawStreamReader(sequence.fused),
        }
        mosaics = {
            name: reader.read_frame(frame_index, crop=crop)
            for name, reader in readers.items()
        }
        result: dict[str, Any] = {
            "source": _prepare_tensor(
                mosaics["source"],
                cfa_pattern=sequence.source.cfa_pattern,
                origin=origin,
                black_level=self.black_source,
                normalize=self.normalize,
            ),
            "denoised": _prepare_tensor(
                mosaics["denoised"],
                cfa_pattern=sequence.denoised.cfa_pattern,
                origin=origin,
                black_level=self.black_dnr,
                normalize=self.normalize,
            ),
            "fused": _prepare_tensor(
                mosaics["fused"],
                cfa_pattern=sequence.fused.cfa_pattern,
                origin=origin,
                black_level=self.black_dnr,
                normalize=self.normalize,
            ),
            "sequence_id": sequence.sequence_id,
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "crop_origin": origin,
            "metadata": sequence.source.metadata,
        }
        if self.temporal_radius == 1:
            for name, neighbor_index in (
                ("source_prev", frame_index - 1),
                ("source_next", frame_index + 1),
            ):
                mosaic = readers["source"].read_frame(neighbor_index, crop=crop)
                result[name] = _prepare_tensor(
                    mosaic,
                    cfa_pattern=sequence.source.cfa_pattern,
                    origin=origin,
                    black_level=self.black_source,
                    normalize=self.normalize,
                )
        return result


def write_manifest(catalog: DatasetCatalog, path: str | Path) -> None:
    output = Path(path)
    output.write_text(
        json.dumps(catalog.manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover and validate the Phase2 dataset")
    parser.add_argument("root", type=Path, help="DATASET directory")
    parser.add_argument("--manifest", type=Path, help="Write the complete JSON manifest")
    parser.add_argument("--report", type=Path, help="Write the validation report")
    parser.add_argument("--deep", action="store_true", help="Read and verify all canonical data")
    args = parser.parse_args(argv)

    catalog = discover_dataset(args.root)
    report = validate_dataset(catalog, deep=args.deep)
    if args.manifest:
        write_manifest(catalog, args.manifest)
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {key: value for key, value in report.items() if key not in {"source_sha256"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Resumable A3 visual previews for every company-dataset RAW stream."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import torch

from raw_fusion.finetune.a4 import infer_video
from raw_fusion.finetune.adapters import SourceAdapter, parse_filename_metadata
from raw_fusion.finetune.stats import measure_black_levels
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, RawStreamReader, pack_frame


REPO_ROOT = Path(__file__).resolve().parents[3]
ISP_ROOT = REPO_ROOT / "ISP_traditional"
if not ISP_ROOT.is_dir():
    raise FileNotFoundError(f"required ISP directory is absent: {ISP_ROOT}")
if str(ISP_ROOT) not in sys.path:
    sys.path.insert(0, str(ISP_ROOT))

from opencv_source_match.pipeline import (  # noqa: E402
    demosaic_edge_aware,
    gamma_srgb,
    to_uint8,
)
from opencv_source_match.highlight_recovery import (  # noqa: E402
    reconstruct_camera_highlights,
)


WHITE_LEVEL_DN = 4095.0
_CFA_CODES = {
    "RGGB": [0, 1, 1, 2],
    "GRBG": [1, 0, 2, 1],
}
_CFA_PROJECT_TO_PHYSICAL = {
    "RGGB": (0, 1, 2, 3),
    "GRBG": (1, 0, 3, 2),
}


@dataclass(frozen=True, slots=True)
class PreviewSequence:
    """One unique stream selected for A3 visual inference."""

    name: str
    group: str
    category: str
    sensor: str
    path: Path
    width: int
    height: int
    frames: int
    bit_shift: int
    cfa: str
    role: str
    wb_rgb: tuple[float, float, float]
    black_levels: tuple[float, float, float, float] | None
    metadata_source: str

    @property
    def adapter(self) -> SourceAdapter:
        return SourceAdapter(
            group=self.group,
            scene=self.name,
            role=self.role,
            bit_shift=self.bit_shift,
            cfa=self.cfa,
            sensor_black_dn=256.0,
            reader_kind="single_file",
        )

    @property
    def packed_shape(self) -> tuple[int, int]:
        return self.height // 2, self.width // 2

    def validate(self, dataset_root: Path) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not self.path.resolve().is_relative_to(dataset_root.resolve()):
            raise ValueError(f"{self.name}: source is outside the read-only dataset root")
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError(f"{self.name}: dimensions must be positive and even")
        if self.frames < 8:
            raise ValueError(f"{self.name}: at least 8 frames are required")
        expected_bytes = self.width * self.height * 2 * self.frames
        if self.path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{self.name}: source size {self.path.stat().st_size} does not match "
                f"{self.frames} frames ({expected_bytes} bytes)"
            )
        if self.bit_shift not in (0, 4):
            raise ValueError(f"{self.name}: unsupported bit shift {self.bit_shift}")
        if self.cfa not in _CFA_CODES:
            raise ValueError(f"{self.name}: unsupported CFA {self.cfa}")
        if len(self.wb_rgb) != 3 or any(value <= 0.0 for value in self.wb_rgb):
            raise ValueError(f"{self.name}: invalid RGB white-balance gains")
        if self.black_levels is not None and len(self.black_levels) != 4:
            raise ValueError(f"{self.name}: black_levels must contain four values")

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path.resolve())
        payload["wb_rgb"] = list(self.wb_rgb)
        payload["black_levels"] = (
            list(self.black_levels) if self.black_levels is not None else None
        )
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _wb_from_metadata(metadata: dict[str, Any], name: str) -> tuple[float, float, float]:
    required = ("wb_r", "wb_g", "wb_b")
    if any(metadata.get(key) is None for key in required):
        raise ValueError(f"{name}: filename metadata does not contain complete white balance")
    green = float(metadata["wb_g"])
    if green <= 0.0:
        raise ValueError(f"{name}: green white-balance coefficient must be positive")
    return float(metadata["wb_r"]) / green, 1.0, float(metadata["wb_b"]) / green


def _sequence_from_a0_manifest(path: Path) -> PreviewSequence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    group = str(payload["group"])
    source = Path(payload["path"])
    if payload["reader_kind"] == "frame_directory":
        source = source.with_suffix(".raw")
    elif payload["reader_kind"] != "single_file":
        raise ValueError(f"{path}: unsupported reader kind {payload['reader_kind']}")
    category = "auxiliary_processed" if group == "03" else "raw_scene"
    if str(payload["role"]).startswith("calibration_"):
        category = "calibration"
    black = tuple(float(payload["black_levels"][name]) for name in CHANNEL_NAMES)
    return PreviewSequence(
        name=path.stem,
        group=group,
        category=category,
        sensor="MIS20S1",
        path=source,
        width=int(payload["width"]),
        height=int(payload["height"]),
        frames=int(payload["frames_total"]),
        bit_shift=int(payload["bit_shift"]),
        cfa=str(payload["cfa"]),
        role=str(payload["role"]),
        wb_rgb=_wb_from_metadata(payload["filename_metadata"], path.stem),
        black_levels=black,
        metadata_source=str(path.resolve()),
    )


def _sequence_from_cross_sensor(name: str, payload: dict[str, Any], config: Path) -> PreviewSequence:
    wb = tuple(float(value) for value in payload["fixed_isp_wb"])
    if len(wb) != 4:
        raise ValueError(f"{name}: fixed_isp_wb must contain four values")
    dataset_parts = Path(payload["path"]).parts
    group_name = next((part for part in dataset_parts if part[:2].isdigit()), "unknown")
    return PreviewSequence(
        name=name,
        group=group_name[:2],
        category="raw_scene",
        sensor=str(payload["sensor"]),
        path=Path(payload["path"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        frames=int(payload["frames_used"]),
        bit_shift=int(payload["bit_shift"]),
        cfa=str(payload["cfa"]),
        role="auxiliary",
        wb_rgb=(wb[0], 1.0, wb[3]),
        black_levels=None,
        metadata_source=str(config.resolve()),
    )


def _sequence_from_a8_candidate(payload: dict[str, Any], config: Path) -> PreviewSequence:
    path = Path(payload["path"])
    metadata = asdict(parse_filename_metadata(path))
    dataset_parts = path.parts
    group_name = next((part for part in dataset_parts if part[:2].isdigit()), "unknown")
    return PreviewSequence(
        name=str(payload["name"]),
        group=group_name[:2],
        category="raw_scene",
        sensor=str(payload["sensor"]),
        path=path,
        width=int(payload["width"]),
        height=int(payload["height"]),
        frames=int(payload["expected_frames"]),
        bit_shift=int(payload["bit_shift"]),
        cfa=str(payload["cfa"]),
        role="auxiliary",
        wb_rgb=_wb_from_metadata(metadata, str(payload["name"])),
        black_levels=None,
        metadata_source=str(config.resolve()),
    )


def discover_sequences(config: dict[str, Any], repo_root: Path = REPO_ROOT) -> list[PreviewSequence]:
    dataset_root = Path(config["dataset_root"])
    catalog = config["catalog"]
    sequences: list[PreviewSequence] = []

    manifest_dir = _relative_path(catalog["a0_manifest_dir"], repo_root)
    sequences.extend(
        _sequence_from_a0_manifest(path)
        for path in sorted(manifest_dir.glob("*.json"))
        if not path.name.startswith("03_")
    )
    for config_name in catalog["cross_sensor_configs"]:
        config_path = _relative_path(config_name, repo_root)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        sequences.extend(
            _sequence_from_cross_sensor(name, item, config_path)
            for name, item in payload["sequences"].items()
        )

    a8_path = _relative_path(catalog["a8_sensor_config"], repo_root)
    a8_payload = json.loads(a8_path.read_text(encoding="utf-8"))
    sequences.extend(
        _sequence_from_a8_candidate(item, a8_path)
        for item in a8_payload["candidate_sources"]
        if str(item["name"]).startswith("sc465sl_normal_")
    )

    by_source: dict[Path, PreviewSequence] = {}
    for sequence in sequences:
        key = sequence.path.resolve()
        if key in by_source:
            raise ValueError(
                f"duplicate source catalog entries: {by_source[key].name} and {sequence.name}"
            )
        sequence.validate(dataset_root)
        by_source[key] = sequence
    result = sorted(by_source.values(), key=lambda item: (item.group, item.category, item.name))
    expected = int(catalog["expected_unique_sequences"])
    if len(result) != expected:
        raise ValueError(f"catalog expected {expected} unique streams, found {len(result)}")
    return result


def clip_starts(frames: int, temporal_frames: int, count: int) -> tuple[int, ...]:
    if frames < temporal_frames:
        raise ValueError("stream is shorter than one temporal window")
    if count <= 0:
        raise ValueError("clip count must be positive")
    last = frames - temporal_frames
    values = np.linspace(0, last, count).round().astype(int)
    return tuple(dict.fromkeys(int(value) for value in values))


def resolve_black_levels(sequence: PreviewSequence, reader: RawStreamReader) -> np.ndarray:
    if sequence.black_levels is not None:
        return np.asarray(sequence.black_levels, dtype=np.float32)
    measured = measure_black_levels(reader, sequence.adapter, sample_frames=16)
    return np.asarray([measured[name] for name in CHANNEL_NAMES], dtype=np.float32)


def normalize_packed(frame_dn: np.ndarray, black: np.ndarray) -> np.ndarray:
    black_view = np.asarray(black, dtype=np.float32)[:, None, None]
    return np.clip(
        (frame_dn.astype(np.float32) - black_view) / (WHITE_LEVEL_DN - black_view),
        0.0,
        1.0,
    )


def denormalize_packed(frame: np.ndarray, black: np.ndarray) -> np.ndarray:
    black_view = np.asarray(black, dtype=np.float32)[:, None, None]
    return frame.astype(np.float32) * (WHITE_LEVEL_DN - black_view) + black_view


def canonical_to_mosaic(frame: np.ndarray, cfa: str) -> np.ndarray:
    """Restore a Bayer mosaic from project order [R, Gr, Gb, B]."""

    if frame.ndim != 3 or frame.shape[0] != 4:
        raise ValueError(f"expected [4,H,W], got {frame.shape}")
    order = _CFA_PROJECT_TO_PHYSICAL.get(cfa)
    if order is None:
        raise ValueError(f"unsupported CFA {cfa}")
    physical = frame[list(order)]
    height, width = physical.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = physical[0]
    mosaic[0::2, 1::2] = physical[1]
    mosaic[1::2, 0::2] = physical[2]
    mosaic[1::2, 1::2] = physical[3]
    return mosaic


def render_traditional_isp(
    frame_dn: np.ndarray,
    black: np.ndarray,
    wb_rgb: tuple[float, float, float],
    exposure_ev: float,
    cfa: str,
    *,
    highlight_recovery: bool = True,
    tone_map: str = "reinhard",
) -> np.ndarray:
    """Use ISP_traditional without claiming unavailable sensor color calibration."""

    raw01 = canonical_to_mosaic(normalize_packed(frame_dn, black), cfa)
    camera_rgb = demosaic_edge_aware(raw01, _CFA_CODES[cfa])
    if highlight_recovery:
        neutral = [1.0 / wb_rgb[0], 1.0, 1.0 / wb_rgb[2]]
        camera_rgb = reconstruct_camera_highlights(camera_rgb, neutral)
    camera_rgb *= np.asarray(wb_rgb, dtype=np.float32)[None, None]
    linear = camera_rgb * np.float32(2.0**exposure_ev)
    if tone_map == "reinhard":
        linear = linear / (1.0 + linear)
    elif tone_map == "hard-clip":
        linear = np.clip(linear, 0.0, 1.0)
    else:
        raise ValueError(f"unsupported tone map {tone_map}")
    return to_uint8(gamma_srgb(linear))


def estimate_exposure_ev(
    reader: RawStreamReader,
    sequence: PreviewSequence,
    black: np.ndarray,
    starts: tuple[int, ...],
    target: float,
    limits: tuple[float, float],
) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    packed_wb = np.asarray(
        [sequence.wb_rgb[0], 1.0, 1.0, sequence.wb_rgb[2]], dtype=np.float32
    )[:, None, None]
    for start in starts:
        for index in (start, start + 3, start + 7):
            packed = pack_frame(reader.read_frame(index), sequence.adapter).astype(np.float32)
            values = normalize_packed(packed, black) * packed_wb
            samples.append(values[:, ::16, ::16].reshape(-1))
    percentile = float(np.quantile(np.concatenate(samples), 0.995))
    raw_ev = math.log2(target / max(percentile, 1e-6))
    exposure_ev = float(np.clip(raw_ev, limits[0], limits[1]))
    return exposure_ev, percentile


def exposure_target_before_tone_map(output_target: float, tone_map: str) -> float:
    """Convert a desired post-tone-map linear level into its input level."""

    if not 0.0 < output_target < 1.0:
        raise ValueError("output exposure target must be strictly between zero and one")
    if tone_map == "reinhard":
        return output_target / (1.0 - output_target)
    if tone_map == "hard-clip":
        return output_target
    raise ValueError(f"unsupported tone map {tone_map}")


def _prediction_paths(sequence_dir: Path) -> tuple[Path, Path, Path]:
    inference_dir = sequence_dir / "inference"
    return (
        inference_dir / "prediction.npy",
        inference_dir / "progress.json",
        inference_dir / "manifest.json",
    )


def prediction_provenance(
    sequence: PreviewSequence,
    source_sha256: str,
    checkpoint_sha256: str,
    config_sha256: str,
    starts: tuple[int, ...],
    black: np.ndarray,
    tile_batch: int,
) -> dict[str, Any]:
    return {
        "sequence": sequence.to_mapping(),
        "source_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "weights": "ema",
        "clip_starts": list(starts),
        "temporal_frames": 8,
        "black_levels_R_Gr_Gb_B": [float(value) for value in black],
        "normalization": "clip((DN-black)/(4095-black),0,1)",
        "canonical_channel_order": ["R", "Gr", "Gb", "B"],
        "tile": 64,
        "stride": 56,
        "tile_batch": tile_batch,
    }


def infer_sequence(
    model: torch.nn.Module,
    device: torch.device,
    sequence: PreviewSequence,
    sequence_dir: Path,
    starts: tuple[int, ...],
    black: np.ndarray,
    provenance: dict[str, Any],
    tile_batch: int,
) -> Path:
    prediction_path, progress_path, manifest_path = _prediction_paths(sequence_dir)
    height, width = sequence.packed_shape
    shape = (len(starts) * 8, 4, height, width)
    resume_key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    completed = 0
    prediction: np.memmap
    if prediction_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        candidate = np.load(prediction_path, mmap_mode="r+")
        if (
            progress.get("resume_key") == resume_key
            and progress.get("shape") == list(shape)
            and candidate.shape == shape
            and candidate.dtype == np.float16
        ):
            prediction = candidate
            completed = int(progress.get("completed_clips", 0))
        else:
            prediction = np.lib.format.open_memmap(
                prediction_path, mode="w+", dtype=np.float16, shape=shape
            )
    else:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction = np.lib.format.open_memmap(
            prediction_path, mode="w+", dtype=np.float16, shape=shape
        )
    reader = RawStreamReader(sequence.path, sequence.width, sequence.height)
    for clip_index, start in enumerate(starts[completed:], start=completed):
        frames = [
            normalize_packed(
                pack_frame(reader.read_frame(start + offset), sequence.adapter), black
            )
            for offset in range(8)
        ]
        video = np.stack(frames).astype(np.float32)
        output = infer_video(model, video, device, tile_batch=tile_batch)
        offset = clip_index * 8
        prediction[offset : offset + 8] = output.astype(np.float16)
        prediction.flush()
        save_json(
            progress_path,
            {
                "resume_key": resume_key,
                "shape": list(shape),
                "completed_clips": clip_index + 1,
            },
        )
        print(
            f"{sequence.name}: inference clip {clip_index + 1}/{len(starts)} complete",
            flush=True,
        )
    save_json(
        manifest_path,
        {
            "provenance": provenance,
            "prediction": str(prediction_path.resolve()),
            "prediction_shape": list(shape),
            "prediction_dtype": "float16 normalized native domain",
            "prediction_sha256": sha256_file(prediction_path),
        },
    )
    return prediction_path


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _comparison_cell(image: np.ndarray, label: str) -> np.ndarray:
    overview = _fit_image(image, 960, 522)
    y0, y1 = image.shape[0] // 4, image.shape[0] * 3 // 4
    x0, x1 = image.shape[1] // 4, image.shape[1] * 3 // 4
    detail = _fit_image(image[y0:y1, x0:x1], 960, 554)
    separator = np.full((4, 960, 3), 24, dtype=np.uint8)
    cell = np.concatenate((overview, separator, detail), axis=0)
    cv2.rectangle(cell, (0, 0), (960, 46), (10, 10, 10), -1)
    cv2.putText(
        cell,
        label,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return cell


def video_encoder(preferred: str) -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    available = result.stdout + result.stderr
    if f" {preferred} " in available:
        return preferred
    for candidate in ("libx264", "libx265", "mpeg4"):
        if f" {candidate} " in available:
            return candidate
    raise RuntimeError("no supported ffmpeg video encoder is available")


def render_sequence_video(
    sequence: PreviewSequence,
    sequence_dir: Path,
    starts: tuple[int, ...],
    black: np.ndarray,
    exposure_ev: float,
    video_config: dict[str, Any],
    isp_config: dict[str, Any],
) -> dict[str, Any]:
    prediction_path, _, inference_manifest_path = _prediction_paths(sequence_dir)
    if not prediction_path.is_file() or not inference_manifest_path.is_file():
        raise FileNotFoundError(f"{sequence.name}: prediction artifacts are absent")
    prediction = np.load(prediction_path, mmap_mode="r")
    expected_shape = (len(starts) * 8, 4, *sequence.packed_shape)
    if prediction.shape != expected_shape:
        raise ValueError(f"{sequence.name}: expected {expected_shape}, got {prediction.shape}")
    reader = RawStreamReader(sequence.path, sequence.width, sequence.height)
    video_dir = sequence_dir / "traditional_isp_video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{sequence.name}_input_vs_a3_traditional_isp.mp4"
    preview_path = video_dir / f"{sequence.name}_frame0000.png"
    encoder = video_encoder(str(video_config["preferred_encoder"]))
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{video_config['width']}x{video_config['height']}",
        "-r",
        str(video_config["fps"]),
        "-i",
        "-",
        "-c:v",
        encoder,
    ]
    if encoder == "libx264":
        command.extend(
            ["-crf", str(video_config["crf"]), "-pix_fmt", video_config["pixel_format"]]
        )
    elif encoder == "libx265":
        command.extend(
            ["-x265-params", f"crf={video_config['crf']}", "-pix_fmt", video_config["pixel_format"]]
        )
    else:
        command.extend(["-q:v", "2", "-pix_fmt", "yuv420p"])
    command.append(str(video_path))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    first_frame: np.ndarray | None = None
    try:
        prediction_index = 0
        for clip_index, start in enumerate(starts):
            for local_index in range(8):
                source_index = start + local_index
                input_dn = pack_frame(
                    reader.read_frame(source_index), sequence.adapter
                ).astype(np.float32)
                prediction_dn = denormalize_packed(
                    np.asarray(prediction[prediction_index], dtype=np.float32), black
                )
                input_rgb = render_traditional_isp(
                    input_dn,
                    black,
                    sequence.wb_rgb,
                    exposure_ev,
                    sequence.cfa,
                    highlight_recovery=bool(isp_config["highlight_recovery"]),
                    tone_map=str(isp_config["tone_map"]),
                )
                prediction_rgb = render_traditional_isp(
                    prediction_dn,
                    black,
                    sequence.wb_rgb,
                    exposure_ev,
                    sequence.cfa,
                    highlight_recovery=bool(isp_config["highlight_recovery"]),
                    tone_map=str(isp_config["tone_map"]),
                )
                clip_marker = f"clip {clip_index + 1}/{len(starts)} | source frame {source_index}"
                frame = np.concatenate(
                    (
                        _comparison_cell(input_rgb, f"Input | {clip_marker}"),
                        _comparison_cell(prediction_rgb, f"A3 EMA | {clip_marker}"),
                    ),
                    axis=1,
                )
                if frame.shape != (1080, 1920, 3):
                    raise ValueError(f"unexpected video frame shape {frame.shape}")
                if first_frame is None:
                    first_frame = frame.copy()
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                prediction_index += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr[-3000:]}")
    except BaseException:
        process.kill()
        process.wait()
        video_path.unlink(missing_ok=True)
        raise
    if first_frame is None:
        raise RuntimeError(f"{sequence.name}: no video frames were rendered")
    if not cv2.imwrite(str(preview_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {preview_path}")
    manifest = {
        "sequence": sequence.to_mapping(),
        "frames": len(starts) * 8,
        "source_frame_indices": [
            start + offset for start in starts for offset in range(8)
        ],
        "fps": int(video_config["fps"]),
        "encoder": encoder,
        "pixel_format": video_config["pixel_format"] if encoder != "mpeg4" else "yuv420p",
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "preview": str(preview_path.resolve()),
        "preview_sha256": sha256_file(preview_path),
        "layout": isp_config["output_layout"],
        "isp": {
            **isp_config,
            "module_root": str(ISP_ROOT.resolve()),
            "black_levels_R_Gr_Gb_B": [float(value) for value in black],
            "wb_rgb": list(sequence.wb_rgb),
            "fixed_exposure_ev": exposure_ev,
            "input_and_a3_share_identical_parameters": True,
            "canonical_channel_order": ["R", "Gr", "Gb", "B"],
            "sensor_color_accuracy_claimed": False,
        },
        "ffmpeg_command": command,
        "limitations": [
            "No clean target is shown; this is a visual Input-versus-A3 comparison.",
            "Sensor ColorMatrix data is absent, so camera RGB color is not calibrated.",
            "Video compression and resize can conceal fine-grained noise differences.",
            "Exposure is frozen within each sequence but is not comparable across sequences.",
        ],
    }
    save_json(video_dir / "video_manifest.json", manifest)
    return manifest

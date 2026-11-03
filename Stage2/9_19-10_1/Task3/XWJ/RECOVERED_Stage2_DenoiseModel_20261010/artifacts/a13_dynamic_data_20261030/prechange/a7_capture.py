"""Read-only validation for the A7 real-motion capture pilot."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import BIT_SHIFT_HIGH_12, SourceAdapter
from raw_fusion.finetune.stream_io import RawStreamReader, pack_frame

CAPTURE_SCHEMA_VERSION = "stage2_a7_motion_pilot_v1"
REPORT_SCHEMA_VERSION = "stage2_a7_motion_validation_v1"
REQUIRED_MOTION_TYPES = frozenset(
    {
        "lateral_walk",
        "depth_walk",
        "fast_slow_gesture",
        "occlusion_reentry",
        "fine_texture_motion",
    }
)
_ALLOWED_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True, slots=True)
class CaptureClip:
    clip_id: str
    scene_id: str
    split: str
    motion_type: str
    path: str
    fps: float
    exposure_us: int
    sensor_gain: int
    isp_gain: int
    wb_r: int
    wb_g: int
    wb_b: int
    nr_disabled: bool
    auto_exposure: bool
    auto_gain: bool
    auto_white_balance: bool
    tripod_fixed: bool


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    sensor: str
    width: int
    height: int
    dtype: str
    bit_alignment: str
    cfa: str
    black_level_dn: float
    white_level_dn: int
    clips: tuple[CaptureClip, ...]


@dataclass(frozen=True, slots=True)
class CaptureValidationThresholds:
    min_frames_per_clip: int = 160
    min_train_scenes: int = 4
    min_validation_scenes: int = 2
    max_target_noise_dn: float = 8.0
    preferred_target_noise_dn: float = 5.0
    max_background_cv: float = 0.015
    max_background_deviation: float = 0.04
    max_saturation_fraction: float = 0.01
    min_active_pair_fraction: float = 0.25
    min_motion_fraction: float = 0.01
    max_motion_fraction: float = 0.60
    sample_pairs: int = 24
    noise_block_size: int = 16
    motion_block_size: int = 4


def load_capture_manifest(path: Path) -> CaptureManifest:
    """Load and strictly validate the capture contract JSON."""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CAPTURE_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )

    expected_format = {
        "sensor": "MIS20S1",
        "dtype": "uint16_le",
        "bit_alignment": "high12",
        "cfa": "RGGB",
        "white_level_dn": 4095,
    }
    for key, expected in expected_format.items():
        if payload.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}, got {payload.get(key)!r}")

    width = _positive_int(payload, "width")
    height = _positive_int(payload, "height")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("width and height must both be even for Bayer packing")
    black_level_dn = _positive_number(payload, "black_level_dn")

    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ValueError("clips must be a non-empty list")

    clips: list[CaptureClip] = []
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, dict):
            raise ValueError(f"clips[{index}] must be an object")
        split = _nonempty_string(raw_clip, "split")
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"clips[{index}].split must be one of {sorted(_ALLOWED_SPLITS)}")
        motion_type = _nonempty_string(raw_clip, "motion_type")
        if motion_type not in REQUIRED_MOTION_TYPES:
            raise ValueError(
                f"clips[{index}].motion_type must be one of {sorted(REQUIRED_MOTION_TYPES)}"
            )
        clips.append(
            CaptureClip(
                clip_id=_nonempty_string(raw_clip, "clip_id"),
                scene_id=_nonempty_string(raw_clip, "scene_id"),
                split=split,
                motion_type=motion_type,
                path=_nonempty_string(raw_clip, "path"),
                fps=_positive_number(raw_clip, "fps"),
                exposure_us=_positive_int(raw_clip, "exposure_us"),
                sensor_gain=_positive_int(raw_clip, "sensor_gain"),
                isp_gain=_positive_int(raw_clip, "isp_gain"),
                wb_r=_positive_int(raw_clip, "wb_r"),
                wb_g=_positive_int(raw_clip, "wb_g"),
                wb_b=_positive_int(raw_clip, "wb_b"),
                nr_disabled=_bool_field(raw_clip, "nr_disabled"),
                auto_exposure=_bool_field(raw_clip, "auto_exposure"),
                auto_gain=_bool_field(raw_clip, "auto_gain"),
                auto_white_balance=_bool_field(raw_clip, "auto_white_balance"),
                tripod_fixed=_bool_field(raw_clip, "tripod_fixed"),
            )
        )

    return CaptureManifest(
        sensor=payload["sensor"],
        width=width,
        height=height,
        dtype=payload["dtype"],
        bit_alignment=payload["bit_alignment"],
        cfa=payload["cfa"],
        black_level_dn=black_level_dn,
        white_level_dn=payload["white_level_dn"],
        clips=tuple(clips),
    )


def validate_capture_inventory(
    manifest: CaptureManifest,
    thresholds: CaptureValidationThresholds,
) -> dict:
    """Validate split isolation, scene counts, motion coverage, and capture settings."""

    failures: list[str] = []
    clip_ids = [clip.clip_id for clip in manifest.clips]
    if len(clip_ids) != len(set(clip_ids)):
        failures.append("clip_id values must be unique")
    paths = [clip.path for clip in manifest.clips]
    if len(paths) != len(set(paths)):
        failures.append("clip paths must be unique")

    gain_settings = sorted({(clip.sensor_gain, clip.isp_gain) for clip in manifest.clips})
    if len(gain_settings) != 1:
        failures.append(f"all clips must use one low-gain setting, got {gain_settings}")
    fps_values = sorted({clip.fps for clip in manifest.clips})
    if len(fps_values) != 1:
        failures.append(f"all clips must use one FPS, got {fps_values}")

    train_scenes = {clip.scene_id for clip in manifest.clips if clip.split == "train"}
    validation_scenes = {
        clip.scene_id for clip in manifest.clips if clip.split == "validation"
    }
    overlap = sorted(train_scenes & validation_scenes)
    if overlap:
        failures.append(f"train/validation scene leakage: {overlap}")
    if len(train_scenes) < thresholds.min_train_scenes:
        failures.append(
            f"need at least {thresholds.min_train_scenes} train scenes, got {len(train_scenes)}"
        )
    if len(validation_scenes) < thresholds.min_validation_scenes:
        failures.append(
            "need at least "
            f"{thresholds.min_validation_scenes} validation scenes, got {len(validation_scenes)}"
        )

    scene_motion_types: dict[str, set[str]] = {}
    for clip in manifest.clips:
        scene_motion_types.setdefault(clip.scene_id, set()).add(clip.motion_type)
        if not clip.nr_disabled:
            failures.append(f"{clip.clip_id}: nr_disabled must be true")
        if clip.auto_exposure:
            failures.append(f"{clip.clip_id}: auto_exposure must be false")
        if clip.auto_gain:
            failures.append(f"{clip.clip_id}: auto_gain must be false")
        if clip.auto_white_balance:
            failures.append(f"{clip.clip_id}: auto_white_balance must be false")
        if not clip.tripod_fixed:
            failures.append(f"{clip.clip_id}: tripod_fixed must be true")

    for scene_id, motion_types in sorted(scene_motion_types.items()):
        missing = sorted(REQUIRED_MOTION_TYPES - motion_types)
        if missing:
            failures.append(f"{scene_id}: missing motion types {missing}")

    return {
        "passed": not failures,
        "train_scenes": sorted(train_scenes),
        "validation_scenes": sorted(validation_scenes),
        "required_motion_types": sorted(REQUIRED_MOTION_TYPES),
        "gain_settings": [list(setting) for setting in gain_settings],
        "fps_values": fps_values,
        "scene_motion_types": {
            scene: sorted(motion_types)
            for scene, motion_types in sorted(scene_motion_types.items())
        },
        "failures": failures,
    }


def analyze_capture_clip(
    dataset_root: Path,
    manifest: CaptureManifest,
    clip: CaptureClip,
    thresholds: CaptureValidationThresholds,
) -> dict:
    """Measure one RAW clip without modifying the source file."""

    root = dataset_root.resolve()
    path = (root / clip.path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{clip.clip_id}: path escapes dataset root: {clip.path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"{clip.clip_id}: missing RAW file {path}")

    reader = RawStreamReader(path, manifest.width, manifest.height)
    adapter = SourceAdapter(
        group="A7",
        scene=clip.scene_id,
        role="reference",
        bit_shift=BIT_SHIFT_HIGH_12,
        cfa=manifest.cfa,
        sensor_black_dn=manifest.black_level_dn,
        reader_kind="single_file",
    )
    pair_indices = np.linspace(
        0,
        max(0, reader.frames - 2),
        min(thresholds.sample_pairs, max(1, reader.frames - 1)),
    ).round().astype(int)
    pair_indices = np.unique(pair_indices)

    alignment_samples = 0
    aligned_samples = 0
    saturated_samples = 0
    total_samples = 0
    pair_noise: list[float] = []
    motion_fractions: list[float] = []
    background_levels: list[float] = []
    detail_energy: list[float] = []

    for index in pair_indices:
        raw_a = reader.read_frame(int(index))
        raw_b = reader.read_frame(int(index) + 1)
        for raw in (raw_a, raw_b):
            sampled = raw[::17, ::19]
            alignment_samples += sampled.size
            aligned_samples += int((sampled % 16 == 0).sum())

        packed_a = pack_frame(raw_a, adapter).astype(np.float32)
        packed_b = pack_frame(raw_b, adapter).astype(np.float32)
        for packed in (packed_a, packed_b):
            saturated_samples += int((packed >= manifest.white_level_dn).sum())
            total_samples += packed.size
            luma = packed.mean(axis=0)
            background_levels.append(
                float(np.percentile(_block_means(luma, thresholds.motion_block_size), 20.0))
            )
            detail_energy.append(_detail_energy(luma))

        noise_dn = _quiet_pair_noise(packed_a, packed_b, thresholds.noise_block_size)
        pair_noise.append(noise_dn)
        motion_fractions.append(
            _motion_fraction(
                packed_a,
                packed_b,
                noise_dn,
                thresholds.motion_block_size,
            )
        )

    background = np.asarray(background_levels, dtype=np.float64)
    background_median = float(np.median(background))
    background_cv = float(np.std(background) / max(background.mean(), 1e-6))
    background_max_deviation = float(
        np.max(np.abs(background - background_median)) / max(background_median, 1e-6)
    )
    target_noise_dn = float(np.median(pair_noise))
    mean_motion_fraction = float(np.mean(motion_fractions))
    active_pair_fraction = float(
        np.mean(np.asarray(motion_fractions) >= thresholds.min_motion_fraction)
    )
    alignment_fraction = aligned_samples / alignment_samples
    saturation_fraction = saturated_samples / total_samples

    checks = {
        "minimum_frames": reader.frames >= thresholds.min_frames_per_clip,
        "high12_alignment": alignment_fraction == 1.0,
        "target_noise": target_noise_dn <= thresholds.max_target_noise_dn,
        "background_cv": background_cv <= thresholds.max_background_cv,
        "background_max_deviation": (
            background_max_deviation <= thresholds.max_background_deviation
        ),
        "saturation": saturation_fraction <= thresholds.max_saturation_fraction,
        "active_motion": active_pair_fraction >= thresholds.min_active_pair_fraction,
        "motion_area_lower_bound": mean_motion_fraction >= thresholds.min_motion_fraction,
        "motion_area_upper_bound": mean_motion_fraction <= thresholds.max_motion_fraction,
    }
    failures = [name for name, passed in checks.items() if not passed]
    warnings: list[str] = []
    if thresholds.preferred_target_noise_dn < target_noise_dn <= thresholds.max_target_noise_dn:
        warnings.append(
            f"target noise {target_noise_dn:.3f} DN passes the hard gate but exceeds "
            f"the preferred {thresholds.preferred_target_noise_dn:.3f} DN"
        )

    return {
        "clip_id": clip.clip_id,
        "scene_id": clip.scene_id,
        "split": clip.split,
        "motion_type": clip.motion_type,
        "path": str(path),
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "frames": reader.frames,
            "sampled_adjacent_pairs": len(pair_indices),
            "high12_alignment_fraction": alignment_fraction,
            "target_noise_dn": target_noise_dn,
            "background_level_dn": background_median,
            "background_cv": background_cv,
            "background_max_deviation": background_max_deviation,
            "saturation_fraction": saturation_fraction,
            "active_pair_fraction": active_pair_fraction,
            "mean_motion_fraction": mean_motion_fraction,
            "detail_energy_dn": float(np.median(detail_energy)),
        },
    }


def validate_capture_dataset(
    manifest_path: Path,
    dataset_root: Path | None = None,
    thresholds: CaptureValidationThresholds | None = None,
) -> dict:
    """Validate the full A7 capture pilot and return a JSON-serializable report."""

    limits = thresholds or CaptureValidationThresholds()
    manifest = load_capture_manifest(manifest_path)
    root = dataset_root if dataset_root is not None else manifest_path.parent
    inventory = validate_capture_inventory(manifest, limits)
    clips = [analyze_capture_clip(root, manifest, clip, limits) for clip in manifest.clips]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "dataset_root": str(root.resolve()),
        "overall_pass": inventory["passed"] and all(clip["passed"] for clip in clips),
        "capture_format": {
            key: value for key, value in asdict(manifest).items() if key != "clips"
        },
        "thresholds": asdict(limits),
        "inventory": inventory,
        "clips": clips,
    }


def write_validation_report(report: dict, path: Path) -> None:
    """Atomically write the validation report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".a7-validation.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _positive_int(payload: dict, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_number(payload: dict, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


def _nonempty_string(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _bool_field(payload: dict, key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _block_means(image: np.ndarray, block: int) -> np.ndarray:
    height = image.shape[-2] // block * block
    width = image.shape[-1] // block * block
    if height == 0 or width == 0:
        raise ValueError(f"image {image.shape[-2:]} is smaller than block size {block}")
    cropped = image[..., :height, :width]
    new_shape = cropped.shape[:-2] + (height // block, block, width // block, block)
    return cropped.reshape(new_shape).mean(axis=(-3, -1))


def _quiet_pair_noise(first: np.ndarray, second: np.ndarray, block: int) -> float:
    difference = first - second
    height = difference.shape[-2] // block * block
    width = difference.shape[-1] // block * block
    cropped = difference[:, :height, :width]
    blocks = cropped.reshape(4, height // block, block, width // block, block)
    block_sigma = blocks.std(axis=(2, 4), ddof=1) / np.sqrt(2.0)
    return float(np.percentile(block_sigma, 20.0))


def _motion_fraction(
    first: np.ndarray,
    second: np.ndarray,
    noise_dn: float,
    block: int,
) -> float:
    first_luma = _block_means(first.mean(axis=0), block)
    second_luma = _block_means(second.mean(axis=0), block)
    block_noise = noise_dn / np.sqrt(4.0 * block * block)
    threshold = max(2.0, 5.0 * np.sqrt(2.0) * block_noise)
    return float((np.abs(first_luma - second_luma) > threshold).mean())


def _detail_energy(image: np.ndarray) -> float:
    center = image[1:-1, 1:-1]
    laplacian = (
        4.0 * center
        - image[:-2, 1:-1]
        - image[2:, 1:-1]
        - image[1:-1, :-2]
        - image[1:-1, 2:]
    )
    return float(np.median(np.abs(laplacian)))

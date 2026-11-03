"""Training-source eligibility checks for A8 sensor adaptation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


_ALLOWED_TARGET_SEMANTICS = {
    "clean_reference",
    "independent_noise2noise",
    "static_disjoint_reference",
}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTHORITATIVE_DATASET_ROOT = Path(
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset"
)
_AUTHORITATIVE_VALIDATION_PATH = (
    _REPO_ROOT / "artifacts/a3_cross_sensor_validation_final_20261012.json"
)
_AUTHORITATIVE_VALIDATION_SHA256 = (
    "14538b3495af329252c075c65b71d0c2e638c9eda28bde632543d10d40cb8f64"
)
_AUTHORITATIVE_FROZEN_SOURCES: dict[str, tuple[str, int, str]] = {
    "sc465sl_darkroom18_256x": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/02_SC465SL_darkroom18/256x/raw_stream_sc465sl_2560x1440_AG109568_darkroom18_256x.raw",
        2211840000,
        "cec46afe5696c9e2bfc569670a0cc93814d4ad851677275fd4f77c274e6c35cd",
    ),
    "sc465sl_darkroom18_2000x": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/02_SC465SL_darkroom18/2000x/raw_stream_sc465sl_2560x1440_AG109568_darkroom18_2000x.raw",
        2211840000,
        "8df12bc10c7b2191791d1d7f4a9715167a6b6c1a6a1805554ef27b2b05aef7b0",
    ),
    "sc635hai_131x_low_cct": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc635hai/131x_low_cct/raw_stream_normal_3200x1800_16_RG_0205213025917_131x.raw",
        1728000000,
        "80445b9133f49dfa7f5117d93bcf9fd5a9dfae1de0e0faae69c9f4144dac440a",
    ),
    "sc635hai_16x_motion_color": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc635hai/16x_motion_color/3200x1800_RG_motion_color_16x.raw",
        2304000000,
        "da64b7a39af359bdd5aa3f16b6456458f620492e57230ac5fee4a57012b2896c",
    ),
    "sc450ai_1435x_motion_black_object": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc450ai/1435x_motion_black_object/raw_stream_2688x1520_16bit@RG[Shutter=83328,SenserG=61440,IspG=24493,R=1620,G=1024,B=2409][bl_R=4128_Gr=4128_Gb=4128_B=4128].raw",
        1716019200,
        "7896fd9346aa8f82b67d7d2a3a02b4929d9ff58e6da952b4ff7aaf3a7e0acba0",
    ),
    "sc450ai_804x_lightchanging": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc450ai/804x_lightchanging/lightchanging_804x.raw",
        1225728000,
        "8e74a2b96de1943a0180eba82650613d98c0398d0e984c3d0ac4c6b1e12359fe",
    ),
    "sc465sl_difficultpattern_4717": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_difficultpattern/shutter_83315/sensorG__ispG_4717_WB_1930_1024_2381/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=4717,R=1930,G=1024,B=2381].raw",
        1474560000,
        "b9a4b244e049af9b167ec4e67dc83c951b271d3f9ae077ed74944e9cb98a09d6",
    ),
    "sc465sl_difficultpattern_8880": (
        "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_difficultpattern/shutter_83315/sensorG__ispG_8880_WB_1962_1024_2172/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=8880,R=1962,G=1024,B=2172].raw",
        1474560000,
        "5551672d7d9c5c12b437b7211515b03fe9249531add833242ed520a945c04f41",
    ),
}
_AUTHORITATIVE_REQUIRED_SENSORS = ("sc635hai", "sc450ai")
_AUTHORITATIVE_CANDIDATE_SENSORS: dict[str, str] = {
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc635hai/131x_low_cct/raw_stream_normal_3200x1800_16_RG_0205213025917_131x.raw": "sc635hai",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc635hai/16x_motion_color/3200x1800_RG_motion_color_16x.raw": "sc635hai",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc450ai/1435x_motion_black_object/raw_stream_2688x1520_16bit@RG[Shutter=83328,SenserG=61440,IspG=24493,R=1620,G=1024,B=2409][bl_R=4128_Gr=4128_Gb=4128_B=4128].raw": "sc450ai",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/01_fist_company/sc450ai/804x_lightchanging/lightchanging_804x.raw": "sc450ai",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_39999/sensorG__ispG_1024_WB_1535_1024_3003/raw_stream_2560x1440_16bit@RG_[Shutter=39999,SenserG=1079,IspG=1024,R=1535,G=1024,B=3003].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_39999/sensorG__ispG_1223_WB_1960_1024_2402/raw_stream_2560x1440_16bit@RG_[Shutter=39999,SenserG=109572,IspG=1223,R=1960,G=1024,B=2402].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_83315/sensorG__ispG_2391_WB_1945_1024_2433/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=2391,R=1945,G=1024,B=2433].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_83315/sensorG__ispG_3560_WB_1976_1024_2365/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=3560,R=1976,G=1024,B=2365].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_83315/sensorG__ispG_5023_WB_1949_1024_2208/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=5023,R=1949,G=1024,B=2208].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_83315/sensorG__ispG_6373_WB_1949_1024_2208/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=6373,R=1949,G=1024,B=2208].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_83315/sensorG__ispG_9798_WB_1970_1024_2143/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=9798,R=1970,G=1024,B=2143].raw": "SC465SL",
    "/HardDisk/jingzeliang/Data/Company_Dataset/dataset/05_2DNR3DNR/465_sh_normal/shutter_83315/sensorG__ispG_19598_WB_2003_1024_1964/raw_stream_2560x1440_16bit@RG_[Shutter=83315,SenserG=109572,IspG=19598,R=2003,G=1024,B=1964].raw": "SC465SL",
}
_AUTHORITATIVE_APPROVED_TRAINING_SOURCES: dict[
    str,
    tuple[str, str, str, str],
] = {}


@dataclass(frozen=True, slots=True)
class A8SourceSpec:
    """Explicit metadata and supervision semantics for one candidate RAW stream."""

    name: str
    sensor: str
    path: Path
    width: int
    height: int
    expected_frames: int
    bit_shift: int
    cfa: str
    target_semantics: str
    approved_for_backprop: bool
    reason: str
    supervision_manifest: Path | None
    supervision_manifest_sha256: str | None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "A8SourceSpec":
        def require_string(key: str) -> str:
            value = payload[key]
            if type(value) is not str or not value:
                raise TypeError(f"{key} must be a non-empty JSON string")
            return value

        def require_integer(key: str) -> int:
            value = payload[key]
            if type(value) is not int:
                raise TypeError(f"{key} must be a JSON integer")
            return value

        approved_for_backprop = payload["approved_for_backprop"]
        if type(approved_for_backprop) is not bool:
            raise TypeError("approved_for_backprop must be a JSON boolean")
        supervision_manifest_value = payload.get("supervision_manifest")
        supervision_manifest_sha256 = payload.get("supervision_manifest_sha256")
        if approved_for_backprop:
            if type(supervision_manifest_value) is not str or not supervision_manifest_value:
                raise TypeError(
                    "approved sources require a non-empty supervision_manifest string"
                )
            if type(supervision_manifest_sha256) is not str:
                raise TypeError(
                    "approved sources require a supervision_manifest_sha256 string"
                )
        elif supervision_manifest_value is not None or supervision_manifest_sha256 is not None:
            raise ValueError("unapproved sources must not declare supervision evidence")
        spec = cls(
            name=require_string("name"),
            sensor=require_string("sensor"),
            path=Path(require_string("path")),
            width=require_integer("width"),
            height=require_integer("height"),
            expected_frames=require_integer("expected_frames"),
            bit_shift=require_integer("bit_shift"),
            cfa=require_string("cfa"),
            target_semantics=require_string("target_semantics"),
            approved_for_backprop=approved_for_backprop,
            reason=require_string("reason"),
            supervision_manifest=(
                Path(supervision_manifest_value)
                if supervision_manifest_value is not None
                else None
            ),
            supervision_manifest_sha256=supervision_manifest_sha256,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.name or not self.sensor:
            raise ValueError("source name and sensor must be non-empty")
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError(f"{self.name}: width and height must be positive and even")
        if self.expected_frames <= 0:
            raise ValueError(f"{self.name}: expected_frames must be positive")
        if self.bit_shift not in {0, 4}:
            raise ValueError(f"{self.name}: bit_shift must be 0 or 4")
        if self.cfa not in {"RGGB", "GRBG"}:
            raise ValueError(f"{self.name}: unsupported CFA {self.cfa!r}")
        if not self.target_semantics or not self.reason:
            raise ValueError(f"{self.name}: target semantics and reason are required")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{field} must be a 64-character SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} is not hexadecimal") from error
    return value.lower()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def load_frozen_audit_sources(validation_path: Path) -> dict[Path, dict[str, Any]]:
    """Load the source manifests covered by the completed cross-sensor audit."""

    validation = _load_json(validation_path)
    roots = validation.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("cross-sensor validation must contain non-empty roots")

    frozen: dict[Path, dict[str, Any]] = {}
    seen_sequences: set[str] = set()
    for root_entry in roots:
        if not isinstance(root_entry, dict):
            raise TypeError("cross-sensor validation root entry must be an object")
        if root_entry.get("require_complete") is not True:
            raise ValueError("frozen audit root was not validated in complete mode")
        if root_entry.get("verify_sources") is not True:
            raise ValueError("frozen audit root did not verify source hashes")
        sequences = root_entry.get("sequences")
        if not isinstance(sequences, dict) or not sequences:
            raise ValueError("frozen audit root must contain sequences")

        root_value = root_entry.get("root")
        if type(root_value) is not str or not root_value:
            raise TypeError("frozen audit root path must be a non-empty string")
        root = Path(root_value).resolve()
        for sequence_name, sequence_entry in sequences.items():
            if type(sequence_name) is not str or not isinstance(sequence_entry, dict):
                raise TypeError("frozen sequence entries must be named JSON objects")
            if sequence_entry.get("complete") is not True:
                raise ValueError(f"{sequence_name}: frozen validation is incomplete")
            policy = _AUTHORITATIVE_FROZEN_SOURCES.get(sequence_name)
            if policy is None:
                raise ValueError(f"{sequence_name}: sequence is absent from frozen policy")
            if sequence_name in seen_sequences:
                raise ValueError(f"{sequence_name}: duplicate frozen sequence")
            seen_sequences.add(sequence_name)
            policy_path = Path(policy[0]).resolve()
            policy_size = policy[1]
            policy_sha256 = policy[2]
            expected_source_sha256 = _require_sha256(
                sequence_entry.get("source_sha256"),
                f"{sequence_name}.source_sha256",
            )
            data_report_value = sequence_entry.get("data_report")
            if type(data_report_value) is not str or not data_report_value:
                raise TypeError(f"{sequence_name}: data_report must be a string")
            data_report = Path(data_report_value).resolve()
            if not data_report.is_file():
                raise FileNotFoundError(data_report)
            sequence_dir = data_report.parents[1]
            if sequence_dir.name != sequence_name:
                raise ValueError(
                    f"{sequence_name}: data report belongs to {sequence_dir.name!r}"
                )
            if not sequence_dir.is_relative_to(root / "sequences"):
                raise ValueError(f"{sequence_name}: data report is outside its audit root")
            manifest_path = sequence_dir / "source_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = _load_json(manifest_path)
            source_path_value = manifest.get("source_path")
            if type(source_path_value) is not str or not source_path_value:
                raise TypeError(f"{sequence_name}: source_path must be a string")
            source_path = Path(source_path_value).resolve()
            if source_path != policy_path:
                raise ValueError(f"{sequence_name}: source path differs from frozen policy")
            if source_path in frozen:
                raise ValueError(f"duplicate frozen source path: {source_path}")
            manifest_source_sha256 = _require_sha256(
                manifest.get("source_sha256"),
                f"{sequence_name}.manifest.source_sha256",
            )
            if manifest_source_sha256 != expected_source_sha256:
                raise ValueError(f"{sequence_name}: validation and manifest SHA-256 differ")
            if manifest_source_sha256 != policy_sha256:
                raise ValueError(f"{sequence_name}: source SHA-256 differs from frozen policy")
            report = _load_json(data_report)
            provenance = report.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{sequence_name}: data report has no provenance")
            for key in ("source_path", "source_size_bytes", "source_sha256"):
                if provenance.get(key) != manifest.get(key):
                    raise ValueError(
                        f"{sequence_name}: report and source manifest differ for {key}"
                    )
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_stat = source_path.stat()
            source_size_bytes = manifest.get("source_size_bytes")
            if type(source_size_bytes) is not int:
                raise TypeError(
                    f"{sequence_name}: source_size_bytes must be a JSON integer"
                )
            if source_size_bytes != policy_size:
                raise ValueError(f"{sequence_name}: source size differs from frozen policy")
            if source_stat.st_size != source_size_bytes:
                raise ValueError(f"{sequence_name}: frozen source size has changed")
            actual_source_sha256 = _sha256(source_path)
            if actual_source_sha256 != manifest_source_sha256:
                raise ValueError(f"{sequence_name}: frozen source SHA-256 has changed")
            frozen[source_path] = {
                "source_sha256": manifest_source_sha256,
                "source_size_bytes": source_size_bytes,
                "device": source_stat.st_dev,
                "inode": source_stat.st_ino,
                "manifest": str(manifest_path.resolve()),
                "data_report": str(data_report),
            }
    expected_sequences = set(_AUTHORITATIVE_FROZEN_SOURCES)
    if seen_sequences != expected_sequences:
        missing = sorted(expected_sequences - seen_sequences)
        extra = sorted(seen_sequences - expected_sequences)
        raise ValueError(f"frozen validation sequence mismatch: missing={missing}, extra={extra}")
    return frozen


def _frame_indices(ranges: Any, field: str, frame_count: int) -> set[int]:
    if not isinstance(ranges, list) or not ranges:
        raise TypeError(f"{field} must be a non-empty list of [start, end) ranges")
    indices: set[int] = set()
    for item in ranges:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(f"{field} entries must be [start, end) lists")
        start, end = item
        if type(start) is not int or type(end) is not int:
            raise TypeError(f"{field} range values must be JSON integers")
        if not 0 <= start < end <= frame_count:
            raise ValueError(f"{field} range [{start}, {end}) is outside the source")
        indices.update(range(start, end))
    return indices


def _validate_supervision_evidence(
    spec: A8SourceSpec,
    source_sha256: str,
) -> dict[str, Any]:
    if spec.supervision_manifest is None or spec.supervision_manifest_sha256 is None:
        raise ValueError(f"{spec.name}: supervision evidence is absent")
    manifest_path = spec.supervision_manifest.resolve()
    if not manifest_path.is_relative_to(_REPO_ROOT):
        raise ValueError(f"{spec.name}: supervision manifest is outside the repository")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected_manifest_sha256 = _require_sha256(
        spec.supervision_manifest_sha256,
        f"{spec.name}.supervision_manifest_sha256",
    )
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(f"{spec.name}: supervision manifest SHA-256 mismatch")

    evidence = _load_json(manifest_path)
    if evidence.get("schema_version") != "a8-supervision-v1":
        raise ValueError(f"{spec.name}: unsupported supervision evidence schema")
    if Path(evidence.get("source_path", "")).resolve() != spec.path.resolve():
        raise ValueError(f"{spec.name}: supervision evidence names another source")
    if evidence.get("source_sha256") != source_sha256:
        raise ValueError(f"{spec.name}: supervision evidence source hash differs")
    if evidence.get("target_semantics") != spec.target_semantics:
        raise ValueError(f"{spec.name}: supervision target semantics differ")
    if evidence.get("independent_target") is not True:
        raise ValueError(f"{spec.name}: supervision target is not declared independent")
    allowed_origins = {
        "clean_reference": "direct_clean_capture",
        "independent_noise2noise": "disjoint_same_sensor_frames",
        "static_disjoint_reference": "disjoint_same_sensor_frames",
    }
    if evidence.get("label_origin") != allowed_origins[spec.target_semantics]:
        raise ValueError(f"{spec.name}: supervision label origin is not approved")
    if evidence.get("old_teacher_used") is not False:
        raise ValueError(f"{spec.name}: old-teacher supervision is forbidden")
    if evidence.get("temporal_average_used_as_dynamic_truth") is not False:
        raise ValueError(f"{spec.name}: dynamic temporal-average truth is forbidden")

    if spec.target_semantics in {"independent_noise2noise", "static_disjoint_reference"}:
        input_indices = _frame_indices(
            evidence.get("input_frame_ranges"),
            "input_frame_ranges",
            spec.expected_frames,
        )
        target_indices = _frame_indices(
            evidence.get("target_frame_ranges"),
            "target_frame_ranges",
            spec.expected_frames,
        )
        if input_indices & target_indices:
            raise ValueError(f"{spec.name}: input and target frame pools overlap")

    target_artifacts = evidence.get("target_artifacts")
    if not isinstance(target_artifacts, list) or not target_artifacts:
        raise ValueError(f"{spec.name}: supervision evidence has no target artifacts")
    verified_artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(target_artifacts):
        if not isinstance(item, dict):
            raise TypeError(f"{spec.name}: target artifact {index} must be an object")
        artifact_path_value = item.get("path")
        if type(artifact_path_value) is not str or not artifact_path_value:
            raise TypeError(f"{spec.name}: target artifact {index} path is invalid")
        artifact_path = Path(artifact_path_value).resolve()
        if not artifact_path.is_relative_to(_REPO_ROOT):
            raise ValueError(f"{spec.name}: target artifact {index} is outside the repository")
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if item.get("role") != "clean_target" or item.get("format") != "npy":
            raise ValueError(f"{spec.name}: target artifact {index} contract is invalid")
        if artifact_path.suffix != ".npy":
            raise ValueError(f"{spec.name}: target artifact {index} must be NPY")
        if item.get("dtype") != "float32":
            raise ValueError(f"{spec.name}: target artifact {index} must be float32")
        declared_shape = item.get("shape")
        if (
            not isinstance(declared_shape, list)
            or not declared_shape
            or any(type(value) is not int or value <= 0 for value in declared_shape)
        ):
            raise TypeError(f"{spec.name}: target artifact {index} shape is invalid")
        expected_sha256 = _require_sha256(
            item.get("sha256"),
            f"{spec.name}.target_artifacts[{index}].sha256",
        )
        actual_sha256 = _sha256(artifact_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{spec.name}: target artifact {index} SHA-256 mismatch")
        if actual_sha256 == source_sha256:
            raise ValueError(
                f"{spec.name}: target artifact {index} duplicates the noisy source"
            )
        target = np.load(artifact_path, mmap_mode="r", allow_pickle=False)
        if str(target.dtype) != "float32" or list(target.shape) != declared_shape:
            raise ValueError(f"{spec.name}: target artifact {index} array contract differs")
        if target.ndim == 3:
            channel_count = target.shape[0]
        elif target.ndim == 4:
            channel_count = target.shape[1]
        else:
            raise ValueError(f"{spec.name}: target artifact {index} must be 3D or 4D")
        if channel_count != 4:
            raise ValueError(f"{spec.name}: target artifact {index} must have four channels")
        verified_artifacts.append(
            {"path": str(artifact_path), "sha256": actual_sha256}
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "target_artifacts": verified_artifacts,
    }


def audit_training_sources(config_path: Path) -> dict[str, Any]:
    """Classify candidate streams without reading or modifying their payload data."""

    config = _load_json(config_path)
    dataset_root = Path(config["dataset_root"]).resolve()
    if dataset_root != _AUTHORITATIVE_DATASET_ROOT.resolve():
        raise ValueError(f"dataset_root must be {_AUTHORITATIVE_DATASET_ROOT}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    validation_path = Path(config["frozen_validation"]).resolve()
    if validation_path != _AUTHORITATIVE_VALIDATION_PATH.resolve():
        raise ValueError(
            f"frozen_validation must be {_AUTHORITATIVE_VALIDATION_PATH}"
        )
    if _sha256(validation_path) != _AUTHORITATIVE_VALIDATION_SHA256:
        raise ValueError("authoritative frozen validation SHA-256 mismatch")
    required_sensor_values = config["required_sensors"]
    if not isinstance(required_sensor_values, list):
        raise TypeError("required_sensors must be a JSON list")
    if any(type(value) is not str for value in required_sensor_values):
        raise TypeError("required_sensors entries must be JSON strings")
    required_sensors = tuple(required_sensor_values)
    if not required_sensors:
        raise ValueError("required_sensors must be non-empty")
    if any(not sensor for sensor in required_sensors):
        raise ValueError("required_sensors must not contain empty names")
    if len(set(required_sensors)) != len(required_sensors):
        raise ValueError("required_sensors must not contain duplicates")
    if required_sensors != _AUTHORITATIVE_REQUIRED_SENSORS:
        raise ValueError(
            f"required_sensors must be {list(_AUTHORITATIVE_REQUIRED_SENSORS)}"
        )

    frozen = load_frozen_audit_sources(validation_path)
    candidate_values = config["candidate_sources"]
    if not isinstance(candidate_values, list):
        raise TypeError("candidate_sources must be a JSON list")
    if any(not isinstance(item, dict) for item in candidate_values):
        raise TypeError("candidate_sources entries must be JSON objects")
    specs = [A8SourceSpec.from_mapping(item) for item in candidate_values]
    if not specs:
        raise ValueError("candidate_sources must be non-empty")

    records: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    frozen_inodes = {
        (entry["device"], entry["inode"])
        for entry in frozen.values()
    }
    frozen_hashes = {entry["source_sha256"] for entry in frozen.values()}
    for spec in specs:
        path = spec.path.resolve()
        if spec.name in seen_names:
            raise ValueError(f"duplicate candidate source name: {spec.name}")
        if path in seen_paths:
            raise ValueError(f"duplicate candidate source path: {path}")
        seen_names.add(spec.name)
        seen_paths.add(path)
        if not path.is_relative_to(dataset_root):
            raise ValueError(f"{spec.name}: source is outside the read-only dataset root")
        authoritative_sensor = _AUTHORITATIVE_CANDIDATE_SENSORS.get(str(path))
        if authoritative_sensor is None:
            raise ValueError(f"{spec.name}: source is absent from candidate policy")
        if spec.sensor != authoritative_sensor:
            raise ValueError(f"{spec.name}: sensor differs from candidate policy")
        if not path.is_file():
            raise FileNotFoundError(path)

        source_stat = path.stat()
        size_bytes = source_stat.st_size
        bytes_per_frame = spec.width * spec.height * 2
        if size_bytes % bytes_per_frame:
            raise ValueError(
                f"{spec.name}: size {size_bytes} is not divisible by {bytes_per_frame}"
            )
        actual_frames = size_bytes // bytes_per_frame
        if actual_frames != spec.expected_frames:
            raise ValueError(
                f"{spec.name}: expected {spec.expected_frames} frames, found {actual_frames}"
            )

        frozen_entry = frozen.get(path)
        frozen_inode_alias = (source_stat.st_dev, source_stat.st_ino) in frozen_inodes
        if frozen_entry is not None and frozen_entry["source_size_bytes"] != size_bytes:
            raise ValueError(
                f"{spec.name}: frozen source size no longer matches its verified manifest"
            )
        source_sha256: str | None = None
        supervision_evidence: dict[str, Any] | None = None
        if frozen_entry is not None or frozen_inode_alias:
            status = "blocked_frozen_audit_source"
        elif not spec.approved_for_backprop:
            status = "blocked_reference_semantics"
        elif spec.target_semantics not in _ALLOWED_TARGET_SEMANTICS:
            status = "blocked_unapproved_target_semantics"
        else:
            source_sha256 = _sha256(path)
            if source_sha256 in frozen_hashes:
                status = "blocked_frozen_content_alias"
            else:
                training_policy = _AUTHORITATIVE_APPROVED_TRAINING_SOURCES.get(
                    str(path)
                )
                if training_policy is None:
                    status = "blocked_not_in_authoritative_training_policy"
                else:
                    policy_sensor, policy_semantics, policy_manifest, policy_sha256 = (
                        training_policy
                    )
                    if spec.sensor != policy_sensor or spec.target_semantics != policy_semantics:
                        raise ValueError(f"{spec.name}: source differs from training policy")
                    if spec.supervision_manifest is None:
                        raise ValueError(f"{spec.name}: supervision manifest is absent")
                    if spec.supervision_manifest.resolve() != Path(policy_manifest).resolve():
                        raise ValueError(f"{spec.name}: supervision manifest differs from policy")
                    if spec.supervision_manifest_sha256 != policy_sha256:
                        raise ValueError(f"{spec.name}: supervision hash differs from policy")
                    supervision_evidence = _validate_supervision_evidence(
                        spec,
                        source_sha256,
                    )
                    status = "trainable"

        sensor_summary = summary.setdefault(
            spec.sensor,
            {"candidates": 0, "trainable": 0, "blocked_frozen": 0, "blocked_semantics": 0},
        )
        sensor_summary["candidates"] += 1
        if status == "trainable":
            sensor_summary["trainable"] += 1
        elif status in {"blocked_frozen_audit_source", "blocked_frozen_content_alias"}:
            sensor_summary["blocked_frozen"] += 1
        else:
            sensor_summary["blocked_semantics"] += 1

        records.append(
            {
                "name": spec.name,
                "sensor": spec.sensor,
                "path": str(path),
                "size_bytes": size_bytes,
                "frames": actual_frames,
                "bit_shift": spec.bit_shift,
                "cfa": spec.cfa,
                "target_semantics": spec.target_semantics,
                "approved_for_backprop": spec.approved_for_backprop,
                "status": status,
                "reason": spec.reason,
                "frozen_audit": frozen_entry,
                "frozen_inode_alias": frozen_inode_alias,
                "source_sha256": source_sha256,
                "supervision_evidence": supervision_evidence,
            }
        )

    missing_trainable = [
        sensor for sensor in required_sensors
        if summary.get(sensor, {}).get("trainable", 0) == 0
    ]
    return {
        "protocol": "A8 sensor adaptation training-source eligibility gate",
        "dataset_root": str(dataset_root),
        "dataset_access": "read_only",
        "frozen_validation": str(validation_path),
        "frozen_audit_source_count": len(frozen),
        "allowed_target_semantics": sorted(_ALLOWED_TARGET_SEMANTICS),
        "required_sensors": list(required_sensors),
        "status": "ready" if not missing_trainable else "blocked_no_compliant_training_source",
        "missing_trainable_sensors": missing_trainable,
        "summary_by_sensor": summary,
        "sources": records,
    }

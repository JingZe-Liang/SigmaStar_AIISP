"""Stream manifest: every measured interpretation fact about one stream.

One JSON per stream under artifacts/finetune_mis20s1/supervision/stream_manifests/.
Downstream stages (A1 reference build, A2 noise model, A3 dataloader) must
read their stream facts from these manifests, never re-derive them.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from raw_fusion.finetune.adapters import FilenameMetadata, SourceAdapter
from raw_fusion.finetune.exposure import ExposureFit
from raw_fusion.finetune.registration import RegistrationResult

CODE_VERSION = "finetune_a0_v1"


@dataclass(slots=True)
class MotionFacts:
    params: dict  # MotionMaskParams as dict
    static_fraction_mean: float
    static_fraction_per_frame: list[float]
    long_term_static_signal_fraction: float  # survey-comparable: static area among signal pixels
    threshold_per_frame_mean_dn: float
    mask_path: str


@dataclass(slots=True)
class StreamManifest:
    group: str
    scene: str
    gain_label: str
    role: str
    variant: str | None  # "denoised" (2DNR) / "fused" (3DNR) for group 03, else None
    path: str
    reader_kind: str
    width: int
    height: int
    frames_total: int
    frames_used: int
    bit_shift: int
    cfa: str
    sensor_black_dn: float
    filename_metadata: dict
    black_levels: dict[str, float]
    sigma_split_half: dict  # {"per_channel": {...}, "population": str}
    frame_mean_cv: float
    frame_mean_max_dev: float
    saturation_fraction: float
    motion: MotionFacts | None
    registration: dict | None  # RegistrationResult as dict
    exposure_fit: dict | None  # ExposureFit as dict
    created_at: str

    @classmethod
    def create(
        cls,
        adapter: SourceAdapter,
        gain_label: str,
        variant: str | None,
        path: Path,
        filename_metadata: FilenameMetadata,
    ) -> "StreamManifest":
        return cls(
            group=adapter.group,
            scene=adapter.scene,
            gain_label=gain_label,
            role=adapter.role,
            variant=variant,
            path=str(path),
            reader_kind=adapter.reader_kind,
            width=filename_metadata.width,
            height=filename_metadata.height,
            frames_total=0,
            frames_used=0,
            bit_shift=adapter.bit_shift,
            cfa=adapter.cfa,
            sensor_black_dn=adapter.sensor_black_dn,
            filename_metadata={k: v for k, v in asdict(filename_metadata).items() if v is not None},
            black_levels={},
            sigma_split_half={},
            frame_mean_cv=0.0,
            frame_mean_max_dev=0.0,
            saturation_fraction=0.0,
            motion=None,
            registration=None,
            exposure_fit=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def write_json(self, out_dir: Path) -> Path:
        """Atomically write the manifest JSON; returns the written path."""

        out_dir.mkdir(parents=True, exist_ok=True)
        name = manifest_name(self.group, self.scene, self.gain_label, self.variant)
        target = out_dir / f"{name}.json"
        payload = asdict(self)
        payload["code_version"] = CODE_VERSION
        handle, temporary = tempfile.mkstemp(prefix=".manifest.", dir=out_dir)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return target


def manifest_name(group: str, scene: str, gain_label: str, variant: str | None) -> str:
    parts = [group, scene, gain_label]
    if variant is not None:
        parts.append(variant)
    return "_".join(parts)


def set_registration(manifest: StreamManifest, result: RegistrationResult) -> None:
    manifest.registration = asdict(result)


def set_exposure_fit(manifest: StreamManifest, fit: ExposureFit) -> None:
    manifest.exposure_fit = asdict(fit)

"""Per-source adaptation table for the company RAW dataset.

Every interpretation decision below comes from the 2026-09-20 pixel-level survey
(docs/RVIDEFORMER_FINETUNE_PLAN_2026-09-20.md section 2). The dataset's own
metadata is unreliable (filenames state wrong frame counts, readme files state
wrong black levels, gain labels are swapped), so the reader layer never trusts
anything but this table plus per-stream measurements.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# 12-bit data stored MSB-aligned in uint16 (raw uint16 value = dn12 << 4).
BIT_SHIFT_HIGH_12 = 4
# 12-bit data stored in the low bits of uint16 (raw uint16 value = dn12).
BIT_SHIFT_LOW_12 = 0

# White clip in 12-bit DN for both alignments.
WHITE_CLIP_DN = 4095


@dataclass(frozen=True, slots=True)
class SourceAdapter:
    """Static interpretation rules for one dataset source."""

    group: str
    scene: str
    role: str  # "reference" | "noisy" | "auxiliary" | "calibration_black" | "calibration_flat"
    bit_shift: int
    cfa: str  # sensor CFA phase: "RGGB" or "GRBG"
    sensor_black_dn: float  # nominal sensor black level in 12-bit DN; per-stream measurement refines it
    reader_kind: str  # "single_file" | "frame_directory"


@dataclass(frozen=True, slots=True)
class FilenameMetadata:
    """Fields parsed from a dataset raw filename. Values the filename does not
    carry stay None; nothing here is trusted as ground truth."""

    width: int
    height: int
    cfa_name: str | None = None
    shutter_us: int | None = None
    sensor_gain: int | None = None
    isp_gain: int | None = None
    wb_r: int | None = None
    wb_g: int | None = None
    wb_b: int | None = None
    frames_stated: int | None = None
    analog_gain: int | None = None
    digital_gain: int | None = None
    bias: int | None = None


# Scene streams of group 04 are MIS20S1 RGGB, high-aligned.
_SHDRKOOM = SourceAdapter("04", "shdarkroom", "noisy", BIT_SHIFT_HIGH_12, "RGGB", 256.0, "single_file")
_OUTDOOR = SourceAdapter("04", "mis20s1_outdoor", "noisy", BIT_SHIFT_HIGH_12, "RGGB", 256.0, "single_file")
_CAL_BLACK = SourceAdapter("04", "mis20s1_calibrationdata", "calibration_black", BIT_SHIFT_HIGH_12, "GRBG", 256.7, "single_file")
_CAL_FLAT = SourceAdapter("04", "mis20s1_calibrationdata", "calibration_flat", BIT_SHIFT_HIGH_12, "GRBG", 256.7, "single_file")
# Group 03 holds 2DNR/3DNR ISP outputs of the MIS20S1 128x/645x streams: low-aligned.
_AUX_03 = SourceAdapter("03", "MIS20S1_2D3D", "auxiliary", BIT_SHIFT_LOW_12, "RGGB", 300.0, "frame_directory")

_REFERENCE_GAIN = "10x"
# Whole-stream phase-correlation baselines chosen during the survey.
REGISTRATION_BASELINE: dict[str, str] = {"shdarkroom": "645x", "mis20s1_outdoor": "1000x"}
# Supervision reference each noisy shdarkroom stream is exposure-normalized against.
EXPOSURE_REFERENCE = ("shdarkroom", "10x")


def resolve_adapter(group: str, scene: str, gain_label: str) -> SourceAdapter:
    """Return the adaptation rules for one discovered stream directory."""

    if group == "04" and scene == "shdarkroom":
        role = "reference" if gain_label == _REFERENCE_GAIN else "noisy"
        return SourceAdapter(
            group="04", scene=scene, role=role, bit_shift=_SHDRKOOM.bit_shift,
            cfa=_SHDRKOOM.cfa, sensor_black_dn=_SHDRKOOM.sensor_black_dn,
            reader_kind=_SHDRKOOM.reader_kind,
        )
    if group == "04" and scene == "mis20s1_outdoor":
        return _OUTDOOR
    if group == "04" and scene == "mis20s1_calibrationdata":
        return _CAL_BLACK
    if group == "03":
        return _AUX_03
    raise ValueError(f"no adaptation entry for group={group!r} scene={scene!r}; "
                     "groups 01/02/05 are Phase C scope and intentionally absent")


def calibration_adapter(kind: str) -> SourceAdapter:
    """Return the adapter for a calibration subdirectory ("black" or "color")."""

    if kind == "black":
        return _CAL_BLACK
    if kind == "color":
        return _CAL_FLAT
    raise ValueError(f"unknown calibration kind {kind!r}")


def resolve_exposure_reference(group: str, scene: str, gain_label: str) -> bool:
    """Whether this stream gets an exposure-normalization fit against the
    supervision reference. Only shdarkroom noisy streams do."""

    return group == "04" and scene == EXPOSURE_REFERENCE[0] and gain_label != EXPOSURE_REFERENCE[1]


def parse_filename_metadata(path: str | Path) -> FilenameMetadata:
    """Parse capture metadata out of a dataset raw filename.

    Scene format:
      raw_stream_{W}x{H}_16bit@{CFA}_[Shutter=..,SenserG=..,IspG=..,R=..,G=..,B=..].raw
    Calibration format:
      raw_stream_normal_{W}x{H}_16_{CFA}_{serial}_[FN=..,US=..,AG=..,DG=..,BV=..,R=..,G=..,B=..].raw
    """

    name = Path(path).name
    size_match = re.search(r"(\d+)x(\d+)", name)
    if size_match is None:
        raise ValueError(f"cannot parse WxH from filename {name!r}")
    width, height = int(size_match.group(1)), int(size_match.group(2))

    cfa_match = re.search(r"(?:16bit@|_16_)(RG|GR|BG|GB)", name)
    fields: dict[str, str] = {}
    bracket = re.search(r"\[([^]]*)\]", name)
    if bracket is not None:
        for item in bracket.group(1).split(","):
            key, sep, value = item.partition("=")
            if sep:
                fields[key.strip()] = value.strip()

    def as_int(key: str) -> int | None:
        value = fields.get(key)
        return int(value) if value is not None else None

    return FilenameMetadata(
        width=width,
        height=height,
        cfa_name=cfa_match.group(1) if cfa_match else None,
        shutter_us=as_int("Shutter") if "Shutter" in fields else as_int("US"),
        sensor_gain=as_int("SenserG"),
        isp_gain=as_int("IspG"),
        wb_r=as_int("R"),
        wb_g=as_int("G"),
        wb_b=as_int("B"),
        frames_stated=as_int("FN"),
        analog_gain=as_int("AG"),
        digital_gain=as_int("DG"),
        bias=as_int("BV"),
    )

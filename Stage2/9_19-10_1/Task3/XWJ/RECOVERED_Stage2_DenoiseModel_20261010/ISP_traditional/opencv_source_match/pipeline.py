from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image

from . import camera_look, dng_color, false_color, highlight_recovery


EXPOSURES = ("under", "normal", "over")
SOURCE_MATCH_EXTRA_EV = {"under": 0.0, "normal": 1.0, "over": 2.0}


@dataclass
class RenderResult:
    image: np.ndarray
    report: dict[str, Any]


def _value(item, default=None):
    if isinstance(item, dict) and "val" in item:
        return item["val"]
    return default if item is None else item


def _numbers(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        return [float(x) for x in value.replace(",", " ").split()]
    if value is None:
        return []
    return [float(value)]


def metadata_path(scene_dir: Path, exposure: str) -> Path:
    scene = scene_dir.name
    return scene_dir / "dng_components" / exposure / f"{scene}_metadata_numeric.json"


def raw_tiff_path(scene_dir: Path, exposure: str) -> Path:
    scene = scene_dir.name
    return (
        scene_dir
        / "dng_components"
        / exposure
        / "raw"
        / "raw_unprocessed"
        / f"{scene}_raw_unprocessed_raw_full_frame.tiff"
    )


def load_metadata_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    data = payload[0] if isinstance(payload, list) else payload
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metadata JSON structure: {path}")

    get = lambda key, default=None: _value(data.get(key), default)
    return {
        "black": float(get("EXIF:SubIFD:Image:BlackLevel", 0)),
        "white": float(get("EXIF:SubIFD:Image:WhiteLevel", 4095)),
        "cfa": [
            int(x)
            for x in _numbers(get("EXIF:SubIFD:Image:CFAPattern2", "2 1 1 0"))
        ],
        "neutral": _numbers(get("EXIF:IFD0:Image:AsShotNeutral", "1 1 1")),
        "color_matrix_1": _numbers(get("EXIF:IFD0:Image:ColorMatrix1", "")),
        "color_matrix_2": _numbers(get("EXIF:IFD0:Image:ColorMatrix2", "")),
        "camera_calibration_1": _numbers(
            get("EXIF:IFD0:Image:CameraCalibration1", "")
        ),
        "camera_calibration_2": _numbers(
            get("EXIF:IFD0:Image:CameraCalibration2", "")
        ),
        "analog_balance": _numbers(get("EXIF:IFD0:Image:AnalogBalance", "1 1 1")),
        "calibration_illuminant_1": int(
            float(get("EXIF:IFD0:Image:CalibrationIlluminant1", 17))
        ),
        "calibration_illuminant_2": int(
            float(get("EXIF:IFD0:Image:CalibrationIlluminant2", 21))
        ),
        "baseline_exposure": float(
            get("EXIF:IFD0:Image:BaselineExposure", 0.0)
        ),
        "camera_model": get("EXIF:IFD0:Image:UniqueCameraModel", ""),
        "default_crop_origin": [
            int(x)
            for x in _numbers(get("EXIF:SubIFD:Image:DefaultCropOrigin", "0 0"))
        ],
        "default_crop_size": [
            int(x)
            for x in _numbers(get("EXIF:SubIFD:Image:DefaultCropSize", ""))
        ],
        "active_area": [
            int(x) for x in _numbers(get("EXIF:SubIFD:Image:ActiveArea", ""))
        ],
        "iso": get("EXIF:ExifIFD:Camera:ISO", None),
        "exposure_time": get("EXIF:ExifIFD:Camera:ExposureTime", None),
        "exposure_compensation": get(
            "EXIF:ExifIFD:Camera:ExposureCompensation", None
        ),
    }


def load_metadata(scene_dir: Path, exposure: str) -> dict[str, Any]:
    return load_metadata_file(metadata_path(scene_dir, exposure))


def normalize_raw(raw: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    black = metadata["black"]
    white = metadata["white"]
    denominator = max(white - black, 1.0)
    normalized = (raw.astype(np.float32) - black) / denominator
    return np.clip(normalized, 0.0, 1.0)


def crop_default(image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    size = metadata.get("default_crop_size") or []
    origin = metadata.get("default_crop_origin") or [0, 0]
    if len(size) >= 2 and size[0] > 0 and size[1] > 0:
        x0, y0 = int(origin[0]), int(origin[1])
        width, height = int(size[0]), int(size[1])
        return image[y0 : y0 + height, x0 : x0 + width]

    active = metadata.get("active_area") or []
    if len(active) >= 4:
        y0, x0, y1, x1 = [int(value) for value in active[:4]]
        return image[y0:y1, x0:x1]
    return image


def cfa_name(cfa: list[int]) -> str:
    names = {0: "R", 1: "G", 2: "B"}
    return "".join(names.get(value, "?") for value in cfa)


def demosaic_edge_aware(raw01: np.ndarray, cfa: list[int]) -> np.ndarray:
    raw16 = np.clip(raw01 * 65535.0, 0, 65535).astype(np.uint16)
    pattern = cfa_name(cfa)
    codes = {
        "BGGR": cv2.COLOR_BayerBGGR2RGB_EA,
        "RGGB": cv2.COLOR_BayerRGGB2RGB_EA,
        "GRBG": cv2.COLOR_BayerGRBG2RGB_EA,
        "GBRG": cv2.COLOR_BayerGBRG2RGB_EA,
    }
    if pattern not in codes:
        raise ValueError(f"Unsupported CFA pattern: {pattern}")
    return cv2.demosaicing(raw16, codes[pattern]).astype(np.float32) / 65535.0


def gamma_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    image = np.clip(linear_rgb, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def resize_long_edge(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    if max_long_edge <= 0 or max(image.shape[:2]) <= max_long_edge:
        return image
    scale = max_long_edge / max(image.shape[:2])
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def process_frame(
    scene_dir: str | Path,
    exposure: str,
    max_long_edge: int = 0,
) -> RenderResult:
    if exposure not in EXPOSURES:
        raise ValueError(f"Exposure must be one of {EXPOSURES}, got: {exposure}")

    scene_dir = Path(scene_dir)
    raw_path = raw_tiff_path(scene_dir, exposure)
    if not raw_path.is_file():
        raise FileNotFoundError(f"RAW TIFF not found: {raw_path}")

    metadata = load_metadata(scene_dir, exposure)
    raw = tifffile.imread(raw_path)
    raw01 = crop_default(normalize_raw(raw, metadata), metadata)

    camera_rgb = demosaic_edge_aware(raw01, metadata["cfa"])
    camera_rgb = false_color.suppress_color_difference_artifacts(camera_rgb)
    camera_rgb = highlight_recovery.reconstruct_camera_highlights(
        camera_rgb, metadata["neutral"]
    )
    linear_srgb, white_xy, matrix_weight = dng_color.camera_to_linear_srgb(
        camera_rgb, metadata
    )
    linear_srgb = camera_look.apply_camera_tint(
        linear_srgb, metadata.get("camera_model", "")
    )

    extra_ev = SOURCE_MATCH_EXTRA_EV[exposure]
    baseline_ev = float(metadata.get("baseline_exposure", 0.0))
    total_ev = baseline_ev + extra_ev
    output = to_uint8(gamma_srgb(dng_color.apply_exposure(linear_srgb, total_ev)))
    output = resize_long_edge(output, max_long_edge)

    report = {
        "tool": "opencv-source-match",
        "tool_version": "1.0.0",
        "scene": scene_dir.name,
        "exposure": exposure,
        "raw_tiff": str(raw_path),
        "metadata_json": str(metadata_path(scene_dir, exposure)),
        "raw_shape": list(raw.shape),
        "crop_shape": list(raw01.shape),
        "output_shape": list(output.shape),
        "raw_dtype": str(raw.dtype),
        "black_level": metadata["black"],
        "white_level": metadata["white"],
        "cfa": cfa_name(metadata["cfa"]),
        "camera_model": metadata.get("camera_model", ""),
        "baseline_exposure_ev": baseline_ev,
        "source_match_extra_ev": extra_ev,
        "total_exposure_ev": total_ev,
        "resolved_white_xy": [float(x) for x in white_xy],
        "dual_illuminant_weight": float(matrix_weight),
        "camera_tint_applied": metadata.get("camera_model", "")
        in camera_look.CAMERA_TINT_GAINS,
        "iso": metadata.get("iso"),
        "exposure_time": metadata.get("exposure_time"),
        "exposure_compensation": metadata.get("exposure_compensation"),
    }
    return RenderResult(image=output, report=report)


def save_rgb(image: np.ndarray, output_path: str | Path, jpeg_quality: int = 95) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError("Output extension must be PNG, JPEG, or TIFF")
    kwargs = {"quality": jpeg_quality, "subsampling": 0} if suffix in {".jpg", ".jpeg"} else {}
    native_path: str | Path = output_path
    # Pillow uses legacy Win32 paths. The experiment output hierarchy can be
    # close to MAX_PATH even though the underlying filesystem accepts it.
    if os.name == "nt" and len(str(output_path)) >= 240:
        native_path = "\\\\?\\" + str(output_path.resolve())
    Image.fromarray(image, mode="RGB").save(native_path, **kwargs)
    return output_path


def render_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    max_long_edge: int = 0,
    image_extension: str = ".png",
) -> list[dict[str, Any]]:
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    extension = image_extension if image_extension.startswith(".") else f".{image_extension}"
    reports = []
    for exposure in EXPOSURES:
        result = process_frame(scene_dir, exposure, max_long_edge=max_long_edge)
        output_path = output_dir / (
            f"{scene_dir.name}_{exposure}_opencv_source_match{extension}"
        )
        save_rgb(result.image, output_path)
        result.report["output_image"] = str(output_path)
        reports.append(result.report)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{scene_dir.name}_processing_report.json"
    report_path.write_text(
        json.dumps({"frames": reports}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return reports

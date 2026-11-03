"""Re-render the report's dynamic comparisons from cached RAW predictions."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from raw_fusion.finetune.all_dataset_preview import (  # noqa: E402
    denormalize_packed,
    render_traditional_isp,
)
from raw_fusion.finetune.cross_sensor import NativeSequenceSpec, open_sequence  # noqa: E402
from raw_fusion.finetune.stream_io import pack_frame  # noqa: E402
from opencv_source_match.pipeline import demosaic_edge_aware  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison_cell(image: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Preserve the original overview, detail crop, and hidden header footprint."""
    width = 640
    overview_height = round(image.shape[0] * width / image.shape[1])
    overview = cv2.resize(image, (width, overview_height), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = crop
    detail = image[y0:y1, x0:x1]
    detail_height = 960 - overview_height
    scale = min(width / detail.shape[1], detail_height / detail.shape[0])
    size = (round(detail.shape[1] * scale), round(detail.shape[0] * scale))
    resized = cv2.resize(detail, size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((detail_height, width, 3), dtype=np.uint8)
    x, y = (width - size[0]) // 2, (detail_height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    separator = np.full((4, width, 3), 24, dtype=np.uint8)
    return np.concatenate((overview, separator, canvas), axis=0)[46:]


def render_figure(
    name: str,
    sensor: str,
    crop_name: str,
    predictions: list[tuple[str, Path]],
    output_name: str,
) -> dict[str, object]:
    config_path = PROJECT_DIR / "configs/a3_cross_sensor_extension_20261012.json"
    config = json.loads(config_path.read_text())
    spec = NativeSequenceSpec.from_mapping(name, config["sequences"][name])
    manifest_path = PROJECT_DIR / (
        "artifacts/a3_all_dataset_traditional_isp_preview_20261017/sequences/"
        f"01/raw_scene/{name}/traditional_isp_video/video_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    isp = manifest["isp"]
    sequence = manifest["sequence"]
    for key in ("path", "width", "height", "bit_shift", "cfa"):
        expected = str(spec.path) if key == "path" else getattr(spec, key)
        if sequence[key] != expected:
            raise ValueError(f"{name}: inconsistent {key} in the frozen ISP manifest")
    black = np.asarray(isp["black_levels_R_Gr_Gb_B"], dtype=np.float32)
    wb = tuple(float(value) for value in isp["wb_rgb"])
    if not np.allclose(wb, np.asarray(spec.fixed_isp_wb)[[0, 1, 3]]):
        raise ValueError(f"{name}: white balance differs from the evaluation protocol")
    exposure_ev = float(isp["fixed_exposure_ev"])
    inference_manifest_path = PROJECT_DIR / (
        "artifacts/a3_cross_sensor_extension_20261012/sequences/"
        f"{sensor}/{name}/inference/manifest.json"
    )
    inference_manifest = json.loads(inference_manifest_path.read_text())
    indices = [index for clip in inference_manifest["directions"]["stress"]["clips"] for index in clip]
    if indices != manifest["source_frame_indices"] or indices[0] != 0:
        raise ValueError(f"{name}: cached predictions do not match the original frame")
    crop = dict(spec.panel_crops)[crop_name]
    reader = open_sequence(spec)
    raw = reader.read_frame(0)
    frames = [("input", pack_frame(raw, spec.adapter).astype(np.float32))]
    prediction_records = []
    for method, path in predictions:
        prediction = np.load(path, mmap_mode="r")
        expected_shape = (len(indices), 4, *spec.packed_shape)
        if prediction.shape != expected_shape:
            raise ValueError(f"{path}: expected {expected_shape}, got {prediction.shape}")
        frame = np.asarray(prediction[0], dtype=np.float32)
        if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
            raise ValueError(f"{path}: invalid normalized RAW values")
        frames.append((method, denormalize_packed(frame, black)))
        prediction_records.append({
            "method": method,
            "path": str(path.relative_to(PROJECT_DIR)),
            "sha256": sha256(path),
            "frame_index": 0,
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
        })
    cells = []
    for _, frame_dn in frames:
        rgb = render_traditional_isp(
            frame_dn, black, wb, exposure_ev, spec.cfa,
            highlight_recovery=True, tone_map="reinhard",
        )
        cells.append(comparison_cell(rgb, crop))
    panel = np.concatenate(cells, axis=1)
    output = ASSET_DIR / output_name
    if not cv2.imwrite(str(output), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {output}")
    return {
        "sequence": name,
        "source_raw": str(spec.path),
        "source_frame": 0,
        "source_frame_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "source_metadata": config["sequences"][name],
        "prediction_sources": prediction_records,
        "prediction_provenance": str(inference_manifest_path.relative_to(PROJECT_DIR)),
        "frozen_isp_manifest": str(manifest_path.relative_to(PROJECT_DIR)),
        "frozen_isp_manifest_sha256": sha256(manifest_path),
        "isp": {
            "module_root": "ISP_traditional",
            "demosaic": "COLOR_BayerRGGB2RGB_EA",
            "black_levels_R_Gr_Gb_B": black.tolist(),
            "white_level_dn": 4095,
            "wb_rgb": list(wb),
            "fixed_exposure_ev": exposure_ev,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "gamma": "sRGB",
            "camera_color_mode": "uncalibrated_camera_rgb",
            "all_columns_share_parameters": True,
            "false_color_suppression": False,
            "2DNR": False,
            "3DNR": False,
        },
        "layout": {
            "columns": [method for method, _ in frames],
            "detail_crop_xyxy": list(crop),
            "original_header_rows_removed": 46,
            "column_width": 640,
            "shape": list(panel.shape),
        },
        "output": str(output.relative_to(PROJECT_DIR)),
        "output_sha256": sha256(output),
    }


def main() -> None:
    cv2.setNumThreads(4)
    mosaic = np.full((8, 8), 20000, dtype=np.uint16)
    mosaic[0::2, 0::2], mosaic[1::2, 1::2] = 30000, 10000
    legacy_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2RGB)[3, 3]
    corrected_rgb = np.rint(
        demosaic_edge_aware(mosaic.astype(np.float32) / 65535, [0, 1, 1, 2])[3, 3] * 65535
    ).astype(int)
    if not np.allclose(corrected_rgb, [30000, 20000, 10000], atol=1):
        raise ValueError("traditional ISP failed the RGGB channel-order check")
    base = PROJECT_DIR / "artifacts/a3_cross_sensor_extension_20261012/sequences"
    candidate = PROJECT_DIR / (
        "artifacts/a8_sensor_adaptation/sc635hai/evaluation_step500_20261013/"
        "dynamic_holdout/inference/prediction_stress.npy"
    )
    figures = [
        render_figure(
            "sc635hai_16x_motion_color", "sc635hai", "center_detail",
            [
                ("method3_ema", base / "sc635hai/sc635hai_16x_motion_color/inference/prediction_stress.npy"),
                ("method8_sc635hai_head_ema", candidate),
            ],
            "fig16_sc635hai_traditional_isp.png",
        ),
        render_figure(
            "sc450ai_1435x_motion_black_object", "sc450ai", "black_subject",
            [("method3_ema", base / "sc450ai/sc450ai_1435x_motion_black_object/inference/prediction_stress.npy")],
            "fig18_sc450ai_traditional_isp.png",
        ),
    ]
    code_paths = [
        Path(__file__),
        PROJECT_DIR / "src/raw_fusion/finetune/all_dataset_preview.py",
        PROJECT_DIR / "ISP_traditional/opencv_source_match/pipeline.py",
        PROJECT_DIR / "ISP_traditional/opencv_source_match/highlight_recovery.py",
    ]
    report = {
        "protocol": "Re-render report figures 16 and 18 from the original cached RAW frame 0",
        "channel_order_check": {
            "expected_rgb": [30000, 20000, 10000],
            "legacy_COLOR_BayerRG2RGB": legacy_rgb.tolist(),
            "traditional_COLOR_BayerRGGB2RGB_EA": corrected_rgb.tolist(),
        },
        "code_sha256": {str(path.relative_to(PROJECT_DIR)): sha256(path) for path in code_paths},
        "figures": figures,
    }
    (ASSET_DIR / "dynamic_isp_render_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for figure in figures:
        print(f"Rendered {figure['output']}")


if __name__ == "__main__":
    main()

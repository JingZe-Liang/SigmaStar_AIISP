"""Render the original native 1000x evaluation frame with the traditional ISP."""

import json
from pathlib import Path

import cv2
import numpy as np

from rerender_dynamic_isp import (
    ASSET_DIR,
    PROJECT_DIR,
    denormalize_packed,
    render_traditional_isp,
    sha256,
)
from raw_fusion.finetune.adapters import SourceAdapter
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, RawStreamReader, pack_frame


def main() -> None:
    cv2.setNumThreads(4)
    supervision = PROJECT_DIR / "artifacts/finetune_mis20s1/supervision"
    evaluation = PROJECT_DIR / "artifacts/finetune_mis20s1/evaluation"
    source_manifest_path = supervision / "stream_manifests/04_shdarkroom_1000x.json"
    prediction_path = evaluation / "native_1000x/native_prediction_1000x.npy"
    reference_path = evaluation / "1000x_self_reference_probe/temporal_mean_native_dn.npy"
    isp_manifest_path = PROJECT_DIR / (
        "artifacts/a3_all_dataset_traditional_isp_preview_20261017/sequences/"
        "04/raw_scene/04_shdarkroom_1000x/traditional_isp_video/video_manifest.json"
    )
    source = json.loads(source_manifest_path.read_text())
    manifest = json.loads(isp_manifest_path.read_text())
    isp = manifest["isp"]
    for key in ("path", "width", "height", "bit_shift", "cfa"):
        if manifest["sequence"][key] != source[key]:
            raise ValueError(f"native evaluation and ISP manifest disagree on {key}")
    black = np.asarray([source["black_levels"][name] for name in CHANNEL_NAMES], dtype=np.float32)
    if not np.array_equal(black, isp["black_levels_R_Gr_Gb_B"]):
        raise ValueError("native evaluation and ISP black levels differ")
    metadata = source["filename_metadata"]
    wb = tuple(float(value) for value in isp["wb_rgb"])
    if not np.allclose(wb, [metadata["wb_r"] / metadata["wb_g"], 1, metadata["wb_b"] / metadata["wb_g"]]):
        raise ValueError("ISP white balance differs from source metadata")
    adapter = SourceAdapter(
        group=source["group"], scene=source["scene"], role=source["role"],
        bit_shift=source["bit_shift"], cfa=source["cfa"],
        sensor_black_dn=source["sensor_black_dn"], reader_kind=source["reader_kind"],
    )
    reader = RawStreamReader(Path(source["path"]), source["width"], source["height"])
    input_dn = pack_frame(reader.read_frame(0), adapter).astype(np.float32)
    prediction = np.load(prediction_path, mmap_mode="r")
    reference_dn = np.load(reference_path, mmap_mode="r")
    if prediction.shape != (source["frames_used"], *input_dn.shape) or reference_dn.shape != input_dn.shape:
        raise ValueError("cached native evaluation arrays have unexpected shapes")
    prediction_frame = np.asarray(prediction[0], dtype=np.float32)
    if not np.isfinite(prediction_frame).all() or prediction_frame.min() < 0 or prediction_frame.max() > 1:
        raise ValueError("cached prediction is not normalized RAW")
    if not np.isfinite(reference_dn).all():
        raise ValueError("native reference contains nonfinite values")
    prediction_dn = denormalize_packed(prediction_frame, black)
    exposure_ev = float(isp["fixed_exposure_ev"])
    cells = []
    for frame_dn in (input_dn, prediction_dn, reference_dn):
        rgb = render_traditional_isp(
            frame_dn, black, wb, exposure_ev, source["cfa"],
            highlight_recovery=True, tone_map="reinhard",
        )
        cells.append(cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA))
    gutter = np.full((360, 8, 3), 255, dtype=np.uint8)
    panel = np.concatenate((cells[0], gutter, cells[1], gutter, cells[2]), axis=1)
    output = ASSET_DIR / "fig11_native_1000x_traditional_isp.png"
    if not cv2.imwrite(str(output), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {output}")
    code_paths = [
        Path(__file__),
        ASSET_DIR / "rerender_dynamic_isp.py",
        PROJECT_DIR / "src/raw_fusion/finetune/all_dataset_preview.py",
        PROJECT_DIR / "ISP_traditional/opencv_source_match/pipeline.py",
        PROJECT_DIR / "ISP_traditional/opencv_source_match/highlight_recovery.py",
    ]
    record = {
        "protocol": "Original native 1000x evaluation, frame 0, identical ISP for all three columns",
        "source_raw": source["path"],
        "source_frame": 0,
        "source_manifest": str(source_manifest_path.relative_to(PROJECT_DIR)),
        "source_manifest_sha256": sha256(source_manifest_path),
        "native_prediction": str(prediction_path.relative_to(PROJECT_DIR)),
        "native_prediction_sha256": sha256(prediction_path),
        "native_prediction_index": 0,
        "native_prediction_shape": list(prediction.shape),
        "native_reference_dn": str(reference_path.relative_to(PROJECT_DIR)),
        "native_reference_sha256": sha256(reference_path),
        "cross_gain_exposure_mapping": False,
        "spatial_registration": False,
        "reference_denormalized_again": False,
        "frozen_isp_manifest": str(isp_manifest_path.relative_to(PROJECT_DIR)),
        "frozen_isp_manifest_sha256": sha256(isp_manifest_path),
        "isp": {
            "module_root": "ISP_traditional",
            "cfa": source["cfa"],
            "bit_shift": source["bit_shift"],
            "black_levels_R_Gr_Gb_B": black.tolist(),
            "white_level_dn": 4095,
            "wb_rgb": list(wb),
            "fixed_exposure_ev": exposure_ev,
            "highlight_recovery": True,
            "tone_map": "reinhard",
            "gamma": "sRGB",
            "camera_color_mode": "uncalibrated_camera_rgb",
            "all_columns_share_parameters": True,
        },
        "layout": {
            "columns": ["input", "method3_ema", "1000x_native_reference"],
            "source_crop_xyxy": [0, 0, source["width"], source["height"]],
            "shape": list(panel.shape),
        },
        "code_sha256": {str(path.relative_to(PROJECT_DIR)): sha256(path) for path in code_paths},
        "output": str(output.relative_to(PROJECT_DIR)),
        "output_sha256": sha256(output),
    }
    (ASSET_DIR / "native_isp_render_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered {output.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()

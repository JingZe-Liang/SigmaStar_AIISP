import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRADITIONAL_ISP_ROOT = REPO_ROOT / "opencv_fixed_raw_compare_isp"
for import_root in (REPO_ROOT, TRADITIONAL_ISP_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from opencv_source_match import false_color
from opencv_source_match.pipeline import demosaic_edge_aware, gamma_srgb
from sigmastar_inference.infer_sigmastar_raw import parse_metadata


def _open_raw(path: Path, width: int, height: int) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(f"RAW stream does not exist: {path}")
    frame_bytes = width * height * np.dtype(np.uint16).itemsize
    if path.stat().st_size == 0 or path.stat().st_size % frame_bytes:
        raise ValueError(f"{path} is not an integer number of {width}x{height} uint16 frames")
    return np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        shape=(path.stat().st_size // frame_bytes, height, width),
    )


def _white_balance(metadata_source: Path) -> np.ndarray:
    metadata = parse_metadata(metadata_source)
    if metadata.red_gain is None or metadata.green_gain is None or metadata.blue_gain is None:
        raise ValueError(
            "--metadata-source must include R/G/B gains in its name, such as the source RAW filename"
        )
    if metadata.green_gain <= 0:
        raise ValueError("The metadata G gain must be positive")
    return np.asarray(
        [
            metadata.red_gain / metadata.green_gain,
            1.0,
            metadata.blue_gain / metadata.green_gain,
        ],
        dtype=np.float32,
    )


def _linear_rgb(raw: np.ndarray, black_level: float, white_level: float,
                white_balance: np.ndarray) -> np.ndarray:
    normalized = np.maximum(raw.astype(np.float32) - black_level, 0.0)
    normalized /= white_level - black_level
    normalized = np.clip(normalized, 0.0, 1.0)
    rgb = demosaic_edge_aware(normalized, [0, 1, 1, 2])
    rgb = false_color.suppress_color_difference_artifacts(rgb)
    return rgb * white_balance.reshape(1, 1, 3)


def _render_frame(raw: np.ndarray, black_level: float, white_level: float,
                  white_balance: np.ndarray, exposure: float) -> np.ndarray:
    linear_rgb = _linear_rgb(raw, black_level, white_level, white_balance)
    return np.clip(np.rint(gamma_srgb(np.clip(linear_rgb * exposure, 0.0, None)) * 255.0), 0, 255).astype(np.uint8)


def render_comparison_video(two_dnr: Path, three_dnr: Path, ai_output: Path,
                            metadata_source: Path, output: Path, width: int = 1920,
                            height: int = 1080, black_level: float = 300.0,
                            white_level: float = 4095.0, fps: float = 30.0,
                            scale: float = 0.5, frame_step: int = 1,
                            overwrite: bool = False,
                            layout: str = "comparison") -> dict[str, object]:
    """Render fixed-tone RGGB comparison panels or the local-align output only."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    if white_level <= black_level:
        raise ValueError("white_level must be greater than black_level")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if layout not in {"comparison", "ai"}:
        raise ValueError("layout must be 'comparison' or 'ai'")
    manifest_path = output.with_suffix(output.suffix + ".json")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        output.unlink()
    if manifest_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output manifest already exists: {manifest_path}. Pass --overwrite to replace it.")
        manifest_path.unlink()

    streams = [_open_raw(path, width, height) for path in (two_dnr, three_dnr, ai_output)]
    frame_counts = {stream.shape[0] for stream in streams}
    if len(frame_counts) != 1:
        raise ValueError(f"Comparison streams have different frame counts: {[stream.shape[0] for stream in streams]}")
    for path, stream in zip((two_dnr, three_dnr, ai_output), streams):
        if int(stream.max()) > int(white_level):
            raise ValueError(f"{path} has codes above configured white_level={white_level}")

    white_balance = _white_balance(metadata_source)
    reference_linear = _linear_rgb(streams[0][0], black_level, white_level, white_balance)
    reference_white = float(np.quantile(reference_linear, 0.995))
    exposure = float(np.clip(0.7 / max(reference_white, 1e-6), 0.25, 8.0))
    frame_width = int(round(width * scale))
    frame_height = int(round(height * scale))
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("scale produces an empty output frame")

    output.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width * 3 if layout == "comparison" else frame_width, frame_height),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    try:
        rendered_count = 0
        for frame_index in range(0, streams[0].shape[0], frame_step):
            if layout == "comparison":
                rendered = [
                    _render_frame(stream[frame_index], black_level, white_level, white_balance, exposure)
                    for stream in streams
                ]
                panels = [
                    cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                    for frame in rendered
                ]
                for panel, label in zip(panels, ("2DNR", "3DNR", "Local align output")):
                    cv2.rectangle(panel, (0, 0), (260, 44), (0, 0, 0), thickness=-1)
                    cv2.putText(panel, label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                output_frame = np.concatenate(panels, axis=1)
            else:
                output_frame = cv2.resize(
                    _render_frame(streams[2][frame_index], black_level, white_level, white_balance, exposure),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_AREA,
                )
            video.write(np.ascontiguousarray(output_frame[:, :, ::-1]))
            rendered_count += 1
            if rendered_count % 10 == 0 or frame_index + frame_step >= streams[0].shape[0]:
                print(
                    f"[local-align render] source_frame={frame_index + 1}/"
                    f"{streams[0].shape[0]} rendered={rendered_count}"
                )
    finally:
        video.release()

    manifest = {
        "tool": "local-align-rggb-render",
        "two_dnr": str(two_dnr),
        "three_dnr": str(three_dnr),
        "ai_output": str(ai_output),
        "output": str(output),
        "source_frame_count": int(streams[0].shape[0]),
        "rendered_frame_count": rendered_count,
        "frame_step": frame_step,
        "input_size": [width, height],
        "output_panel_size": [frame_width, frame_height],
        "panel_order": ["2DNR", "3DNR", "local-align AI"] if layout == "comparison" else ["local-align AI"],
        "layout": layout,
        "cfa": "RGGB",
        "black_level": black_level,
        "white_level": white_level,
        "white_balance": white_balance.tolist(),
        "fixed_exposure": exposure,
        "reference_white_quantile": reference_white,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a fixed RGGB local-align comparison or local-align output"
    )
    parser.add_argument("--two-dnr", type=Path, required=True)
    parser.add_argument("--three-dnr", type=Path, required=True)
    parser.add_argument("--ai-output", type=Path, required=True)
    parser.add_argument("--metadata-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--black-level", type=float, default=300.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--layout", choices=["comparison", "ai"], default="comparison")
    args = parser.parse_args()
    manifest = render_comparison_video(
        two_dnr=args.two_dnr,
        three_dnr=args.three_dnr,
        ai_output=args.ai_output,
        metadata_source=args.metadata_source,
        output=args.output,
        width=args.width,
        height=args.height,
        black_level=args.black_level,
        white_level=args.white_level,
        fps=args.fps,
        scale=args.scale,
        frame_step=args.frame_step,
        overwrite=args.overwrite,
        layout=args.layout,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import load_config, project_root, validate_scene
from .dataset_fast import open_scene_streams
from .isp import simple_isp
from .raw_io import RawSpec, RawStream


PANEL_WIDTH = 640
PANEL_HEIGHT = 360


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "company.yaml"


def _label(image: np.ndarray, text: str, detail: str | None = None) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (10, 10, 10), -1)
    cv2.putText(
        output,
        text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if detail:
        size = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        cv2.putText(
            output,
            detail,
            (output.shape[1] - size[0] - 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return output


def _panel(image: np.ndarray, label: str, detail: str | None = None) -> np.ndarray:
    resized = cv2.resize(
        image, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA
    )
    return _label(resized, label, detail)


def _crop_panel(
    image: np.ndarray,
    crop: list[int] | tuple[int, int, int, int],
    label: str,
) -> np.ndarray:
    x0, y0, x1, y1 = map(int, crop)
    height, width = image.shape[:2]
    x0, x1 = np.clip((x0, x1), 0, width)
    y0, y1 = np.clip((y0, y1), 0, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid motion crop {crop} for image {image.shape}")
    crop_image = image[y0:y1, x0:x1]
    resized = cv2.resize(
        crop_image, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    return _label(resized, label, "motion crop")


def comparison_frame(
    denoised: np.ndarray,
    fused: np.ndarray,
    learned: np.ndarray,
    crop: list[int],
    frame_index: int,
) -> np.ndarray:
    top = np.hstack(
        (
            _panel(denoised, "2DNR", "motion-safe anchor"),
            _panel(fused, "3DNR", "temporal candidate"),
            _panel(learned, "Learned safe fusion", f"frame {frame_index:04d}"),
        )
    )
    bottom = np.hstack(
        (
            _crop_panel(denoised, crop, "2DNR"),
            _crop_panel(fused, crop, "3DNR"),
            _crop_panel(learned, crop, "Learned safe fusion"),
        )
    )
    return np.vstack((top, bottom))


def render_video(
    config: dict,
    scene_name: str,
    learned_raw: Path,
    output_video: Path,
    contact_sheet: Path,
    fps: float,
    contact_frame: int,
    overwrite: bool,
) -> dict:
    scene = validate_scene(config, scene_name)
    streams = open_scene_streams(config, scene_name)
    raw = config["raw"]
    learned_stream = RawStream(
        learned_raw, RawSpec(width=raw["width"], height=raw["height"])
    )
    if learned_stream.frame_count != streams.frame_count:
        raise ValueError("Learned RAW frame count does not match candidates")
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"Video exists; pass --overwrite: {output_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_video.with_suffix(".part" + output_video.suffix)
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (PANEL_WIDTH * 3, PANEL_HEIGHT * 2),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")

    isp_args = {
        "black": raw["candidate_black"],
        "white": raw["white"],
        "wb": scene["wb"],
        "exposure": scene["display_exposure"],
    }
    wrote_contact = False
    try:
        for frame_index in tqdm(range(streams.frame_count), desc=f"video {scene_name}"):
            denoised = simple_isp(streams.denoised.frame(frame_index), **isp_args)
            fused = simple_isp(streams.fused.frame(frame_index), **isp_args)
            learned = simple_isp(learned_stream.frame(frame_index), **isp_args)
            canvas = comparison_frame(
                denoised,
                fused,
                learned,
                scene["motion_crop"],
                frame_index,
            )
            writer.write(canvas)
            if frame_index == contact_frame:
                if not cv2.imwrite(str(contact_sheet), canvas):
                    raise RuntimeError(f"Failed to write contact sheet {contact_sheet}")
                wrote_contact = True
    finally:
        writer.release()
    if not wrote_contact:
        raise ValueError(f"Contact frame {contact_frame} is outside the sequence")
    temporary.replace(output_video)
    return {
        "scene": scene_name,
        "frames": streams.frame_count,
        "fps": fps,
        "duration_seconds": streams.frame_count / fps,
        "resolution": [PANEL_WIDTH * 3, PANEL_HEIGHT * 2],
        "video": str(output_video.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "display_isp": isp_args,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--scene", choices=("645x", "128x"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--contact-frame", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root(config) / "outputs"
    learned = args.input or root / "raw" / f"{args.scene}_learned_fusion.raw"
    output = args.output or root / "videos" / f"{args.scene}_comparison.mp4"
    sheet = args.contact_sheet or root / "images" / f"{args.scene}_comparison_frame_0050.png"
    fps = args.fps or float(config["inference"]["fps"])
    result = render_video(
        config,
        args.scene,
        learned.resolve(),
        output.resolve(),
        sheet.resolve(),
        fps,
        args.contact_frame,
        args.overwrite,
    )
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


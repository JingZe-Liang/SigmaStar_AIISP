"""Fixed traditional ISP for a fair under/fused/over RAW comparison.

This module deliberately has no image-adaptive exposure, white balance, local
tone mapping, highlight reconstruction, or camera-specific tint correction.
All three images share one validated sensor/color metadata record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from . import dng_color, false_color, highlight_recovery
from .pipeline import (
    cfa_name,
    crop_default,
    demosaic_edge_aware,
    gamma_srgb,
    load_metadata_file,
    normalize_raw,
    resize_long_edge,
    save_rgb,
    to_uint8,
)


SHARED_METADATA_FIELDS = (
    "black",
    "white",
    "cfa",
    "neutral",
    "color_matrix_1",
    "color_matrix_2",
    "camera_calibration_1",
    "camera_calibration_2",
    "analog_balance",
    "calibration_illuminant_1",
    "calibration_illuminant_2",
    "default_crop_origin",
    "default_crop_size",
    "active_area",
    "camera_model",
)


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float, list, tuple)) and isinstance(
        right, (int, float, list, tuple)
    ):
        try:
            return bool(np.allclose(left, right, rtol=0.0, atol=1e-6))
        except (TypeError, ValueError):
            return False
    return left == right


def validate_shared_metadata(under: dict[str, Any], over: dict[str, Any]) -> None:
    mismatches = [
        key
        for key in SHARED_METADATA_FIELDS
        if not _metadata_equal(under.get(key), over.get(key))
    ]
    if mismatches:
        raise ValueError(
            "under/over sensor or color metadata differ for: {}. "
            "A fair fixed comparison requires one shared record.".format(
                ", ".join(mismatches)
            )
        )


def canonical_packed_to_mosaic(packed: np.ndarray, cfa: list[int]) -> tuple[np.ndarray, float]:
    """Convert U2Fusion canonical [R, Gr, B, Gb] packed RAW to a Bayer mosaic."""
    packed = np.asarray(packed, dtype=np.float32)
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError("Fused packed RAW must have shape [4,H,W] in [R,Gr,B,Gb] order")
    if packed.shape[1] == 0 or packed.shape[2] == 0:
        raise ValueError("Fused packed RAW has an empty spatial dimension")
    clipped_fraction = float(np.mean((packed < 0.0) | (packed > 1.0)))
    packed = np.clip(packed, 0.0, 1.0)
    pattern = cfa_name(cfa)
    channel_order = {
        "RGGB": (0, 1, 3, 2),
        "BGGR": (2, 3, 1, 0),
        "GRBG": (1, 0, 2, 3),
        "GBRG": (3, 2, 0, 1),
    }.get(pattern)
    if channel_order is None:
        raise ValueError("Unsupported CFA pattern for canonical packed RAW: {}".format(pattern))
    physical = packed[list(channel_order)]
    height, width = physical.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = physical[0]
    mosaic[0::2, 1::2] = physical[1]
    mosaic[1::2, 0::2] = physical[2]
    mosaic[1::2, 1::2] = physical[3]
    return mosaic, clipped_fraction


def render_fixed(
    raw01: np.ndarray,
    metadata: dict[str, Any],
    fixed_ev: float,
    enable_false_color_suppression: bool,
    enable_highlight_recovery: bool,
    tone_map: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one already-normalized mosaic through the shared deterministic chain."""
    camera_rgb = demosaic_edge_aware(raw01, metadata["cfa"])
    if enable_false_color_suppression:
        camera_rgb = false_color.suppress_color_difference_artifacts(camera_rgb)
    if enable_highlight_recovery:
        camera_rgb = highlight_recovery.reconstruct_camera_highlights(
            camera_rgb, metadata["neutral"]
        )
    linear_srgb, white_xy, matrix_weight = dng_color.camera_to_linear_srgb(
        camera_rgb, metadata
    )
    # The same constant is used for all frames.  No baseline or role EV is used.
    linear_srgb = dng_color.apply_exposure(linear_srgb, fixed_ev)
    if tone_map == "reinhard":
        # Fixed, per-pixel shoulder: it preserves ratios better than channel clipping.
        linear_srgb = linear_srgb / (1.0 + linear_srgb)
    elif tone_map != "hard-clip":
        raise ValueError("Unsupported tone map: {}".format(tone_map))
    image = to_uint8(gamma_srgb(linear_srgb))
    return image, {
        "normalized_mosaic_mean": float(np.mean(raw01)),
        "normalized_mosaic_median": float(np.median(raw01)),
        "resolved_white_xy": [float(value) for value in white_xy],
        "dual_illuminant_weight": float(matrix_weight),
    }


def _load_raw_tiff(path: Path, metadata: dict[str, Any]) -> np.ndarray:
    raw = tifffile.imread(path)
    if raw.ndim != 2:
        raise ValueError("Expected a one-channel RAW TIFF: {}".format(path))
    return crop_default(normalize_raw(raw, metadata), metadata)


def _make_preview(images: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    panels = []
    for name, image in images:
        header = Image.new("RGB", (image.shape[1], 28), "white")
        ImageDraw.Draw(header).text((8, 7), name, fill="black")
        panel = Image.new("RGB", (image.shape[1], image.shape[0] + 28), "white")
        panel.paste(header, (0, 0))
        panel.paste(Image.fromarray(image, mode="RGB"), (0, 28))
        panels.append(panel)
    canvas = Image.new(
        "RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white"
    )
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    canvas.save(output_path)


def render_comparison(
    under_raw: str | Path,
    over_raw: str | Path,
    fused_packed: str | Path,
    under_metadata: str | Path,
    over_metadata: str | Path,
    output_dir: str | Path,
    fixed_ev: float = 0.0,
    enable_false_color_suppression: bool = False,
    enable_highlight_recovery: bool = False,
    tone_map: str = "hard-clip",
    max_long_edge: int = 0,
    preview_long_edge: int = 1200,
    image_extension: str = ".png",
) -> dict[str, Any]:
    """Create fair under/fused/over RGB outputs from a common deterministic ISP."""
    under_raw = Path(under_raw)
    over_raw = Path(over_raw)
    fused_packed = Path(fused_packed)
    output_dir = Path(output_dir)
    for path in (under_raw, over_raw, fused_packed):
        if not path.is_file():
            raise FileNotFoundError(path)

    under_meta = load_metadata_file(under_metadata)
    over_meta = load_metadata_file(over_metadata)
    validate_shared_metadata(under_meta, over_meta)
    shared = under_meta

    under01 = _load_raw_tiff(under_raw, shared)
    over01 = _load_raw_tiff(over_raw, shared)
    fused01, fused_clip_fraction = canonical_packed_to_mosaic(
        np.load(fused_packed, allow_pickle=False), shared["cfa"]
    )
    fused01 = crop_default(fused01, shared)
    shapes = {under01.shape, fused01.shape, over01.shape}
    if len(shapes) != 1:
        raise ValueError(
            "under, fused and over mosaics do not have the same cropped shape: {}".format(
                sorted(shapes)
            )
        )

    extension = image_extension if image_extension.startswith(".") else "." + image_extension
    if extension.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError("image_extension must be PNG, JPEG, or TIFF")
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, np.ndarray]] = []
    frame_report: dict[str, Any] = {}
    for name, mosaic in (("under", under01), ("fused", fused01), ("over", over01)):
        image, details = render_fixed(
            mosaic,
            shared,
            fixed_ev,
            enable_false_color_suppression,
            enable_highlight_recovery,
            tone_map,
        )
        image = resize_long_edge(image, max_long_edge)
        output_path = output_dir / (name + extension)
        save_rgb(image, output_path)
        rendered.append((name, resize_long_edge(image, preview_long_edge)))
        frame_report[name] = {
            "input": str(
                {"under": under_raw, "fused": fused_packed, "over": over_raw}[name].resolve()
            ),
            "output": str(output_path.resolve()),
            "details": details,
        }

    preview_path = output_dir / "comparison_preview.png"
    _make_preview(rendered, preview_path)
    report = {
        "tool": "opencv-fixed-raw-compare-isp",
        "fairness_contract": {
            "shared_metadata_reference": str(Path(under_metadata).resolve()),
            "shared_metadata_validated_against": str(Path(over_metadata).resolve()),
            "fixed_ev_for_all_frames": float(fixed_ev),
            "uses_baseline_exposure": False,
            "uses_exposure_time_or_iso": False,
            "uses_role_dependent_ev": False,
            "uses_highlight_reconstruction": bool(enable_highlight_recovery),
            "uses_camera_specific_tint": False,
            "uses_image_adaptive_tone_mapping": False,
            "tone_map": tone_map,
            "false_color_suppression_enabled": bool(enable_false_color_suppression),
        },
        "shared_metadata": {
            "black_level": shared["black"],
            "white_level": shared["white"],
            "cfa": cfa_name(shared["cfa"]),
            "as_shot_neutral": shared["neutral"],
        },
        "fused_packed_contract": "[R, Gr, B, Gb] normalized packed RAW",
        "fused_input_out_of_range_fraction_clipped": fused_clip_fraction,
        "frames": frame_report,
        "preview": str(preview_path.resolve()),
        "preview_long_edge": int(preview_long_edge),
    }
    report_path = output_dir / "fairness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report

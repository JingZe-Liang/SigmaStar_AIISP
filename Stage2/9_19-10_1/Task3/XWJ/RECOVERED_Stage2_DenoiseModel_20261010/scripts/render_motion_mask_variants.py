#!/usr/bin/env python3
"""Render current and stricter motion-confidence mask variants for review."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from build_dynamic_n2n_probe import (  # noqa: E402
    _aligned_motion_mask,
    _flow_confidence,
    _parse_crop,
    _read_aligned_frame,
    _render_packed,
)
from diagnose_motion_mask_noise import _parse_pairs  # noqa: E402
from raw_fusion.finetune.a3 import _channel_vector, _load_json  # noqa: E402
from raw_fusion.finetune.adapters import resolve_adapter  # noqa: E402
from raw_fusion.finetune.stream_io import open_stream  # noqa: E402


def _mask_image(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    height, width = shape[:2]
    resized = cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = resized
    return image


def run(
    *,
    supervision_dir: Path,
    gain: str,
    pairs: list[tuple[int, int]],
    crop: tuple[int, int, int, int],
    output_dir: Path,
) -> None:
    report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
    reference_black = _channel_vector(report["black_levels"]).astype(np.float32)
    manifest = _load_json(supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json")
    adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
    reader, _ = open_stream(Path(manifest["path"]), adapter)
    frames = sorted({frame for pair in pairs for frame in pair})
    loaded = {
        frame: _read_aligned_frame(reader, adapter, manifest, reference_black, frame)
        for frame in frames
    }
    masks = {frame: _aligned_motion_mask(manifest, frame) for frame in frames}
    x, y, width, height = crop
    output_dir.mkdir(parents=True, exist_ok=True)
    for start, end in pairs:
        source = loaded[start]
        target = loaded[end]
        warped, flow, flow_error, photometric_error, flow_magnitude = _flow_confidence(source, target)
        h, w = flow_magnitude.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        target_motion = cv2.remap(
            masks[end], grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0,
        )
        dynamic = (masks[start] >= 0.5) | (target_motion >= 0.5)
        valid = (flow_error < 1.5) & (photometric_error < 0.08) & (flow_magnitude < 30.0)
        current = dynamic & valid
        strict_1 = current & (flow_magnitude >= 1.0)
        strict_2 = current & (flow_magnitude >= 2.0)

        source_rgb = _render_packed(source, reference_black)[2 * y:2 * (y + height), 2 * x:2 * (x + width)]
        target_rgb = _render_packed(warped, reference_black)[2 * y:2 * (y + height), 2 * x:2 * (x + width)]
        panels = [source_rgb, target_rgb]
        for mask in (current, strict_1, strict_2):
            panels.append(_mask_image(mask[y:y + height, x:x + width], source_rgb.shape))
        name = f"pair_{start:04d}_{end:04d}_mask_variants.png"
        cv2.imwrite(str(output_dir / name), np.hstack(panels)[:, :, ::-1])
        print(json.dumps({
            "pair": [start, end],
            "current_fraction": float(current.mean()),
            "flow_ge_1_fraction": float(strict_1.mean()),
            "flow_ge_2_fraction": float(strict_2.mean()),
            "path": str((output_dir / name).resolve()),
        }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-dir", type=Path, default=Path("artifacts/finetune_mis20s1/supervision"))
    parser.add_argument("--gain", default="645x")
    parser.add_argument("--pairs", default="30:34,46:50,140:144")
    parser.add_argument("--crop", default="650,100,240,420")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/finetune_mis20s1/evaluation/motion_mask_variants_645x"),
    )
    args = parser.parse_args()
    run(
        supervision_dir=args.supervision_dir,
        gain=args.gain,
        pairs=_parse_pairs(args.pairs),
        crop=_parse_crop(args.crop),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

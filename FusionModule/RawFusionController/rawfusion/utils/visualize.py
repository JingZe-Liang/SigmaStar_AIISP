from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw


def to_u16_code(x: np.ndarray, code_max: float = 4095.0) -> np.ndarray:
    return np.clip(np.rint(x * code_max), 0, code_max).astype(np.uint16)


def save_pgm_u16(path: str | Path, image_u16: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_u16.ndim != 2 or image_u16.dtype != np.uint16:
        raise ValueError(f"save_pgm_u16 expects [H,W] uint16, got {image_u16.shape}, {image_u16.dtype}")
    h, w = image_u16.shape
    header = f"P5\n{w} {h}\n65535\n".encode("ascii")
    data_be = image_u16.astype(">u2", copy=False).tobytes()
    with path.open("wb") as f:
        f.write(header)
        f.write(data_be)


def raw_to_uint8(x: np.ndarray, code_max: float = 4095.0, auto_stretch: bool = True) -> np.ndarray:
    x = np.asarray(x).astype(np.float32)
    if auto_stretch:
        lo, hi = np.percentile(x, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        y = (x - lo) / (hi - lo)
    else:
        y = x / float(code_max)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def save_png_gray(path: str | Path, image: np.ndarray, code_max: float = 4095.0, auto_stretch: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raw_to_uint8(image, code_max=code_max, auto_stretch=auto_stretch), mode="L").save(path)


def save_alpha_png(path: str | Path, alpha: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(a, mode="L").save(path)


def make_compare_grid(
    images: Dict[str, np.ndarray],
    out_path: str | Path,
    code_max: float = 4095.0,
    auto_stretch: bool = True,
    max_width_each: int = 360,
) -> None:
    """Create a simple labeled horizontal comparison grid."""
    panels = []
    labels = []
    for name, arr in images.items():
        labels.append(name)
        if name.lower().startswith("alpha") or name.lower().startswith("weight"):
            u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            u8 = raw_to_uint8(arr, code_max=code_max, auto_stretch=auto_stretch)
        im = Image.fromarray(u8, mode="L")
        if im.width > max_width_each:
            new_h = max(1, int(im.height * max_width_each / im.width))
            im = im.resize((max_width_each, new_h), Image.BILINEAR)
        panels.append(im.convert("RGB"))
    pad = 8
    label_h = 28
    w = sum(p.width for p in panels) + pad * (len(panels) + 1)
    h = max(p.height for p in panels) + label_h + pad * 2
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    x = pad
    for label, panel in zip(labels, panels):
        draw.text((x, pad), label, fill=(0, 0, 0))
        canvas.paste(panel, (x, pad + label_h))
        x += panel.width + pad
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

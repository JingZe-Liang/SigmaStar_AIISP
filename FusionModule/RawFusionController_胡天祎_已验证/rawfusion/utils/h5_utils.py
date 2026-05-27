from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

REQUIRED_KEYS = ("noisy", "2dnr", "3dnr", "clean")


def inspect_h5(path: str | Path, max_frames_for_stats: int = 3) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out: Dict[str, object] = {"path": str(path.resolve()), "keys": {}, "valid": True, "errors": []}
    with h5py.File(path, "r") as f:
        if not all(k in f for k in REQUIRED_KEYS):
            out["valid"] = False
            out["errors"].append(f"Missing keys. Expected {REQUIRED_KEYS}, got {list(f.keys())}")
            return out
        for k in REQUIRED_KEYS:
            d = f[k]
            info = {
                "shape": list(d.shape),
                "dtype": str(d.dtype),
                "chunks": list(d.chunks) if d.chunks is not None else None,
                "compression": d.compression,
                "compression_opts": d.compression_opts,
            }
            n = min(int(d.shape[0]), max_frames_for_stats)
            vals = []
            for i in range(n):
                arr = d[i]
                vals.append((int(arr.min()), int(arr.max()), float(arr.mean())))
            if vals:
                info["sample_min"] = min(v[0] for v in vals)
                info["sample_max"] = max(v[1] for v in vals)
                info["sample_mean"] = float(np.mean([v[2] for v in vals]))
            out["keys"][k] = info
        try:
            n0 = f["noisy"][0, 0]
            n1 = f["noisy"][0, 1]
            out["first_frame_duplicate"] = bool(np.array_equal(n0, n1))
        except Exception:
            out["first_frame_duplicate"] = None
    return out

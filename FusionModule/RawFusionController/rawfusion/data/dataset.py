from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_KEYS = ("noisy", "2dnr", "3dnr", "clean")


@dataclass(frozen=True)
class SampleRef:
    file_path: str
    kind: str  # root | group
    frame_index: int
    group_name: str = ""


def read_path_list(list_path: str | Path) -> List[str]:
    """Read a txt list. Absolute paths are preserved; relative paths are resolved from the list file directory."""
    list_path = Path(list_path)
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    files: List[str] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip().lstrip("\ufeff")
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (list_path.parent / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"H5 path in list does not exist: {p}")
        files.append(str(p))
    # stable de-duplication
    seen = set()
    out: List[str] = []
    for f in files:
        rp = str(Path(f).resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    if not out:
        raise RuntimeError(f"No usable h5 path in list: {list_path}")
    return out


def discover_h5_files(data_root: str | Path) -> List[str]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")
    files = sorted(str(p.resolve()) for p in root.rglob("*.h5"))
    if not files:
        raise RuntimeError(f"No .h5 files found under {root}")
    return files


def build_index(files: Sequence[str]) -> List[SampleRef]:
    refs: List[SampleRef] = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            if all(k in f and isinstance(f[k], h5py.Dataset) for k in REQUIRED_KEYS):
                n = int(f["noisy"].shape[0])
                refs.extend(SampleRef(fp, "root", i) for i in range(n))
                continue
            for gname in f.keys():
                g = f[gname]
                if isinstance(g, h5py.Group) and all(k in g and isinstance(g[k], h5py.Dataset) for k in REQUIRED_KEYS):
                    n = int(g["noisy"].shape[0])
                    refs.extend(SampleRef(fp, "group", i, gname) for i in range(n))
    if not refs:
        raise RuntimeError("No valid samples found. Expected keys: noisy, 2dnr, 3dnr, clean")
    return refs


def _normalize_uint16(x: np.ndarray, data_max_value: float, strict_range_check: bool) -> np.ndarray:
    if x.dtype != np.uint16:
        raise TypeError(f"Expected uint16 storage, got {x.dtype}")
    if strict_range_check:
        x_max = int(x.max())
        if x_max > data_max_value:
            raise ValueError(
                f"Sample value max={x_max} exceeds data_max_value={data_max_value}. "
                "Check bit depth, or run with --allow_value_clip."
            )
    y = x.astype(np.float32) / float(data_max_value)
    return np.clip(y, 0.0, 1.0)


def _edge_map(x_chw: np.ndarray) -> np.ndarray:
    """Compute a simple normalized gradient magnitude from a [1,H,W] or [H,W] float array."""
    if x_chw.ndim == 3:
        x = x_chw[0]
    else:
        x = x_chw
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gx[:, 1:] = np.abs(x[:, 1:] - x[:, :-1])
    gy[1:, :] = np.abs(x[1:, :] - x[:-1, :])
    g = np.sqrt(gx * gx + gy * gy)
    m = float(g.max())
    if m > 1e-8:
        g = g / m
    return g[None, ...].astype(np.float32)


def build_fusion_features(
    noisy_2hw: np.ndarray,
    dnr2_1hw: np.ndarray,
    dnr3_1hw: np.ndarray,
    feature_mode: str = "strong",
) -> np.ndarray:
    """
    Build model input features.

    strong mode returns 7 channels:
    prev, curr, |curr-prev|, 2dnr, 3dnr, |2dnr-3dnr|, edge(curr)
    """
    if noisy_2hw.shape[0] != 2:
        raise ValueError(f"noisy must be [2,H,W], got {noisy_2hw.shape}")
    prev = noisy_2hw[0:1]
    curr = noisy_2hw[1:2]
    motion = np.abs(curr - prev)
    disagree = np.abs(dnr2_1hw - dnr3_1hw)
    if feature_mode == "baseline":
        return np.concatenate([prev, curr], axis=0).astype(np.float32)
    if feature_mode == "no_edge":
        return np.concatenate([prev, curr, motion, dnr2_1hw, dnr3_1hw, disagree], axis=0).astype(np.float32)
    if feature_mode == "strong":
        edge = _edge_map(curr)
        return np.concatenate([prev, curr, motion, dnr2_1hw, dnr3_1hw, disagree, edge], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


class H5FusionDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        data_max_value: float = 4095.0,
        patch_size: int = 0,
        strict_range_check: bool = True,
        feature_mode: str = "strong",
        max_samples: int = 0,
        seed: int = 42,
    ):
        self.files = [str(Path(x).resolve()) for x in files]
        self.refs = build_index(self.files)
        if max_samples and max_samples > 0:
            self.refs = self.refs[: int(max_samples)]
        self.data_max_value = float(data_max_value)
        self.patch_size = int(patch_size or 0)
        self.strict_range_check = bool(strict_range_check)
        self.feature_mode = feature_mode
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.refs)

    def _read_frame(self, ref: SampleRef) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(ref.file_path, "r") as f:
            g = f if ref.kind == "root" else f[ref.group_name]
            i = ref.frame_index
            h, w = g["clean"].shape[-2:]
            if self.patch_size and self.patch_size > 0:
                ps = min(self.patch_size, h, w)
                y0 = int(self.rng.integers(0, max(h - ps + 1, 1)))
                x0 = int(self.rng.integers(0, max(w - ps + 1, 1)))
                sl2 = (slice(y0, y0 + ps), slice(x0, x0 + ps))
                noisy = g["noisy"][i, :, sl2[0], sl2[1]]
                dnr2 = g["2dnr"][i, sl2[0], sl2[1]]
                dnr3 = g["3dnr"][i, sl2[0], sl2[1]]
                clean = g["clean"][i, sl2[0], sl2[1]]
            else:
                noisy = g["noisy"][i]
                dnr2 = g["2dnr"][i]
                dnr3 = g["3dnr"][i]
                clean = g["clean"][i]
        return noisy, dnr2, dnr3, clean

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        ref = self.refs[idx]
        noisy, dnr2, dnr3, clean = self._read_frame(ref)

        if noisy.ndim != 3 or noisy.shape[0] != 2:
            raise ValueError(f"noisy frame must be [2,H,W], got {noisy.shape} @ {ref.file_path}")
        if dnr2.ndim != 2 or dnr3.ndim != 2 or clean.ndim != 2:
            raise ValueError(f"2dnr/3dnr/clean must be [H,W], got {dnr2.shape}, {dnr3.shape}, {clean.shape}")
        if noisy.shape[1:] != clean.shape or dnr2.shape != clean.shape or dnr3.shape != clean.shape:
            raise ValueError("Spatial shape mismatch among noisy/2dnr/3dnr/clean")

        noisy_f = _normalize_uint16(np.ascontiguousarray(noisy), self.data_max_value, self.strict_range_check)
        dnr2_f = _normalize_uint16(np.ascontiguousarray(dnr2), self.data_max_value, self.strict_range_check)[None, ...]
        dnr3_f = _normalize_uint16(np.ascontiguousarray(dnr3), self.data_max_value, self.strict_range_check)[None, ...]
        clean_f = _normalize_uint16(np.ascontiguousarray(clean), self.data_max_value, self.strict_range_check)[None, ...]
        x = build_fusion_features(noisy_f, dnr2_f, dnr3_f, self.feature_mode)

        return {
            "x": torch.from_numpy(x),
            "dnr2": torch.from_numpy(dnr2_f.astype(np.float32)),
            "dnr3": torch.from_numpy(dnr3_f.astype(np.float32)),
            "clean": torch.from_numpy(clean_f.astype(np.float32)),
            "file_path": ref.file_path,
            "frame_index": ref.frame_index,
        }

import re
import json
import random
import itertools
import math
import copy

import argparse
from sympy.codegen import Print
from torch.utils.data import Dataset, DataLoader
import os, csv, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml


def _natural_key(value: Any) -> List[Any]:
    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]

# ---- optional image IO backends ----
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import imageio.v3 as iio
    _HAS_IIO = True
except Exception:
    _HAS_IIO = False
# =========================
# Bayer / CFA utilities
# =========================
def _tile_from_cfa(cfa: str) -> List[str]:
    """
    CFA string like 'GBRG', 'RGGB', 'BGGR', 'GRBG'
    Returns 2x2 tile in row-major order: [t00, t01, t10, t11]
    """
    cfa = cfa.upper()
    if cfa not in {"GBRG", "RGGB", "BGGR", "GRBG"}:
        raise ValueError(f"Unsupported CFA: {cfa}. Use one of GBRG/RGGB/BGGR/GRBG.")
    return [cfa[0], cfa[1], cfa[2], cfa[3]]
def planes_to_mosaic(planes: Dict[str, torch.Tensor], cfa: str) -> torch.Tensor:
    """
    planes: {'R','G1','G2','B'} each [..., H, W]
    return: mosaic [..., H*2, W*2]
    """
    tile = _tile_from_cfa(cfa)  # 2x2 tile row-major: [t00,t01,t10,t11]

    # map greens to G1/G2 by scan order within tile
    g_list = []
    for t in tile:
        if t == "G":
            g_list.append("G1" if len(g_list) == 0 else "G2")
    if len(g_list) != 2:
        raise RuntimeError(f"CFA {cfa} must have 2 greens.")

    pos_keys = []
    gi = 0
    for t in tile:
        if t == "R":
            pos_keys.append("R")
        elif t == "B":
            pos_keys.append("B")
        elif t == "G":
            pos_keys.append(g_list[gi])
            gi += 1
        else:
            raise RuntimeError("Bad CFA token")

    sample = planes[pos_keys[0]]
    *prefix, H, W = sample.shape
    mosaic = torch.zeros((*prefix, H * 2, W * 2), device=sample.device, dtype=sample.dtype)

    mosaic[..., 0::2, 0::2] = planes[pos_keys[0]]
    mosaic[..., 0::2, 1::2] = planes[pos_keys[1]]
    mosaic[..., 1::2, 0::2] = planes[pos_keys[2]]
    mosaic[..., 1::2, 1::2] = planes[pos_keys[3]]
    return mosaic
def split_bayer_to_4ch(mosaic: np.ndarray, cfa: str = "GBRG",
                      out_order: str = "R,G1,G2,B") -> np.ndarray:
    """
    Split Bayer mosaic (H,W) into 4 planes (H/2,W/2).
    Two greens are preserved as G1 and G2 (their positions depend on CFA).

    out_order controls channel order; default: R,G1,G2,B
    Returns np.ndarray with shape (4, H//2, W//2), float32.
    """
    if mosaic.ndim != 2:
        raise ValueError(f"mosaic must be 2D (H,W), got {mosaic.shape}")

    H, W = mosaic.shape
    H2, W2 = H // 2, W // 2
    tile = _tile_from_cfa(cfa)  # [t00,t01,t10,t11]

    # positions in the 2x2 tile:
    # (0,0)->even,even ; (0,1)->even,odd ; (1,0)->odd,even ; (1,1)->odd,odd
    planes: Dict[str, np.ndarray] = {}
    # collect R/B single planes
    if tile[0] == "R": planes["R"] = mosaic[0::2, 0::2]
    if tile[0] == "B": planes["B"] = mosaic[0::2, 0::2]
    if tile[1] == "R": planes["R"] = mosaic[0::2, 1::2]
    if tile[1] == "B": planes["B"] = mosaic[0::2, 1::2]
    if tile[2] == "R": planes["R"] = mosaic[1::2, 0::2]
    if tile[2] == "B": planes["B"] = mosaic[1::2, 0::2]
    if tile[3] == "R": planes["R"] = mosaic[1::2, 1::2]
    if tile[3] == "B": planes["B"] = mosaic[1::2, 1::2]

    # two greens keep as G1/G2 by their tile positions (stable definition)
    # G1 = first green encountered in tile scan order, G2 = second
    g_list: List[np.ndarray] = []
    pos_arr = [
        mosaic[0::2, 0::2],  # t00
        mosaic[0::2, 1::2],  # t01
        mosaic[1::2, 0::2],  # t10
        mosaic[1::2, 1::2],  # t11
    ]
    for t, arr in zip(tile, pos_arr):
        if t == "G":
            g_list.append(arr)
    if len(g_list) != 2:
        raise RuntimeError(f"CFA {cfa} should have 2 greens, got {len(g_list)}")
    planes["G1"], planes["G2"] = g_list[0], g_list[1]

    # ensure R/B exist
    if "R" not in planes or "B" not in planes:
        raise RuntimeError(f"Failed to infer R/B planes for CFA={cfa}")

    order = [x.strip().upper() for x in out_order.split(",")]
    # map tokens
    token_map = {"R": "R", "B": "B", "G1": "G1", "G2": "G2"}
    out = []
    for tok in order:
        if tok not in token_map:
            raise ValueError(f"Bad out_order token: {tok}. Use R,G1,G2,B")
        out.append(planes[token_map[tok]].astype(np.float32))
    out_arr = np.stack(out, axis=0)  # (4,H/2,W/2)
    assert out_arr.shape[1] == H2 and out_arr.shape[2] == W2
    return out_arr
# ============================================================
# 3) Metrics
# ============================================================

def calc_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return float(psnr.detach().cpu().item())

def calc_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    if not _HAS_SKIMAGE:
        raise RuntimeError("SSIM needs scikit-image (pip install scikit-image)")
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    return float(_ssim(t, p, data_range=data_range))
# =========================
# File patterns (auto-detect)
# =========================
_RE_SCENE = re.compile(r"^scene(\d+)$", re.IGNORECASE)
_RE_ISO = re.compile(r"^ISO(\d+)$", re.IGNORECASE)
def detect_dataset_kind(iso_dir: Path) -> str:
    """
    Inspect filenames in one ISO folder to detect its kind.
    Returns: "noisy" | "gt" | "2dnr" | "3dnr" | "unknown"
    """
    names = [p.name for p in iso_dir.glob("*.tiff")]
    s = set(names)
    # noisy signatures
    if any(re.match(r"^frame\d+_noisy\d+\.tiff$", n, re.IGNORECASE) for n in s):
        return "noisy"
    # 3DNR signatures
    if any(n.lower().endswith("_clean_and_temporal_denoised.tiff") for n in s):
        return "3dnr"
    # gt or 2dnr share "_clean_and_slightly_denoised.tiff"
    if any(n.lower().endswith("_clean_and_slightly_denoised.tiff") for n in s):
        # can't distinguish gt vs 2dnr purely by name; treat as "sld"
        return "sld"
    return "unknown"
def read_tiff(path: Path) -> np.ndarray:
    if _HAS_CV2:
        img = cv2.imread(str(path), -1)
        if img is None:
            raise FileNotFoundError(f"Failed to read: {path}")
        return img
    if _HAS_IIO:
        img = iio.imread(str(path))
        return img
    raise RuntimeError("No image backend available. Install opencv-python or imageio.")
# =========================
# Dataset
# =========================
class CRVDSeqDataset(Dataset):
    """
    Sequence-wise CRVD-style loader that can align multiple roots:
      - indoor_raw_noisy (frameX_noisy0..9.tiff)
      - indoor_raw_gt    (frameX_clean_and_slightly_denoised.tiff)
      - 2DNR_*           (frameX_clean_and_slightly_denoised.tiff)
      - 3DNR_*           (frameX_clean_and_temporal_denoised.tiff)

    Returns a dict for each sample (one sequence by default).
    """

    def __init__(
        self,
        roots: Dict[str, str],
        scenes: Optional[List[int]] = None,
        iso_levels: Optional[List[int]] = None,
        frame_ids: Optional[List[int]] = None,   # default [1..7]
        sequence_mode: bool = True,
        noisy_versions: int = 10,
        noisy_pick: str = "random",              # "random" | "fixed" | "all"
        fixed_noisy_id: int = 0,
        strict_align: bool = True,
        # CFA / output
        cfa: str = "GBRG",
        color_mode: str = "mixed",               # "mixed" | "separated" | "mosaic"
        out_order: str = "R,G1,G2,B",
        # numeric
        to_float32: bool = True,
        normalize: str = "none",                 # "none" | "scale_0_1"
        scale_div: float = 4095.0,
        # performance
        build_index_cache: bool = False,
        index_cache_path: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.roots = {k: Path(v) for k, v in roots.items()}
        self.scenes = scenes
        self.iso_levels = iso_levels
        self.frame_ids = frame_ids if frame_ids is not None else list(range(1, 8))
        self.sequence_mode = sequence_mode
        self.noisy_versions = noisy_versions
        self.noisy_pick = noisy_pick.lower()
        self.fixed_noisy_id = fixed_noisy_id
        self.strict_align = strict_align

        self.cfa = cfa
        self.color_mode = color_mode.lower()
        self.out_order = out_order

        self.to_float32 = to_float32
        self.normalize = normalize.lower()
        self.scale_div = scale_div

        self.build_index_cache = build_index_cache
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None

        self.rng = random.Random(seed)

        self.samples = self._build_or_load_index()

    def _build_or_load_index(self) -> List[Dict[str, Any]]:
        cache_path = self.index_cache_path
        if cache_path is None:
            # default cache in root folder if build_index_cache
            any_root = next(iter(self.roots.values()))
            cache_path = any_root / "_crvd_seq_index.json"

        if self.build_index_cache and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        # choose a "primary" root to enumerate scene/iso; prefer noisy if present
        primary_key = "noisy" if "noisy" in self.roots else next(iter(self.roots.keys()))
        primary_root = self.roots[primary_key]

        samples: List[Dict[str, Any]] = []
        for scene_dir in sorted(primary_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            m = _RE_SCENE.match(scene_dir.name)
            if not m:
                continue
            scene_id = int(m.group(1))
            if self.scenes is not None and scene_id not in self.scenes:
                continue

            for iso_dir in sorted(scene_dir.iterdir()):
                if not iso_dir.is_dir():
                    continue
                m2 = _RE_ISO.match(iso_dir.name)
                if not m2:
                    continue
                iso = int(m2.group(1))
                if self.iso_levels is not None and iso not in self.iso_levels:
                    continue

                # each (scene, iso) is one sequence sample by default
                if self.sequence_mode:
                    samples.append({"scene": scene_id, "iso": iso})
                else:
                    # single-frame samples
                    for fid in self.frame_ids:
                        samples.append({"scene": scene_id, "iso": iso, "frame": fid})

        if self.build_index_cache:
            try:
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] failed to write index cache: {cache_path} ({e})")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_paths_for_frame(self, scene: int, iso: int, frame: int, noisy_id: Optional[int]) -> Dict[str, Optional[Path]]:
        """
        Return dict of paths for each source at a given (scene,iso,frame)
        Keys can include: noisy, noisy_clean, gt, nr2d, nr3d (depending on roots provided).
        """
        out: Dict[str, Optional[Path]] = {}

        for key, root in self.roots.items():
            base = root / f"scene{scene}" / f"ISO{iso}"

            if key == "noisy":
                if noisy_id is None:
                    out["noisy"] = None
                else:
                    out["noisy"] = base / f"frame{frame}_noisy{noisy_id}.tiff"
                    out["noisy_clean"] = base / f"frame{frame}_clean.tiff"
            else:
                # decide filename by sniffing folder or key hint
                # Prefer explicit key names: gt / 2dnr / 3dnr
                k = key.lower()
                if "3d" in k:
                    out[key] = base / f"frame{frame}_clean_and_temporal_denoised.tiff"
                else:
                    # gt or 2d typically share this suffix
                    out[key] = base / f"frame{frame}_clean_and_slightly_denoised.tiff"

        # validate existence depending on strict_align
        if self.strict_align:
            for k, p in out.items():
                if p is None:
                    continue
                if not p.exists():
                    raise FileNotFoundError(f"Missing {k}: {p}")
        else:
            for k, p in list(out.items()):
                if p is None:
                    continue
                if not p.exists():
                    out[k] = None

        return out

    def _postprocess_img(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            # Some readers may return (H,W,1). Squeeze.
            if img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                raise ValueError(f"Expected mosaic grayscale tiff, got {img.shape}")

        if self.to_float32:
            img = img.astype(np.float32)
        if self.normalize == "scale_0_1":
            img = img / float(self.scale_div)
        return img

    def _apply_color_mode(self, mosaic: np.ndarray) -> Any:
        """
        Returns:
          - mosaic mode: (H,W)
          - mixed mode : (4,H/2,W/2)
          - separated  : dict of planes {'R':..., 'G1':..., 'G2':..., 'B':...} each (H/2,W/2)
        """
        if self.color_mode == "mosaic":
            return mosaic
        planes4 = split_bayer_to_4ch(mosaic, cfa=self.cfa, out_order=self.out_order)  # (4,H/2,W/2)
        if self.color_mode == "mixed":
            return planes4
        if self.color_mode == "separated":
            # match out_order token names
            tokens = [x.strip().upper() for x in self.out_order.split(",")]
            return {tok: planes4[i] for i, tok in enumerate(tokens)}
        raise ValueError(f"Unknown color_mode: {self.color_mode}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        scene = item["scene"]
        iso = item["iso"]

        # pick noisy_id: sequence-wise consistency
        noisy_id: Optional[int] = None
        if "noisy" in self.roots:
            if self.noisy_pick == "random":
                noisy_id = self.rng.randint(0, self.noisy_versions - 1)
            elif self.noisy_pick == "fixed":
                noisy_id = int(self.fixed_noisy_id)
            elif self.noisy_pick == "all":
                # handle later: return all noisy versions
                noisy_id = None
            else:
                raise ValueError("noisy_pick must be random|fixed|all")

        # frames to load
        if self.sequence_mode:
            frames = self.frame_ids
        else:
            frames = [item["frame"]]

        # main output dict
        out: Dict[str, Any] = {
            "meta": {
                "scene": scene,
                "iso": iso,
                "frames": frames,
                "sequence_mode": self.sequence_mode,
                "cfa": self.cfa,
                "color_mode": self.color_mode,
                "out_order": self.out_order,
                "noisy_pick": self.noisy_pick,
                "noisy_id": noisy_id if self.noisy_pick != "all" else "all",
            },
            "data": {}  # each key -> tensor shaped [T, ...]
        }

        # If noisy_pick == all: we return noisy as [T, V, ...] (V=noisy_versions)
        if "noisy" in self.roots and self.noisy_pick == "all":
            # For each frame, stack 10 versions
            noisy_seq = []
            noisy_clean_seq = []
            for f in frames:
                versions = []
                for vid in range(self.noisy_versions):
                    paths = self._resolve_paths_for_frame(scene, iso, f, vid)
                    img = self._postprocess_img(read_tiff(paths["noisy"]))
                    versions.append(self._apply_color_mode(img))
                # clean
                paths0 = self._resolve_paths_for_frame(scene, iso, f, 0)
                clean_img = self._postprocess_img(read_tiff(paths0["noisy_clean"]))
                noisy_clean_seq.append(self._apply_color_mode(clean_img))

                noisy_seq.append(versions)

            # convert to tensors
            out["data"]["noisy"] = torch.from_numpy(np.array(noisy_seq, dtype=object))  # object if separated; debug mode
            out["data"]["noisy_clean"] = torch.from_numpy(np.array(noisy_clean_seq, dtype=object))
            # NOTE: for separated mode, you’ll likely want a custom collate; this is initial debug-friendly behavior.
            return out

        # Normal path: one noisy_id for whole sequence
        # We load each source into list, then stack into torch.
        per_key_lists: Dict[str, List[Any]] = {}

        for f in frames:
            paths = self._resolve_paths_for_frame(scene, iso, f, noisy_id)

            # read for each path key
            for k, p in paths.items():
                if p is None:
                    # allow missing in loose mode
                    continue
                img = self._postprocess_img(read_tiff(p))
                payload = self._apply_color_mode(img)
                per_key_lists.setdefault(k, []).append(payload)

        # pack to torch tensors (mixed/mosaic are numeric arrays; separated is dict -> keep dict-of-tensors)
        for k, lst in per_key_lists.items():
            if len(lst) != len(frames):
                if self.strict_align:
                    raise RuntimeError(f"Key {k}: loaded {len(lst)} frames, expected {len(frames)}")
                # loose: skip
                continue

            if self.color_mode in {"mosaic", "mixed"}:
                arr = np.stack(lst, axis=0)  # [T, ...]
                out["data"][k] = torch.from_numpy(arr)
            else:
                # separated: lst is list of dicts -> dict of stacked tensors
                toks = [x.strip().upper() for x in self.out_order.split(",")]
                dd = {}
                for tok in toks:
                    arr = np.stack([d[tok] for d in lst], axis=0)  # [T, H/2, W/2]
                    dd[tok] = torch.from_numpy(arr)
                out["data"][k] = dd

        return out
# =========================
# Color—Seperated—batch patch_permutations
# =========================
def Color_Seperatedexpand_batch_with_all_patch_permutations_generator(batch):
    """
    极致内存优化版：使用 Python 生成器，每次仅产生 1 种排列。
    内存消耗：几乎与原 Batch 持平。
    """
    data_dict = batch['data']
    meta_dict = batch['meta']

    # 1. 获取基础维度
    first_key = next(iter(data_dict.keys()))
    first_ch = next(iter(data_dict[first_key].keys()))
    B, T, H, W = data_dict[first_key][first_ch].shape
    h_half, w_half = H // 2, W // 2

    # 2. 预先切分 Patch (这一步是 View 操作，不占额外内存拷贝)
    # 我们把所有 Key 和 Channel 的 4 个 Patch 预先算好
    patch_storage = {}
    for data_key, channels in data_dict.items():
        patch_storage[data_key] = {}
        for ch, tensor in channels.items():
            # 利用 reshape 和 permute 得到 [B, 4, T, H/2, W/2]
            # 只要不调用 .contiguous()，它在内存中就只是一个“视图”
            x = tensor.reshape(B, T, 2, h_half, 2, w_half)
            patches = x.permute(0, 2, 4, 1, 3, 5).reshape(B, 4, T, h_half, w_half)
            patch_storage[data_key][ch] = patches

    # 3. 产生 24 种排列
    perms = list(itertools.permutations(range(4)))

    for perm in perms:
        perm_idx = torch.tensor(perm, dtype=torch.long)
        current_data = {key: {} for key in data_dict.keys()}

        for data_key in data_dict.keys():
            for ch in ['R', 'G1', 'G2', 'B']:
                # 获取当前排列的 Patch: [B, 4, T, H/2, W/2] -> [B, 4(reordered), T, H/2, W/2]
                reordered_patches = patch_storage[data_key][ch][:, perm_idx]

                # 重新拼回整图 [B, T, H, W]
                # 这里必须要 reshape，会产生临时内存，但函数结束后会被销毁
                res = reordered_patches.reshape(B, 2, 2, T, h_half, w_half)
                current_data[data_key][ch] = res.permute(0, 3, 1, 4, 2, 5).reshape(B, T, H, W).contiguous()

        # 构建当前的 Meta
        current_meta = {}
        for key, values in meta_dict.items():
            current_meta[key] = values  # 这一批的 meta 与原 meta 相同
        current_meta['patch_permutation'] = [perm] * B

        # 抛出当前这一组排列（大小为 B）
        yield {
            'meta': current_meta,
            'data': current_data
        }


def main():
    # =========================
    # CONFIG: 你只改这里
    # =========================
    CFG = {
        # dataset root
        "root": r"E:\CRVD_dataset",

        # sub dirs
        "noisy_dir": "indoor_raw_noisy",
        "gt_dir": "indoor_raw_gt",
        "nr2d_dir": "2DNR_bm3d",
        "nr3d_dir": "3DNR_vbm3d",

        # which sources to load
        "use_2d": True,
        "use_3d": True,

        # subset control (None=all)
        "scenes": None,                 # e.g. [1,2,3]
        "isos": None,                   # e.g. [1600, 3200]
        "frames": [1,2,3,4,5,6,7],      # keep order!

        # sampling mode
        "sequence_mode": True,          # True: (scene,ISO)->7帧; False: 单帧样本

        # noisy settings
        "noisy_versions": 10,
        "noisy_pick": "Fixed",         # "random" | "fixed" | "all"
        "fixed_noisy_id": 1,
        "seed": 0,

        # CFA / output
        "cfa": "GBRG",
        "color_mode": "separated",          # "mixed" | "separated" | "mosaic"
        "out_order": "R,G1,G2,B",       # Seprerated & Mixed Order

        # numeric
        "to_float32": True,
        "normalize": "none",            # "none" | "scale_0_1"
        "scale_div": 4095.0,

        # index cache
        "cache_index": True,
        "index_cache_path": "",         # "" -> 默认写到 root/_crvd_seq_index.json

        # dataloader
        "batch_size": 2,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": False,

        # debug
        "num_batches_to_print": 2,
        "strict_align": True,           # True: 缺文件直接报错，利于debug
    }

    # =========================
    # build roots dict
    # =========================
    root = Path(CFG["root"])
    roots = {
        "noisy": str(root / CFG["noisy_dir"]),
        "gt": str(root / CFG["gt_dir"]),
    }
    if CFG["use_2d"]:
        roots["nr2d"] = str(root / CFG["nr2d_dir"])
    if CFG["use_3d"]:
        roots["nr3d"] = str(root / CFG["nr3d_dir"])

    index_cache_path = CFG["index_cache_path"].strip() or None

    # =========================
    # dataset + loader
    # =========================
    ds = CRVDSeqDataset(
        roots=roots,
        scenes=CFG["scenes"],
        iso_levels=CFG["isos"],
        frame_ids=CFG["frames"],
        sequence_mode=CFG["sequence_mode"],
        noisy_versions=CFG["noisy_versions"],
        noisy_pick=CFG["noisy_pick"],
        fixed_noisy_id=CFG["fixed_noisy_id"],
        strict_align=CFG["strict_align"],
        cfa=CFG["cfa"],
        color_mode=CFG["color_mode"],
        out_order=CFG["out_order"],
        to_float32=CFG["to_float32"],
        normalize=CFG["normalize"],
        scale_div=CFG["scale_div"],
        build_index_cache=CFG["cache_index"],
        index_cache_path=index_cache_path,
        seed=CFG["seed"],
    )

    print(f"[INFO] dataset size = {len(ds)} samples")
    print(f"[INFO] roots = {roots}")
    print(f"[INFO] sequence_mode={CFG['sequence_mode']} frames={CFG['frames']} color_mode={CFG['color_mode']} cfa={CFG['cfa']}")

    loader = DataLoader(
        ds,
        batch_size=CFG["batch_size"],
        shuffle=CFG["shuffle"],
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        drop_last=CFG["drop_last"],
    )

    for bi, batch in enumerate(loader):
        print(f"\n===== batch {bi} =====")
        # summarize_batch(batch)
        for sub_batch in Color_Seperatedexpand_batch_with_all_patch_permutations_generator(batch):
            # inspect_single_expanded_sample(sub_batch, sample_idx=1, frame_indices=[0, 3, 6])
            nr2d_data = sub_batch['data']['nr2d']  # 这是一个字典
            R_tensor = nr2d_data['R']  # 取出 R 通道的 Tensor
            G1_tensor = nr2d_data['G1']
            G2_tensor = nr2d_data['G2']
            B_tensor = nr2d_data['B']
        if bi + 1 >= CFG["num_batches_to_print"]:
            break

# ========= 依赖你文件里已有的函数/类 =========
# Fusion-Net
# Fusion-Net
# Fusion-Net
# ============================================

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# ============================================================
# 1) FusionNet backbone -> GateNet (输出 0~1, 偏2DNR)
# ============================================================

class Resblock(nn.Module):
    def __init__(self, channel: int = 32):
        super().__init__()
        self.conv20 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv21 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs1 = self.conv21(rs1)
        return x + rs1

def init_weights_simple(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class FusionGateNet(nn.Module):
    """
    输入:  [B,3,H,W] = (prev, cur, cur-prev)
    输出:  [B,1,H,W] = gate x in [0,1], 语义偏2DNR
    backbone: FusionNet 的 4 Resblock
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(
            Resblock(base_ch), Resblock(base_ch), Resblock(base_ch), Resblock(base_ch)
        )
        self.conv_out = nn.Conv2d(base_ch, 1, 3, 1, 1, bias=True)
        init_weights_simple(self)

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        feat = self.backbone(feat)
        gate = torch.sigmoid(self.conv_out(feat))
        return gate
# ============================================================
# UnetGateNet – 经典 U-Net 结构，两下两上，带跳跃连接
# 输入: [B, 3, H, W]  输出: [B, 1, H, W] (sigmoid)
# 用法与 FusionGateNet 完全相同
# ============================================================
class UnetGateNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        # ---------- 编码器 (下采样) ----------
        # 第一层 (in_ch -> base_ch)
        self.enc1_conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        self.enc1_conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # 第二层 (base_ch -> base_ch*2)
        self.enc2_conv1 = nn.Conv2d(base_ch, base_ch*2, kernel_size=3, padding=1)
        self.enc2_conv2 = nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈层 (base_ch*2 -> base_ch*4)
        self.bottleneck_conv1 = nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(base_ch*4, base_ch*4, kernel_size=3, padding=1)

        # ---------- 解码器 (上采样) ----------
        # 第一次上采样 (base_ch*4 -> base_ch*2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1_conv1 = nn.Conv2d(base_ch*4 + base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.dec1_conv2 = nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3, padding=1)

        # 第二次上采样 (base_ch*2 -> base_ch)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2_conv1 = nn.Conv2d(base_ch*2 + base_ch, base_ch, kernel_size=3, padding=1)
        self.dec2_conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)

        # 输出层 (base_ch -> 1)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        init_weights_simple(self)   # 使用文件中已有的初始化函数

    def forward(self, x):
        # 编码器
        enc1 = self.relu(self.enc1_conv1(x))
        enc1 = self.relu(self.enc1_conv2(enc1))   # [B, base_ch, H, W]
        p1 = self.pool1(enc1)                     # [B, base_ch, H/2, W/2]

        enc2 = self.relu(self.enc2_conv1(p1))
        enc2 = self.relu(self.enc2_conv2(enc2))   # [B, base_ch*2, H/2, W/2]
        p2 = self.pool2(enc2)                      # [B, base_ch*2, H/4, W/4]

        # 瓶颈
        bottleneck = self.relu(self.bottleneck_conv1(p2))
        bottleneck = self.relu(self.bottleneck_conv2(bottleneck))  # [B, base_ch*4, H/4, W/4]

        # 解码器第一次上采样 + 跳跃连接
        up1 = self.up1(bottleneck)                  # [B, base_ch*4, H/2, W/2] (近似)
        # 确保尺寸与 enc2 一致（因输入尺寸可能为奇数，插值后可能稍有偏差）
        if up1.shape[-2:] != enc2.shape[-2:]:
            up1 = F.interpolate(up1, size=enc2.shape[-2:], mode='bilinear', align_corners=False)
        cat1 = torch.cat([up1, enc2], dim=1)        # [B, base_ch*4+base_ch*2, H/2, W/2]
        dec1 = self.relu(self.dec1_conv1(cat1))
        dec1 = self.relu(self.dec1_conv2(dec1))     # [B, base_ch*2, H/2, W/2]

        # 解码器第二次上采样 + 跳跃连接
        up2 = self.up2(dec1)                         # [B, base_ch*2, H, W] (近似)
        if up2.shape[-2:] != enc1.shape[-2:]:
            up2 = F.interpolate(up2, size=enc1.shape[-2:], mode='bilinear', align_corners=False)
        cat2 = torch.cat([up2, enc1], dim=1)         # [B, base_ch*2+base_ch, H, W]
        dec2 = self.relu(self.dec2_conv1(cat2))
        dec2 = self.relu(self.dec2_conv2(dec2))      # [B, base_ch, H, W]

        # 输出
        out = self.out_conv(dec2)                     # [B, 1, H, W]
        return torch.sigmoid(out)
# ============================================================
# 4) Core fusion: x偏2DNR
# ============================================================
def fuse_with_gate(nr2d: Dict[str, torch.Tensor],
                   nr3d: Dict[str, torch.Tensor],
                   x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    nr2d/nr3d: dict ch -> [B,6,H,W]  (frames 2..7)
    x:         [B,6,1,H,W] or [B,6,H,W]
    fused = 2DNR*x + 3DNR*(1-x)
    """
    if x.dim() == 5:
        x2 = x.squeeze(2)
    else:
        x2 = x
    out = {}
    for ch in ["R", "G1", "G2", "B"]:
        out[ch] = nr2d[ch] * x2 + nr3d[ch] * (1.0 - x2)
    return out
# ============================================================
# 5) Split by scene (避免泄露)
# ============================================================

def split_scenes(scene_ids: List[int],x) -> Tuple[List[int], List[int]]:
    s = sorted(set(scene_ids))
    n = len(s)
    n_train = max(1, int(round(n * x)))
    train = s[:n_train]
    val = s[n_train:]
    if len(val) == 0:
        val = [train.pop()]
    return train, val

def explore(batch):
    """
    只打印 batch 中每个样本的 scene 和 iso 值。
    支持 batch 为列表（多个样本）或单个样本字典。
    """
    if isinstance(batch, list):
        for i, sample in enumerate(batch):
            if isinstance(sample, dict) and 'meta' in sample:
                meta = sample['meta']
                scene = meta.get('scene')
                iso = meta.get('iso')
                # 如果 scene/iso 是 Tensor，提取数值
                if hasattr(scene, 'item'):
                    scene = scene.item()
                if hasattr(iso, 'item'):
                    iso = iso.item()
                print(f"Sample {i}: scene={scene}, iso={iso}")
            else:
                print(f"Sample {i}: unexpected structure (no 'meta' dict)")
    elif isinstance(batch, dict):
        # 单个样本的情况
        if 'meta' in batch:
            meta = batch['meta']
            scene = meta.get('scene')
            iso = meta.get('iso')
            if hasattr(scene, 'item'):
                scene = scene.item()
            if hasattr(iso, 'item'):
                iso = iso.item()
            print(f" 此次batch的scene是={scene}, iso={iso}")
        else:
            print("No 'meta' found in the batch")
    else:
        print("Batch is neither a list nor a dict")
# ============================================================
# VMamba-based GateNet (VmambaGateNet)
# 输入: [B, 3, H, W]  输出: [B, 1, H, W] (sigmoid)
# 采用 VMamba 的交叉扫描 Mamba 块，U-Net 式编码器-解码器
# ============================================================
class CrossScan(nn.Module):
    """VMamba 中的交叉扫描：将 2D 特征图按四个方向展开为序列"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 四个方向：行主序、行反序、列主序、列反序
        xs = []
        # 1. 行主序 (从左到右，从上到下)
        xs.append(x.view(B, C, -1).transpose(1, 2))  # (B, H*W, C)
        # 2. 行反序 (从右到左，从上到下)
        x_flip_h = torch.flip(x, dims=[3])
        xs.append(x_flip_h.view(B, C, -1).transpose(1, 2))
        # 3. 列主序 (从上到下，从左到右) -> 转置后按行主序
        x_t = x.transpose(2, 3)  # (B, C, W, H)
        xs.append(x_t.view(B, C, -1).transpose(1, 2))  # (B, W*H, C)
        # 4. 列反序 (从下到上，从左到右)
        x_t_flip_v = torch.flip(x_t, dims=[2])
        xs.append(x_t_flip_v.view(B, C, -1).transpose(1, 2))

        return torch.cat(xs, dim=0)  # (4*B, L, C)


class CrossMerge(nn.Module):
    """将四个方向的扫描结果合并回 2D 特征图"""
    def __init__(self):
        super().__init__()

    def forward(self, xs: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # xs: (4*B, L, C)  其中 L = H*W 或 W*H
        B = xs.shape[0] // 4
        C = xs.shape[-1]
        # 分离四个方向
        xs = xs.view(4, B, -1, C)  # (4, B, L, C)
        out = []
        for i in range(4):
            out.append(xs[i].transpose(1, 2).view(B, C, H, W))
        # 合并：四个方向取平均
        out = torch.stack(out, dim=0).mean(dim=0)  # (B, C, H, W)
        return out


class VSSBlock(nn.Module):
    """VMamba 的基本块：LayerNorm -> CrossScan -> Mamba -> CrossMerge + 残差"""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(dim)
        self.cross_scan = CrossScan()
        self.cross_merge = CrossMerge()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        xs = self.cross_scan(x)               # (4*B, L, C)
        xs = self.mamba(xs)                    # (4*B, L, C)
        x = self.cross_merge(xs, H, W)         # (B, C, H, W)

        return x + shortcut


class PatchEmbed(nn.Module):
    """图像分块嵌入，同时下采样 2 倍"""
    def __init__(self, in_ch=3, embed_dim=32, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                     # (B, embed_dim, H/2, W/2)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging(nn.Module):
    """下采样：空间减半，通道加倍"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 将 2x2 块拼接到通道上
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H//2)*(W//2), -1)   # (B, L, 4*C)
        x = self.norm(x)
        x = self.reduction(x)                  # (B, L, 2*C)
        # 恢复为 2D 特征图
        x = x.transpose(1, 2).view(B, 2*C, H//2, W//2)
        return x


class PatchExpanding(nn.Module):
    """上采样：空间加倍，通道减半（与 PatchMerging 对称）"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim // 2, 2 * (dim // 2), bias=False)  # 简化实现，实际应使用转置卷积
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        x = self.norm(x)
        x = self.expand(x)                        # (B, H, W, 2*C)
        x = x.view(B, H, W, 2, C//2)              # 沿通道维度扩展
        x = x.permute(0, 1, 3, 2, 4).contiguous() # (B, H, 2, W, C//2)
        x = x.reshape(B, H*2, W, C//2).transpose(2, 1).contiguous()  # (B, H*2, W, C//2)
        x = x.permute(0, 3, 1, 2).contiguous()    # (B, C//2, H*2, W)
        return x


class VmambaGateNet(nn.Module):
    """
    使用 VMamba 块的 U-Net 结构，两下两上，带跳跃连接。
    输入: [B, 3, H, W]  输出: [B, 1, H, W] (sigmoid)
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()

        # 编码器
        self.patch_embed = PatchEmbed(in_ch, base_ch, stride=2)  # 下采样2倍
        self.enc1 = nn.Sequential(
            VSSBlock(base_ch),
            VSSBlock(base_ch),
        )
        self.down1 = PatchMerging(base_ch)      # 下采样2倍 -> base_ch*2

        self.enc2 = nn.Sequential(
            VSSBlock(base_ch*2),
            VSSBlock(base_ch*2),
        )
        self.down2 = PatchMerging(base_ch*2)    # 下采样2倍 -> base_ch*4

        # 瓶颈
        self.bottleneck = nn.Sequential(
            VSSBlock(base_ch*4),
            VSSBlock(base_ch*4),
        )

        # 解码器
        self.up2 = PatchExpanding(base_ch*4)    # 上采样2倍 -> base_ch*2
        self.dec2 = nn.Sequential(
            VSSBlock(base_ch*2),
            VSSBlock(base_ch*2),
        )

        self.up1 = PatchExpanding(base_ch*2)    # 上采样2倍 -> base_ch
        self.dec1 = nn.Sequential(
            VSSBlock(base_ch),
            VSSBlock(base_ch),
        )

        # 输出头
        self.output = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1),
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        x = self.patch_embed(x)          # [B, base_ch, H/2, W/2]
        enc1_out = self.enc1(x)           # [B, base_ch, H/2, W/2]
        x = self.down1(enc1_out)          # [B, base_ch*2, H/4, W/4]

        enc2_out = self.enc2(x)           # [B, base_ch*2, H/4, W/4]
        x = self.down2(enc2_out)          # [B, base_ch*4, H/8, W/8]

        # 瓶颈
        x = self.bottleneck(x)            # [B, base_ch*4, H/8, W/8]

        # 解码器（带跳跃连接）
        x = self.up2(x)                   # [B, base_ch*2, H/4, W/4]
        # 可能因尺寸取整导致不一致，需调整
        if x.shape[-2:] != enc2_out.shape[-2:]:
            x = F.interpolate(x, size=enc2_out.shape[-2:], mode='bilinear', align_corners=False)
        x = x + enc2_out                   # 跳跃连接（也可用 concat，这里用加法简化）
        x = self.dec2(x)                   # [B, base_ch*2, H/4, W/4]

        x = self.up1(x)                    # [B, base_ch, H/2, W/2]
        if x.shape[-2:] != enc1_out.shape[-2:]:
            x = F.interpolate(x, size=enc1_out.shape[-2:], mode='bilinear', align_corners=False)
        x = x + enc1_out                    # 跳跃连接
        x = self.dec1(x)                    # [B, base_ch, H/2, W/2]

        # 上采样回原尺寸
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [B, base_ch, H, W]

        out = self.output(x)                # [B, 1, H, W]
        return torch.sigmoid(out)
# ============================================================
# 6) Runner config
# ============================================================
@dataclass
class PipelineCFG:#一个用于快速配置的类
    # paths
    root: str = r"E:\CRVD_dataset"
    noisy_dir: str = "indoor_raw_noisy"
    use_noisy: bool = False
    gt_dir: str = "indoor_raw_gt"
    nr2d_dir: str = "2DNR_bilateral"
    nr3d_dir: str = "3DNR_vbm3d"

    # CFA
    cfa: str = "GBRG"

    # dataset subset (None = all)
    scenes: Optional[List[int]] = None
    isos: Optional[List[int]] = None
    frames: List[int] = None  # None -> [1..7]

    # noisy sampling
    noisy_pick: str = "fixed"
    fixed_noisy_id: int = 1

    # training
    device: str = "cuda"
    epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0

    batch_size: int = 2
    num_workers: int = 0
    shuffle: bool = True

    # gate input plane (你自己改)
    gate_plane: str = "G1"  # 'R'/'G1'/'G2'/'B'

    # perm augmentation
    train_use_all_24_perms: bool = True
    perms_per_batch: int = 24  # 想减速调试就改小于24

    # metrics range (normalize=none -> 4095; normalize to [0,1] -> 1.0)
    data_range: float = 1

    # outputs
    out_dir: str = "./runs_gate"
    val_csv: str = "val_metrics_by_scene.csv"
    model_type: str = "FusionGateNet"   # <-- 新增
    base_ch: int = 32                   # <-- 新增
    Train_portion: float = 0.8          # <-- 新增
# ============================================================
# 7) Full pipeline runner
# ============================================================
class FullPipeline:
    MODEL_REGISTRY = {
        "FusionGateNet": FusionGateNet,
        "UnetGateNet": UnetGateNet,
    }
    def __init__(self, cfg: PipelineCFG):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model_class = self.MODEL_REGISTRY.get(cfg.model_type)
        if model_class is None:
            raise ValueError(f"未知的 model_type: {cfg.model_type}。可用选项: {list(self.MODEL_REGISTRY.keys())}")

        self.model = model_class(in_ch=3, base_ch=cfg.base_ch).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        os.makedirs(cfg.out_dir, exist_ok=True)

        self.train_loader, self.val_loader = self._build_loaders()

    def _build_loaders(self):
        root = Path(self.cfg.root)
        roots = {
            "gt": str(root / self.cfg.gt_dir),
            "nr2d": str(root / self.cfg.nr2d_dir),
            "nr3d": str(root / self.cfg.nr3d_dir),
        }
        if self.cfg.use_noisy:
            roots["noisy"] = str(root / self.cfg.noisy_dir)

        frames = self.cfg.frames if self.cfg.frames is not None else [1,2,3,4,5,6,7]

        # tmp enumerate to get all scene ids (then split by scene)
        tmp_ds = CRVDSeqDataset(
            roots=roots,
            scenes=self.cfg.scenes,
            iso_levels=self.cfg.isos,
            frame_ids=frames,
            sequence_mode=True,
            noisy_versions=10,
            noisy_pick=self.cfg.noisy_pick,
            fixed_noisy_id=self.cfg.fixed_noisy_id,
            strict_align=True,
            cfa=self.cfg.cfa,
            color_mode="separated",
            out_order="R,G1,G2,B",
            to_float32=True,
            normalize="scale_0_1",
            scale_div=4095.0,
            build_index_cache=False,
            index_cache_path=None,
            seed=0,
        )
        all_scene_ids = [s["scene"] for s in tmp_ds.samples]
        train_scenes, val_scenes = split_scenes(all_scene_ids,self.cfg.Train_portion)

        train_ds = CRVDSeqDataset(
            roots=roots,
            scenes=train_scenes,
            iso_levels=self.cfg.isos,
            frame_ids=frames,
            sequence_mode=True,
            noisy_versions=10,
            noisy_pick=self.cfg.noisy_pick,
            fixed_noisy_id=self.cfg.fixed_noisy_id,
            strict_align=True,
            cfa=self.cfg.cfa,
            color_mode="separated",
            out_order="R,G1,G2,B",
            to_float32=True,
            normalize="scale_0_1",
            scale_div=4095.0,
            build_index_cache=False,
            index_cache_path=None,
            seed=0,
        )

        val_ds = CRVDSeqDataset(
            roots=roots,
            scenes=val_scenes,
            iso_levels=self.cfg.isos,
            frame_ids=frames,
            sequence_mode=True,
            noisy_versions=10,
            noisy_pick="fixed",
            fixed_noisy_id=self.cfg.fixed_noisy_id,
            strict_align=True,
            cfa=self.cfg.cfa,
            color_mode="separated",
            out_order="R,G1,G2,B",
            to_float32=True,
            normalize="scale_0_1",
            scale_div=4095.0,
            build_index_cache=False,
            index_cache_path=None,
            seed=0,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers, pin_memory=False, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False, drop_last=False
        )
        return train_loader, val_loader

    def _make_gate_input(self, seq_plane_7: torch.Tensor) -> torch.Tensor:
        """
        seq_plane_7: [B,7,H,W]
        return: [B*6,3,H,W] for frames 2..7 (skip first frame)
        """
        prev = seq_plane_7[:, :-1]     # [B,6,H,W]
        cur  = seq_plane_7[:, 1:]      # [B,6,H,W]
        diff = cur - prev
        x = torch.stack([prev, cur, diff], dim=2)  # [B,6,3,H,W]
        B, S, C, H, W = x.shape
        return x.reshape(B * S, C, H, W)

    def _get_planes7(self, batch: Dict[str, Any], key: str) -> Dict[str, torch.Tensor]:
        d = batch["data"][key]
        return {ch: d[ch] for ch in ["R","G1","G2","B"]}  # each [B,7,H,W]

    def _slice_6(self, planes7: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {ch: v[:, 1:] for ch, v in planes7.items()}  # [B,6,H,W]

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total = 0.0
        steps = 0
        t0 = time.time()

        for batch in self.train_loader:
            explore(batch)

            perm_iter = Color_Seperatedexpand_batch_with_all_patch_permutations_generator(batch) \
                if self.cfg.train_use_all_24_perms else iter([batch])

            for pi, sub_batch in enumerate(perm_iter):
                if self.cfg.train_use_all_24_perms and pi >= self.cfg.perms_per_batch:
                    break
                print(f"{pi}th in starts")
                # CPU -> get planes
                nr2d_7 = self._get_planes7(sub_batch, "nr2d")
                nr3d_7 = self._get_planes7(sub_batch, "nr3d")
                gt_7   = self._get_planes7(sub_batch, "gt")

                # gate input plane (single plane for now)
                gate_seq = nr2d_7[self.cfg.gate_plane].to(self.device, non_blocking=True)  # [B,7,H,W]
                xin = self._make_gate_input(gate_seq)  # [B*6,3,H,W] (GPU)
                x = self.model(xin)                    # [B*6,1,H,W]
                B = gate_seq.shape[0]
                x = x.reshape(B, 6, 1, gate_seq.shape[-2], gate_seq.shape[-1])  # [B,6,1,H,W]

                # move fusion sources to GPU (frames2..7)
                nr2d_6 = {ch: nr2d_7[ch][:, 1:].to(self.device, non_blocking=True) for ch in ["R","G1","G2","B"]}
                nr3d_6 = {ch: nr3d_7[ch][:, 1:].to(self.device, non_blocking=True) for ch in ["R","G1","G2","B"]}
                gt_6   = {ch: gt_7[ch][:, 1:].to(self.device, non_blocking=True)   for ch in ["R","G1","G2","B"]}

                fused_6 = fuse_with_gate(nr2d_6, nr3d_6, x)

                fused_m = planes_to_mosaic(fused_6, self.cfg.cfa)  # [B,6,Hf,Wf]
                gt_m    = planes_to_mosaic(gt_6,   self.cfg.cfa)

                loss = torch.mean(torch.abs(fused_m - gt_m))  # L1

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                total += float(loss.detach().cpu().item())
                steps += 1

        print(f"[Epoch {epoch}] train_loss={total/max(1,steps):.6f} time={time.time()-t0:.1f}s")

    @torch.no_grad()
    # def validate(self, epoch: int) -> Dict[int, Dict[str, float]]:
    #     self.model.eval()
    #     acc: Dict[int, Dict[str, List[float]]] = {}
    #
    #     for batch in self.val_loader:
    #         # meta 在默认 collate 后通常是 list
    #         scene_id = int(batch["meta"]["scene"][0]) if isinstance(batch["meta"]["scene"], list) else int(batch["meta"]["scene"])
    #
    #         nr2d_7 = self._get_planes7(batch, "nr2d")
    #         nr3d_7 = self._get_planes7(batch, "nr3d")
    #         gt_7   = self._get_planes7(batch, "gt")
    #
    #         # gate inference
    #         gate_seq = nr2d_7[self.cfg.gate_plane].to(self.device)
    #         xin = self._make_gate_input(gate_seq)
    #         x = self.model(xin)  # [1*6,1,H,W]
    #         x = x.reshape(1, 6, 1, gate_seq.shape[-2], gate_seq.shape[-1])
    #
    #         nr2d_6 = {ch: nr2d_7[ch][:, 1:].to(self.device) for ch in ["R","G1","G2","B"]}
    #         nr3d_6 = {ch: nr3d_7[ch][:, 1:].to(self.device) for ch in ["R","G1","G2","B"]}
    #         gt_6   = {ch: gt_7[ch][:, 1:].to(self.device)   for ch in ["R","G1","G2","B"]}
    #
    #         fused_6 = fuse_with_gate(nr2d_6, nr3d_6, x)
    #
    #         fused_m = planes_to_mosaic(fused_6, self.cfg.cfa)[0]  # [6,Hf,Wf]
    #         gt_m    = planes_to_mosaic(gt_6,   self.cfg.cfa)[0]
    #         nr2d_m  = planes_to_mosaic(nr2d_6, self.cfg.cfa)[0]
    #         nr3d_m  = planes_to_mosaic(nr3d_6, self.cfg.cfa)[0]
    #
    #         acc.setdefault(scene_id, {
    #             "pred_psnr": [], "pred_ssim": [],
    #             "2d_psnr": [],   "2d_ssim": [],
    #             "3d_psnr": [],   "3d_ssim": [],
    #         })
    #
    #         for t in range(fused_m.shape[0]):  # 6 frames (skip first)
    #             pred = fused_m[t]; gt = gt_m[t]
    #             d2   = nr2d_m[t];   d3 = nr3d_m[t]
    #
    #             acc[scene_id]["pred_psnr"].append(calc_psnr(pred, gt, self.cfg.data_range))
    #             acc[scene_id]["2d_psnr"].append(calc_psnr(d2,   gt, self.cfg.data_range))
    #             acc[scene_id]["3d_psnr"].append(calc_psnr(d3,   gt, self.cfg.data_range))
    #
    #             if _HAS_SKIMAGE:
    #                 acc[scene_id]["pred_ssim"].append(calc_ssim(pred, gt, self.cfg.data_range))
    #                 acc[scene_id]["2d_ssim"].append(calc_ssim(d2,   gt, self.cfg.data_range))
    #                 acc[scene_id]["3d_ssim"].append(calc_ssim(d3,   gt, self.cfg.data_range))
    #
    #     report = {sid: {k: float(np.mean(v)) if len(v) else float("nan") for k, v in d.items()}
    #               for sid, d in acc.items()}
    #
    #     # save CSV each epoch
    #     # 构建多级输出目录
    #     epoch_dir = Path(self.cfg.out_dir) / self.cfg.model_type / f"gate_{self.cfg.gate_plane}" / f"epoch_{epoch:03d}"
    #     epoch_dir.mkdir(parents=True, exist_ok=True)
    #
    #     model_name = self.cfg.model_type
    #     # 保存 CSV 文件（文件名保持原样）
    #     out_csv = epoch_dir / f"{model_name}-gate_{self.cfg.gate_plane}-epoch{epoch:03d}-base{self.cfg.base_ch}"
    #     # out_csv = epoch_dir / f"epoch_{epoch:03d}_{self.cfg.val_csv}"
    #     cols = ["scene", "pred_psnr", "pred_ssim", "2d_psnr", "2d_ssim", "3d_psnr", "3d_ssim"]
    #     with out_csv.open("w", newline="", encoding="utf-8") as f:
    #         w = csv.DictWriter(f, fieldnames=cols)
    #         w.writeheader()
    #         for sid in sorted(report.keys()):
    #             row = {"scene": sid}
    #             row.update(report[sid])
    #             w.writerow(row)
    #
    #     print(f"[VAL epoch {epoch}] saved {out_csv}")
    #
    #     # 保存模型（文件名格式不变）
    #     filename = f"{model_name}-gate_{self.cfg.gate_plane}-epoch{epoch:03d}-base{self.cfg.base_ch}.pth"
    #     save_path = epoch_dir / filename
    #     torch.save(self.model.state_dict(), save_path)
    #     print(f"模型已保存至: {save_path}")
    #
    #     return report
    @torch.no_grad()
    def validate(self, epoch: int) -> None:  # 返回类型可改为 None，因为不再返回聚合报告
        self.model.eval()
        # 用于收集所有详细记录
        all_records = []

        for batch in self.val_loader:
            # 提取 scene 和 iso（可能因 collate 成为列表）
            if isinstance(batch["meta"]["scene"], list):
                scene_id = int(batch["meta"]["scene"][0])
                iso = int(batch["meta"]["iso"][0])
            else:
                scene_id = int(batch["meta"]["scene"])
                iso = int(batch["meta"]["iso"])

            nr2d_7 = self._get_planes7(batch, "nr2d")
            nr3d_7 = self._get_planes7(batch, "nr3d")
            gt_7 = self._get_planes7(batch, "gt")

            # gate inference
            gate_seq = nr2d_7[self.cfg.gate_plane].to(self.device)
            xin = self._make_gate_input(gate_seq)
            x = self.model(xin)  # [1*6,1,H,W]
            x = x.reshape(1, 6, 1, gate_seq.shape[-2], gate_seq.shape[-1])

            nr2d_6 = {ch: nr2d_7[ch][:, 1:].to(self.device) for ch in ["R", "G1", "G2", "B"]}
            nr3d_6 = {ch: nr3d_7[ch][:, 1:].to(self.device) for ch in ["R", "G1", "G2", "B"]}
            gt_6 = {ch: gt_7[ch][:, 1:].to(self.device) for ch in ["R", "G1", "G2", "B"]}

            fused_6 = fuse_with_gate(nr2d_6, nr3d_6, x)

            fused_m = planes_to_mosaic(fused_6, self.cfg.cfa)[0]  # [6, Hf, Wf]
            gt_m = planes_to_mosaic(gt_6, self.cfg.cfa)[0]
            nr2d_m = planes_to_mosaic(nr2d_6, self.cfg.cfa)[0]
            nr3d_m = planes_to_mosaic(nr3d_6, self.cfg.cfa)[0]

            # 遍历 6 帧，记录每帧的指标
            for t in range(fused_m.shape[0]):
                pred = fused_m[t]
                gt = gt_m[t]
                d2 = nr2d_m[t]
                d3 = nr3d_m[t]

                pred_psnr = calc_psnr(pred, gt, self.cfg.data_range)
                d2_psnr = calc_psnr(d2, gt, self.cfg.data_range)
                d3_psnr = calc_psnr(d3, gt, self.cfg.data_range)

                record = {
                    "scene": scene_id,
                    "iso": iso,
                    "frame": t + 2,  # 帧索引从 2 开始（因为跳过了第1帧）
                    "pred_psnr": pred_psnr,
                    "2d_psnr": d2_psnr,
                    "3d_psnr": d3_psnr,
                }

                if _HAS_SKIMAGE:
                    record["pred_ssim"] = calc_ssim(pred, gt, self.cfg.data_range)
                    record["2d_ssim"] = calc_ssim(d2, gt, self.cfg.data_range)
                    record["3d_ssim"] = calc_ssim(d3, gt, self.cfg.data_range)

                all_records.append(record)

        # 构建输出目录（模型类型/通道/epoch）
        epoch_dir = Path(self.cfg.out_dir) / self.cfg.model_type / f"gate_{self.cfg.gate_plane}" / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # 保存 CSV（使用正确的文件名和扩展名）
        csv_filename = f"{self.cfg.model_type}-gate_{self.cfg.gate_plane}-epoch{epoch:03d}-base{self.cfg.base_ch}.csv"
        csv_path = epoch_dir / csv_filename

        # 定义 CSV 列名
        fieldnames = ["scene", "iso", "frame", "pred_psnr", "2d_psnr", "3d_psnr"]
        if _HAS_SKIMAGE:
            fieldnames += ["pred_ssim", "2d_ssim", "3d_ssim"]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in all_records:
                writer.writerow(rec)

        print(f"[VAL epoch {epoch}] saved detailed metrics to {csv_path}")

        # 保存模型权重（文件名也相应修正）
        model_filename = f"{self.cfg.model_type}-gate_{self.cfg.gate_plane}-epoch{epoch:03d}-base{self.cfg.base_ch}.pth"
        model_path = epoch_dir / model_filename
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # 不再返回报告，因为不再需要
    def run(self):
        for ep in range(1, self.cfg.epochs + 1):
            print(f"[EPOCH {ep}] start")
            self.train_one_epoch(ep)
            print(f"[EPOCH {ep}] end")
            self.validate(ep)
# ============================================================
# 8) Click-run main
# ============================================================
def main_pipeline(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    # 将字典转换为 PipelineCFG 对象（需处理嵌套或缺失字段）
    CFG = PipelineCFG(**config_dict)  # 假设所有字段名一致
    print("Start training")
    runner = FullPipeline(CFG)
    runner.run()


def _rf_squeeze_video_data(data: np.ndarray, channel_index: Optional[int] = None) -> np.ndarray:
    data = np.squeeze(data)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    while data.ndim > 3:
        if channel_index is not None and data.ndim == 4:
            data = data[:, channel_index, ...]
        elif data.shape[-1] <= 4:
            data = data[..., 0]
        else:
            data = data[:, 0, ...]
    if data.ndim != 3:
        raise ValueError(f"Unexpected video shape after squeeze: {data.shape}")
    return data


def _rf_extract_int(pattern: re.Pattern, path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = pattern.search(part)
        if match:
            return int(match.group(1))
    match = pattern.search(path.name)
    return int(match.group(1)) if match else None


def _rf_list_h5_files(input_root: Path) -> List[Path]:
    if input_root.is_file() and input_root.suffix.lower() in {".h5", ".hdf5"}:
        return [input_root]
    files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}]
    return sorted(files, key=lambda p: _natural_key(p))


def _rf_write_tiff(path: Path, frame: np.ndarray, clip: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(frame), 0, clip).astype(np.uint16)
    try:
        import tifffile as tiff
        tiff.imwrite(path, arr)
    except Exception:
        if _HAS_IIO:
            iio.imwrite(path, arr)
        elif _HAS_CV2:
            cv2.imwrite(str(path), arr)
        else:
            raise RuntimeError("Need tifffile, imageio, or cv2 to write TIFF files")


def export_h5_to_raw_fusion_dataset(
    input_root: str,
    output_root: str,
    clean_key: str = "clean",
    noisy_key: str = "noisy",
    nr2d_key: str = "2dnr",
    nr3d_key: str = "3dnr",
    noisy_channel: Optional[int] = None,
    default_scene: int = 1,
    default_iso: int = 1600,
    noisy_id: int = 1,
    clip: float = 4095.0,
    cfa: str = "GBRG",
    write_config: bool = False,
    raw_fusion_frames: int = 7,
    include_noisy: bool = False,
) -> None:
    import h5py

    scene_re = re.compile(r"scene\s*[_-]?(\d+)", re.IGNORECASE)
    iso_re = re.compile(r"ISO\s*[_-]?(\d+)", re.IGNORECASE)
    input_path = Path(input_root)
    output_path = Path(output_root)
    files = _rf_list_h5_files(input_path)
    if not files:
        raise FileNotFoundError(f"No H5 files found under {input_path}")

    frame_counters: Dict[Tuple[int, int], int] = {}
    records = []

    for h5_path in files:
        with h5py.File(h5_path, "r") as h5:
            scene = int(h5.attrs["scene"]) if "scene" in h5.attrs else (_rf_extract_int(scene_re, h5_path) or default_scene)
            iso = int(h5.attrs["iso"]) if "iso" in h5.attrs else (_rf_extract_int(iso_re, h5_path) or default_iso)
            stacks = {
                "clean": _rf_squeeze_video_data(h5[clean_key][...]),
                "nr2d": _rf_squeeze_video_data(h5[nr2d_key][...]),
                "nr3d": _rf_squeeze_video_data(h5[nr3d_key][...]),
            }
            if include_noisy:
                stacks["noisy"] = _rf_squeeze_video_data(h5[noisy_key][...], noisy_channel)

        base_shape = stacks["clean"].shape
        for key, stack in stacks.items():
            if stack.shape != base_shape:
                raise ValueError(f"Shape mismatch in {h5_path}: clean={base_shape}, {key}={stack.shape}")

        counter_key = (scene, iso)
        frame_counters.setdefault(counter_key, 0)
        start_frame = frame_counters[counter_key] + 1

        for local_idx in range(base_shape[0]):
            frame_id = frame_counters[counter_key] + 1
            frame_counters[counter_key] = frame_id

            gt_dir = output_path / "indoor_raw_gt" / f"scene{scene}" / f"ISO{iso}"
            nr2d_dir = output_path / "2DNR_bilateral" / f"scene{scene}" / f"ISO{iso}"
            nr3d_dir = output_path / "3DNR_vbm3d" / f"scene{scene}" / f"ISO{iso}"

            gt_out = gt_dir / f"frame{frame_id}_clean_and_slightly_denoised.tiff"
            nr2d_out = nr2d_dir / f"frame{frame_id}_clean_and_slightly_denoised.tiff"
            nr3d_out = nr3d_dir / f"frame{frame_id}_clean_and_temporal_denoised.tiff"

            _rf_write_tiff(gt_out, stacks["clean"][local_idx], clip)
            _rf_write_tiff(nr2d_out, stacks["nr2d"][local_idx], clip)
            _rf_write_tiff(nr3d_out, stacks["nr3d"][local_idx], clip)
            noisy_out = None
            if include_noisy:
                noisy_dir = output_path / "indoor_raw_noisy" / f"scene{scene}" / f"ISO{iso}"
                noisy_out = noisy_dir / f"frame{frame_id}_noisy{noisy_id}.tiff"
                _rf_write_tiff(noisy_out, stacks["noisy"][local_idx], clip)
                _rf_write_tiff(noisy_dir / f"frame{frame_id}_clean.tiff", stacks["clean"][local_idx], clip)
            records.append({
                "scene": scene,
                "iso": iso,
                "frame": frame_id,
                "source_h5": str(h5_path),
                "source_frame": local_idx,
                "noisy": str(noisy_out) if noisy_out is not None else None,
                "gt": str(gt_out),
                "nr2d": str(nr2d_out),
                "nr3d": str(nr3d_out),
            })

        print(f"[EXPORT] {h5_path} -> scene{scene}/ISO{iso}/frame{start_frame}..{frame_counters[counter_key]}")

    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "h5_export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"frames": records}, f, ensure_ascii=False, indent=2)

    if write_config:
        cfg = {
            "root": str(output_path),
            "noisy_dir": "indoor_raw_noisy",
            "use_noisy": include_noisy,
            "gt_dir": "indoor_raw_gt",
            "nr2d_dir": "2DNR_bilateral",
            "nr3d_dir": "3DNR_vbm3d",
            "cfa": cfa,
            "scenes": None,
            "isos": None,
            "frames": list(range(1, raw_fusion_frames + 1)),
            "noisy_pick": "fixed",
            "fixed_noisy_id": noisy_id,
            "device": "cuda",
            "epochs": 2,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": True,
            "gate_plane": "G1",
            "train_use_all_24_perms": True,
            "perms_per_batch": 24,
            "data_range": 1,
            "out_dir": "./runs_gate",
            "model_type": "FusionGateNet",
            "base_ch": 32,
            "Train_portion": 0.8,
        }
        with (output_path / "raw_fusion_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[EXPORT DONE] {len(records)} frames -> {output_path}")


def _rf_stack_to_planes_dict(stack: np.ndarray, cfa: str, device: torch.device, scale_div: float) -> Dict[str, torch.Tensor]:
    stack = _rf_squeeze_video_data(stack)
    planes = [split_bayer_to_4ch(frame, cfa=cfa, out_order="R,G1,G2,B") for frame in stack]
    arr = np.stack(planes, axis=0).astype(np.float32) / float(scale_div)  # [T,4,H/2,W/2]
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    return {
        "R": tensor[:, :, 0],
        "G1": tensor[:, :, 1],
        "G2": tensor[:, :, 2],
        "B": tensor[:, :, 3],
    }


def _rf_make_gate_input(seq_plane: torch.Tensor) -> torch.Tensor:
    prev = seq_plane[:, :-1]
    cur = seq_plane[:, 1:]
    diff = cur - prev
    x = torch.stack([prev, cur, diff], dim=2)
    b, s, c, h, w = x.shape
    return x.reshape(b * s, c, h, w)


def _rf_output_path_for_h5(input_root: Path, h5_path: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / h5_path.name
    try:
        rel = h5_path.relative_to(input_root)
    except ValueError:
        rel = Path(h5_path.name)
    return output_root / rel


class CheckpointUnetGateNet(nn.Module):
    """U-Net gate layout used by the provided best.pth checkpoint."""

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.enc1 = self.ConvBlock(in_ch, base_ch)
        self.enc2 = self.ConvBlock(base_ch, base_ch * 2)
        self.bottleneck = self.ConvBlock(base_ch * 2, base_ch * 4)
        self.dec2 = self.ConvBlock(base_ch * 6, base_ch * 2)
        self.dec1 = self.ConvBlock(base_ch * 3, base_ch)
        self.out = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return torch.sigmoid(self.out(dec1))


@torch.no_grad()
def infer_fusion_h5(
    input_root: str,
    output_root: str,
    model_path: str,
    model_type: str = "FusionGateNet",
    base_ch: int = 32,
    gate_plane: str = "G1",
    cfa: str = "GBRG",
    nr2d_key: str = "2dnr",
    nr3d_key: str = "3dnr",
    output_key: str = "fusion",
    first_frame_source: str = "2dnr",
    scale_div: float = 4095.0,
    compression: str = "gzip",
    device: str = "cuda",
    copy_inputs: bool = True,
) -> None:
    import h5py

    model_registry = {
        "FusionGateNet": FusionGateNet,
        "UnetGateNet": UnetGateNet,
        "CheckpointUnetGateNet": CheckpointUnetGateNet,
    }
    if model_type not in model_registry:
        raise ValueError(f"Unknown model_type={model_type}. Use one of {list(model_registry)}")
    if gate_plane not in {"R", "G1", "G2", "B"}:
        raise ValueError("gate_plane must be one of R/G1/G2/B")
    if first_frame_source not in {"2dnr", "3dnr"}:
        raise ValueError("first_frame_source must be 2dnr or 3dnr")

    run_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model_registry[model_type](in_ch=3, base_ch=base_ch).to(run_device)
    checkpoint = torch.load(model_path, map_location=run_device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()

    input_path = Path(input_root)
    output_path = Path(output_root)
    files = _rf_list_h5_files(input_path)
    if not files:
        raise FileNotFoundError(f"No H5 files found under {input_path}")

    compression_arg = None if compression == "none" else compression
    output_path.mkdir(parents=True, exist_ok=True)

    for h5_path in files:
        with h5py.File(h5_path, "r") as src:
            if nr2d_key not in src or nr3d_key not in src:
                raise KeyError(f"{h5_path} must contain {nr2d_key} and {nr3d_key}")
            nr2d = _rf_squeeze_video_data(src[nr2d_key][...])
            nr3d = _rf_squeeze_video_data(src[nr3d_key][...])

        if nr2d.shape != nr3d.shape:
            raise ValueError(f"Shape mismatch in {h5_path}: {nr2d_key}={nr2d.shape}, {nr3d_key}={nr3d.shape}")
        if nr2d.shape[0] < 2:
            raise ValueError(f"Need at least 2 frames for AI gate inference, got {nr2d.shape[0]} in {h5_path}")

        nr2d_planes = _rf_stack_to_planes_dict(nr2d, cfa, run_device, scale_div)
        nr3d_planes = _rf_stack_to_planes_dict(nr3d, cfa, run_device, scale_div)
        gate_seq = nr2d_planes[gate_plane]
        gate_input = _rf_make_gate_input(gate_seq)
        gate = model(gate_input)
        gate = gate.reshape(1, nr2d.shape[0] - 1, 1, gate_seq.shape[-2], gate_seq.shape[-1])

        nr2d_tail = {ch: tensor[:, 1:] for ch, tensor in nr2d_planes.items()}
        nr3d_tail = {ch: tensor[:, 1:] for ch, tensor in nr3d_planes.items()}
        fused_tail = fuse_with_gate(nr2d_tail, nr3d_tail, gate)
        fused_tail_mosaic = planes_to_mosaic(fused_tail, cfa)[0].detach().cpu().numpy()
        fused_tail_mosaic = np.clip(np.rint(fused_tail_mosaic * float(scale_div)), 0, scale_div).astype(np.uint16)

        first_frame = nr2d[:1] if first_frame_source == "2dnr" else nr3d[:1]
        fused = np.concatenate([first_frame.astype(np.uint16), fused_tail_mosaic], axis=0)

        out_h5 = _rf_output_path_for_h5(input_path, h5_path, output_path)
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "r") as src, h5py.File(out_h5, "w") as dst:
            if copy_inputs:
                for key in src.keys():
                    dst.create_dataset(key, data=src[key][...], compression=compression_arg)
            dst.create_dataset(output_key, data=fused, compression=compression_arg)
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["fusion_model_path"] = str(model_path)
            dst.attrs["fusion_model_type"] = model_type
            dst.attrs["fusion_gate_plane"] = gate_plane
            dst.attrs["fusion_first_frame_source"] = first_frame_source
        print(f"[FUSION] {h5_path} -> {out_h5} ({output_key}, shape={fused.shape})")

    print(f"[FUSION DONE] output: {output_path}")


# ============================================================================
# Local-alignment self-supervised RAW denoising (new mode)
#
# This section is intentionally independent from the original gate pipeline
# above.  It keeps the copied script backwards-compatible while adding a
# trainable U-Net that consumes locally aligned temporal observations.
# ============================================================================

def _la_read_frame(raw: np.memmap, index: int, black: float, container_scale: float,
                   clip: float, stride: int, cfa: str) -> np.ndarray:
    """Read one 16-bit mosaic frame and return normalized R,G1,G2,B planes."""
    mosaic = raw[index].astype(np.float32) / float(container_scale)
    planes = split_bayer_to_4ch(mosaic, cfa=cfa, out_order="R,G1,G2,B")
    if stride > 1:
        planes = planes[:, ::stride, ::stride]
    return np.clip((planes - float(black)) / float(clip), 0.0, 1.0).astype(np.float32)


def _la_open_raw(path: Path, frames: int, height: int, width: int) -> np.memmap:
    expected = int(frames) * int(height) * int(width)
    actual = path.stat().st_size // np.dtype(np.uint16).itemsize
    if actual != expected:
        raise ValueError(
            f"RAW size mismatch for {path}: expected {expected} uint16 values "
            f"({frames}x{height}x{width}), found {actual}. "
            "Set --local_frames/--local_height/--local_width explicitly."
        )
    return np.memmap(path, dtype=np.uint16, mode="r", shape=(frames, height, width))


def _la_align_to_current(cur_nr: np.ndarray, hist_nr: np.ndarray,
                         hist_payload: np.ndarray,
                         alignment_mode: str = "farneback") -> Tuple[np.ndarray, np.ndarray]:
    """Warp arbitrary history channels onto current coordinates and return trust."""
    if alignment_mode not in {"farneback", "identity"}:
        raise ValueError(f"Unsupported local alignment mode: {alignment_mode}")
    cur_luma = np.ascontiguousarray((cur_nr[1] + cur_nr[2]) * 0.5, dtype=np.float32)
    hist_luma = np.ascontiguousarray((hist_nr[1] + hist_nr[2]) * 0.5, dtype=np.float32)
    if alignment_mode == "identity":
        trust = np.exp(-np.abs(cur_luma - hist_luma) / 0.035)
        return hist_payload.astype(np.float32), np.clip(trust, 0.0, 1.0).astype(np.float32)
    if not _HAS_CV2:
        raise RuntimeError("Local alignment requires opencv-python (cv2).")
    # Flow direction is current -> history, so remapping history with x+flow
    # samples the history pixel corresponding to each current pixel.
    flow = cv2.calcOpticalFlowFarneback(
        cur_luma, hist_luma, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = cur_luma.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped_payload = np.empty_like(hist_payload)
    for channel in range(hist_payload.shape[0]):
        warped_payload[channel] = cv2.remap(
            hist_payload[channel], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    # Forward-backward and photometric checks suppress moving/occluded pixels.
    back = cv2.calcOpticalFlowFarneback(
        hist_luma, cur_luma, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    back_x = cv2.remap(back[..., 0], map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    back_y = cv2.remap(back[..., 1], map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    fb_error = np.sqrt((flow[..., 0] + back_x) ** 2 +
                       (flow[..., 1] + back_y) ** 2)
    photo_error = np.abs(cur_luma - cv2.remap(
        hist_luma, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    ))
    # The thresholds are in normalized RAW units / Bayer-plane pixels.  The
    # product is deliberately conservative: uncertain alignment means baseline.
    trust = np.exp(-photo_error / 0.035) * np.exp(-fb_error / 1.5)
    valid = cv2.remap(np.ones_like(cur_luma), map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    trust = np.clip(trust * valid, 0.0, 1.0).astype(np.float32)
    return warped_payload.astype(np.float32), trust


class LocalAlignSequence:
    """Lazy RAW reader.  Frames remain memory-mapped; only sampled frames load."""

    def __init__(self, source: Path, denoised: Path, frames: int, height: int,
                 width: int, stride: int, source_black: float, denoised_black: float,
                 container_scale: float, clip: float, cfa: str,
                 alignment_mode: str = "farneback"):
        self.frames = int(frames)
        self.stride = int(stride)
        self.clip = float(clip)
        self.src = _la_open_raw(source, frames, height, width)
        self.nr2d = _la_open_raw(denoised, frames, height, width)
        self.source_black = float(source_black)
        self.denoised_black = float(denoised_black)
        self.container_scale = float(container_scale)
        self.cfa = cfa.upper()
        self.alignment_mode = alignment_mode
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._align_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._future_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def frame(self, kind: str, index: int) -> np.ndarray:
        key = (kind, int(index))
        if key not in self._cache:
            if kind == "raw":
                self._cache[key] = _la_read_frame(
                    self.src, index, self.source_black, self.container_scale,
                    self.clip, self.stride, self.cfa
                )
            elif kind == "2dnr":
                self._cache[key] = _la_read_frame(
                    self.nr2d, index, self.denoised_black, 1.0,
                    self.clip, self.stride, self.cfa
                )
            else:
                raise ValueError(f"Unknown frame kind: {kind}")
            # Keep a small bounded cache so a CPU smoke test does not consume GBs.
            while len(self._cache) > 12:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def aligned_history(self, current: int, delta: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (int(current), int(delta))
        if key not in self._align_cache:
            cur_nr = self.frame("2dnr", current)
            hist = current - delta
            payload = np.concatenate([self.frame("raw", hist), self.frame("2dnr", hist)], axis=0)
            warped, trust = _la_align_to_current(
                cur_nr, self.frame("2dnr", hist), payload, self.alignment_mode
            )
            self._align_cache[key] = (warped[:4], warped[4:], trust)
            while len(self._align_cache) > 6:
                self._align_cache.pop(next(iter(self._align_cache)))
        return self._align_cache[key]

    def aligned_future_target(self, current: int, delta: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        future = int(current) + int(delta)
        if future < 0 or future >= self.frames:
            raise IndexError(f"Future frame {future} is outside [0, {self.frames})")
        key = (int(current), int(delta))
        if key not in self._future_cache:
            cur_nr = self.frame("2dnr", current)
            # The future frame is held out from model inputs and only provides
            # a Noise2Noise-style target after being aligned to the current one.
            self._future_cache[key] = _la_align_to_current(
                cur_nr, self.frame("2dnr", future), self.frame("raw", future), self.alignment_mode
            )
            while len(self._future_cache) > 6:
                self._future_cache.pop(next(iter(self._future_cache)))
        return self._future_cache[key]


class _LAConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalAlignResidualUNet(nn.Module):
    """Dual-branch U-Net with current-frame skips and confidence-gated history."""

    def __init__(self, in_ch: int = 26, base_ch: int = 16):
        super().__init__()
        if in_ch != 26:
            raise ValueError(f"LocalAlignResidualUNet requires 26 input channels, got {in_ch}")
        self.current_enc1 = _LAConvBlock(8, base_ch)
        self.temporal_enc1 = _LAConvBlock(18, base_ch)
        self.fuse1 = _LAConvBlock(base_ch * 2, base_ch)
        self.current_enc2 = _LAConvBlock(base_ch, base_ch * 2)
        self.temporal_enc2 = _LAConvBlock(base_ch, base_ch * 2)
        self.fuse2 = _LAConvBlock(base_ch * 4, base_ch * 2)
        self.bottleneck = _LAConvBlock(base_ch * 2, base_ch * 4)
        self.dec2 = _LAConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = _LAConvBlock(base_ch * 2 + base_ch, base_ch)
        self.residual = nn.Conv2d(base_ch, 4, 1)
        self.gate = nn.Conv2d(base_ch, 4, 1)
        # Identity initialization makes the first inference exactly 2DNR.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != 26:
            raise ValueError(f"Expected 26 local-alignment input channels, got {x.shape[1]}")
        confidence = x[:, 24:26].amin(dim=1, keepdim=True).clamp(0.0, 1.0)
        current1 = self.current_enc1(x[:, :8])
        temporal1 = self.temporal_enc1(x[:, 8:])
        fused1 = self.fuse1(torch.cat([current1, temporal1 * confidence], dim=1))
        current2 = self.current_enc2(F.max_pool2d(fused1, 2))
        temporal2 = self.temporal_enc2(F.max_pool2d(temporal1, 2))
        confidence2 = F.interpolate(confidence, size=temporal2.shape[-2:], mode="area")
        fused2 = self.fuse2(torch.cat([current2, temporal2 * confidence2], dim=1))
        bottleneck = self.bottleneck(F.max_pool2d(fused2, 2))
        d2 = F.interpolate(bottleneck, size=fused2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, fused2], dim=1))
        d1 = F.interpolate(d2, size=current1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, current1], dim=1))
        residual = 0.25 * torch.tanh(self.residual(d1))
        learned_gate = torch.sigmoid(self.gate(d1))
        effective_gate = learned_gate * confidence
        output = torch.clamp(base + residual * effective_gate, 0.0, 1.0)
        return output, effective_gate


def _la_input_full(seq: LocalAlignSequence, current: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cur_raw = seq.frame("raw", current)
    cur_nr = seq.frame("2dnr", current)
    hist1_raw, hist1_nr, trust1 = seq.aligned_history(current, 1)
    hist2_raw, hist2_nr, trust2 = seq.aligned_history(current, 2)
    trust = np.minimum(trust1, trust2)
    inp = np.concatenate([
        cur_raw, cur_nr, hist1_raw, hist1_nr, hist2_raw, hist2_nr,
        trust1[None], trust2[None],
    ], axis=0).astype(np.float32)
    return inp, cur_nr.astype(np.float32), trust[None].astype(np.float32)


def _la_future_target_bundle(seq: LocalAlignSequence, current: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    future_targets = []
    future_trusts = []
    for delta in (1, 2):
        if current + delta >= seq.frames:
            continue
        future_raw, future_trust = seq.aligned_future_target(current, delta)
        future_targets.append(future_raw)
        future_trusts.append(future_trust)
    if not future_targets:
        raise IndexError(f"No future target available for frame {current}")

    target_stack = np.stack(future_targets, axis=0)
    trust_stack = np.stack(future_trusts, axis=0)
    weight_sum = np.maximum(trust_stack.sum(axis=0), 1e-6)
    pseudo_target = (target_stack * trust_stack[:, None]).sum(axis=0) / weight_sum[None]
    if len(future_targets) > 1:
        disagreement = np.abs(target_stack[0] - target_stack[1]).mean(axis=0)
        agreement = np.exp(-disagreement / 0.04).astype(np.float32)
        target_variance = np.square(disagreement).astype(np.float32)
    else:
        agreement = np.ones_like(weight_sum, dtype=np.float32)
        target_variance = np.zeros_like(weight_sum, dtype=np.float32)
    target_trust = np.clip(weight_sum / len(future_targets), 0.0, 1.0) * agreement
    return target_stack.astype(np.float32), trust_stack.astype(np.float32), \
        pseudo_target.astype(np.float32), target_trust.astype(np.float32), target_variance


def _la_patch(seq: LocalAlignSequence, current: int, y: int, x: int,
              patch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inp_full, base_full, history_trust = _la_input_full(seq, current)
    _, _, target_full, target_trust, _ = _la_future_target_bundle(seq, current)
    supervision_trust = np.minimum(history_trust[0], target_trust)
    sl = np.s_[..., y:y + patch, x:x + patch]
    return inp_full[sl].astype(np.float32), base_full[sl].astype(np.float32), \
        target_full[sl].astype(np.float32), history_trust[sl].astype(np.float32), \
        supervision_trust[None, y:y + patch, x:x + patch].astype(np.float32)


def _la_fixed_splits(frames: int) -> tuple[range, range, range]:
    """Return the audited non-overlapping temporal split for company sequences."""
    if frames != 200:
        raise ValueError(
            "The local-alignment protocol is defined only for the 200-frame company sequences; "
            f"got {frames}. Do not invent a new split without an audited protocol."
        )
    return range(2, 120), range(135, 156), range(169, 199)


@torch.no_grad()
def _la_evaluate(model: LocalAlignResidualUNet, seq: LocalAlignSequence,
                 frame_indices: range, patch: int, device: torch.device,
                 max_frames: int = 8, max_patches: int = 8) -> Dict[str, float]:
    model.eval()
    base_errors, ai_errors, trust_values = [], [], []
    frame_ids = list(frame_indices)
    if len(frame_ids) > max_frames:
        frame_ids = [frame_ids[i] for i in np.linspace(0, len(frame_ids) - 1, max_frames, dtype=int)]
    for current in frame_ids:
        shape = seq.frame("raw", current).shape
        h, w = shape[-2:]
        positions = [(y, x) for y in range(0, h - patch + 1, patch)
                     for x in range(0, w - patch + 1, patch)]
        if len(positions) > max_patches:
            positions = [positions[i] for i in np.linspace(0, len(positions) - 1, max_patches, dtype=int)]
        for y, x in positions:
            inp, base, target, history_trust, supervision_trust = _la_patch(seq, current, y, x, patch)
            if float(supervision_trust.mean()) < 0.25:
                continue
            xb = torch.from_numpy(inp[None]).to(device)
            bb = torch.from_numpy(base[None]).to(device)
            pred, _ = model(xb, bb)
            weight = torch.from_numpy(supervision_trust[None]).to(device)
            denom = float(weight.sum().cpu()) * 4.0 + 1e-6
            base_errors.append(float((torch.abs(bb - torch.from_numpy(target[None]).to(device)) * weight).sum().cpu()) / denom)
            ai_errors.append(float((torch.abs(pred - torch.from_numpy(target[None]).to(device)) * weight).sum().cpu()) / denom)
            trust_values.append(float(history_trust.mean()))
    base_mae = float(np.mean(base_errors)) if base_errors else math.nan
    ai_mae = float(np.mean(ai_errors)) if ai_errors else math.nan
    return {
        "proxy_mae_2dnr": base_mae,
        "proxy_mae_ai": ai_mae,
        "proxy_improvement": base_mae - ai_mae if base_errors else math.nan,
        "mean_alignment_trust": float(np.mean(trust_values)) if trust_values else math.nan,
        "patches": len(ai_errors),
    }


def train_local_align_unet(source: str, denoised: str, out: str,
                           frames: int = 200, height: int = 1080, width: int = 1920,
                           source_black: float = 252.0, denoised_black: float = 300.0,
                           container_scale: float = 16.0, clip: float = 4095.0,
                           stride: int = 2, patch: int = 64, steps: int = 80,
                           base_ch: int = 16,
                           learning_rate: float = 2e-4, seed: int = 0,
                           device: str = "cpu", eval_frames: int = 8,
                           eval_patches: int = 8, validate_every: int = 20,
                           min_improvement: float = 2e-5, cfa: str = "RGGB",
                           alignment_mode: str = "farneback") -> Dict[str, Any]:
    """Train and validate the local-alignment U-Net using held-out noisy frames."""
    random.seed(seed)
    np.random.seed(seed)
    run_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    seq = LocalAlignSequence(Path(source), Path(denoised), frames, height, width,
                             stride, source_black, denoised_black, container_scale, clip, cfa,
                             alignment_mode)
    model = LocalAlignResidualUNet(in_ch=26, base_ch=base_ch).to(run_device)
    ema_model = copy.deepcopy(model).to(run_device)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    identity_state = copy.deepcopy(model.state_dict())
    best_state = copy.deepcopy(identity_state)
    best_selection_improvement = 0.0
    ema_decay = 0.995
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    train_range, selection_range, report_range = _la_fixed_splits(frames)
    train_indices = list(train_range)
    main_steps = int(steps)
    if main_steps < 1:
        raise ValueError("Training steps must be positive")
    probe_shape = seq.frame("raw", train_indices[0]).shape
    if patch < 4 or patch > min(probe_shape[-2:]):
        raise ValueError(f"patch must be in [4, {min(probe_shape[-2:])}], got {patch}")
    print(f"[local-align] device={run_device} train_frames={train_indices[0]}..{train_indices[-1]} "
          f"selection_frames={selection_range.start}..{selection_range.stop - 1} "
          f"report_frames={report_range.start}..{report_range.stop - 1} stride={stride}")

    last_loss = math.nan
    for step in range(main_steps):
        current = rng.choice(train_indices)
        probe = seq.frame("raw", current)
        h, w = probe.shape[-2:]
        candidate_patches = []
        for _ in range(16):
            y = rng.randrange(0, h - patch + 1)
            x = rng.randrange(0, w - patch + 1)
            inp, base, target, history_trust, supervision_trust = _la_patch(seq, current, y, x, patch)
            score = float(supervision_trust.mean())
            if score >= 0.20:
                candidate_patches.append((score, y, x, inp, base, target, history_trust, supervision_trust))
        if not candidate_patches:
            continue
        high = [candidate for candidate in candidate_patches if candidate[0] >= 0.70]
        medium = [candidate for candidate in candidate_patches if 0.40 <= candidate[0] < 0.70]
        if high and rng.random() < 0.60:
            selected = rng.choice(high)
        elif medium and rng.random() < 0.75:
            selected = rng.choice(medium)
        else:
            selected = rng.choice(candidate_patches)
        _, y, x, inp, base, target, history_trust, supervision_trust = selected
        xb = torch.from_numpy(inp[None]).to(run_device)
        bb = torch.from_numpy(base[None]).to(run_device)
        history_tb = torch.from_numpy(history_trust[None]).to(run_device)
        supervision_tb = torch.from_numpy(supervision_trust[None]).to(run_device)
        target_t = torch.from_numpy(target[None]).to(run_device)
        future_targets_full, future_trusts_full, _, _, future_variance_full = _la_future_target_bundle(
            seq, current
        )
        target_stack_t = torch.from_numpy(
            future_targets_full[..., y:y + patch, x:x + patch]
        ).to(run_device)
        target_trust_stack_t = torch.from_numpy(
            future_trusts_full[:, y:y + patch, x:x + patch]
        ).to(run_device)
        target_precision_t = torch.from_numpy(
            1.0 / (1.0 + future_variance_full[None, y:y + patch, x:x + patch] / 0.0025)
        ).to(run_device).clamp(0.10, 1.0)
        pred, gate = model(xb, bb)
        weight = (supervision_tb * target_precision_t).clamp_min(0.05)
        consensus_loss = (torch.sqrt((pred - target_t) ** 2 + 1e-4) * weight).sum() / (
            weight.sum() * 4.0 + 1e-6
        )
        pair_weight = (
            torch.minimum(history_tb, target_trust_stack_t[:, None]) * target_precision_t
        ).clamp_min(0.05)
        pair_error = torch.sqrt((pred[:, None] - target_stack_t) ** 2 + 1e-4)
        pair_loss = (pair_error * pair_weight).sum() / (pair_weight.sum() * 4.0 + 1e-6)
        data_loss = 0.65 * consensus_loss + 0.35 * pair_loss
        # Keep changes small where the temporal evidence is weak.  Since the
        # output head is zero-initialized, training starts exactly at 2DNR.
        identity_loss = 0.02 * (torch.abs(pred - bb) * (1.0 - history_tb)).mean()
        gate_loss = 0.002 * (gate * (1.0 - history_tb)).mean()
        loss = data_loss + identity_loss + gate_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                ema_parameter.mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
        last_loss = float(loss.detach().cpu())
        if step % 10 == 0 or step == main_steps - 1:
            print(f"[local-align step {step:03d}] loss={last_loss:.6f} trust={float(history_tb.mean()):.3f}")
        if (step + 1) % max(1, validate_every) == 0 or step == main_steps - 1:
            selection = _la_evaluate(
                ema_model, seq, selection_range, patch, run_device,
                max_frames=eval_frames, max_patches=eval_patches
            )
            improvement = selection["proxy_improvement"]
            print(f"[local-align selection] step={step:03d} improvement={improvement:.8f}")
            if math.isfinite(improvement) and improvement > max(min_improvement, best_selection_improvement):
                best_selection_improvement = improvement
                best_state = copy.deepcopy(ema_model.state_dict())

    accepted = best_selection_improvement > min_improvement
    model.load_state_dict(best_state if accepted else identity_state)
    report = _la_evaluate(model, seq, report_range, patch, run_device,
                          max_frames=eval_frames, max_patches=eval_patches)
    report.update({"steps": main_steps, "last_loss": last_loss,
                   "model": "LocalAlignResidualUNet", "architecture": "dual_branch_26ch_v1",
                   "input_channels": 26,
                   "target_mode": "future_t1_t2_consensus_plus_pairwise_n2n",
                   "target_variance_scale": 0.0025,
                   "alignment_mode": alignment_mode,
                   "ema_decay": ema_decay,
                   "selection_proxy_improvement": best_selection_improvement,
                   "minimum_selection_improvement": min_improvement,
                   "accepted_against_2dnr": accepted,
                   "train_frame_range": [train_range.start, train_range.stop - 1],
                   "selection_frame_range": [selection_range.start, selection_range.stop - 1],
                   "report_frame_range": [report_range.start, report_range.stop - 1],
                   "cfa": cfa.upper(),
                   "source": str(source), "denoised": str(denoised)})
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": report,
                "architecture": "dual_branch_26ch_v1",
                "base_ch": base_ch, "stride": stride}, out_path.with_suffix(".pth"))
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _la_tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be positive and overlap must satisfy 0 <= overlap < tile")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def _la_predict_tiled(model: LocalAlignResidualUNet, inp: np.ndarray, base: np.ndarray,
                      device: torch.device, tile: int, overlap: int) -> np.ndarray:
    """Run a frame in overlapping tiles, avoiding a full-resolution CPU spike."""
    _, h, w = inp.shape
    output_sum = np.zeros_like(base, dtype=np.float32)
    weight_sum = np.zeros((1, h, w), dtype=np.float32)
    for y in _la_tile_starts(h, tile, overlap):
        for x in _la_tile_starts(w, tile, overlap):
            y2, x2 = min(h, y + tile), min(w, x + tile)
            xb = torch.from_numpy(inp[None, :, y:y2, x:x2]).to(device)
            bb = torch.from_numpy(base[None, :, y:y2, x:x2]).to(device)
            pred, _ = model(xb, bb)
            output_sum[:, y:y2, x:x2] += pred[0].cpu().numpy()
            weight_sum[:, y:y2, x:x2] += 1.0
    return output_sum / np.maximum(weight_sum, 1.0)


def _la_planes_to_mosaic(planes: np.ndarray, cfa: str) -> np.ndarray:
    """Inverse of split_bayer_to_4ch for a single [R,G1,G2,B] frame."""
    if planes.shape[0] != 4:
        raise ValueError(f"Expected 4 planes, got {planes.shape}")
    plane_map = {"R": planes[0], "G1": planes[1], "G2": planes[2], "B": planes[3]}
    h, w = planes.shape[-2:]
    mosaic = np.empty((h * 2, w * 2), dtype=planes.dtype)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    green_index = 0
    for token, (yy, xx) in zip(_tile_from_cfa(cfa), positions):
        if token == "G":
            key = "G1" if green_index == 0 else "G2"
            green_index += 1
        else:
            key = token
        mosaic[yy::2, xx::2] = plane_map[key]
    return mosaic


def infer_local_align_unet(source: str, denoised: str, model_path: str, output: str,
                           frames: int = 200, height: int = 1080, width: int = 1920,
                           source_black: float = 252.0, denoised_black: float = 300.0,
                           container_scale: float = 16.0, clip: float = 4095.0,
                           stride: Optional[int] = None, cfa: str = "RGGB",
                           tile: int = 128, overlap: int = 32,
                           device: str = "cpu", overwrite: bool = False) -> None:
    """Generate a RAW sequence, preserving the exact 2DNR baseline when needed."""
    output_path = Path(output)
    source_path = Path(source)
    denoised_path = Path(denoised)
    if output_path.resolve() == denoised_path.resolve():
        raise ValueError("Refusing to overwrite the input 2DNR RAW.")
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Refusing to overwrite the input source RAW.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --local_infer_overwrite to replace it.")
    run_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=run_device, weights_only=True)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    architecture = checkpoint.get("architecture") if isinstance(checkpoint, dict) else None
    if architecture != "dual_branch_26ch_v1":
        raise ValueError(
            "Checkpoint architecture is not dual_branch_26ch_v1. "
            "Old single-encoder or GBRG checkpoints cannot be used for this RGGB pipeline."
        )
    checkpoint_stride = checkpoint.get("stride", 2) if isinstance(checkpoint, dict) else 2
    model_stride = int(stride if stride is not None else checkpoint_stride)
    base_ch = int(checkpoint.get("base_ch", 16)) if isinstance(checkpoint, dict) else 16
    alignment_mode = str(checkpoint_cfg.get("alignment_mode", "farneback"))
    seq = LocalAlignSequence(source_path, denoised_path, frames, height, width,
                             model_stride, source_black, denoised_black,
                             container_scale, clip, cfa, alignment_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_raw = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(frames, height, width))
    accepted = bool(checkpoint_cfg.get("accepted_against_2dnr"))
    if not accepted:
        out_raw[:] = seq.nr2d[:]
        out_raw.flush()
        manifest = {
            "model": str(model_path), "source": str(source), "denoised": str(denoised),
            "output": str(output_path), "frames": frames, "height": height, "width": width,
            "stride": model_stride, "cfa": cfa.upper(), "architecture": architecture,
            "alignment_mode": alignment_mode,
            "accepted_against_2dnr": False, "output_mode": "exact_2dnr_copy",
        }
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"[local-align infer done] rejected checkpoint copied 2DNR to {output_path}")
        return

    model = LocalAlignResidualUNet(in_ch=26, base_ch=base_ch).to(run_device)
    model.load_state_dict(state)
    model.eval()
    print(f"[local-align infer] device={run_device} frames={frames} stride={model_stride} output={output_path}")
    for current in range(frames):
        if current < 2:
            out_raw[current] = seq.nr2d[current]
            continue
        inp, base_low, _ = _la_input_full(seq, current)
        pred_low = _la_predict_tiled(model, inp, base_low, run_device, tile, overlap)
        delta_low = pred_low - base_low
        # The model may be trained at a downsampled Bayer-plane resolution.
        # Upsample only its residual and add it to exact full-resolution 2DNR.
        base_full = _la_read_frame(seq.nr2d, current, denoised_black, 1.0, clip, 1, cfa)
        delta_full = np.empty_like(base_full)
        for channel in range(4):
            delta_full[channel] = cv2.resize(
                delta_low[channel], (base_full.shape[2], base_full.shape[1]),
                interpolation=cv2.INTER_LINEAR
            )
        delta_codes = np.rint(_la_planes_to_mosaic(delta_full, cfa) * clip).astype(np.int32)
        out_raw[current] = np.clip(
            seq.nr2d[current].astype(np.int32) + delta_codes,
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        if current % 10 == 0 or current == frames - 1:
            print(f"[local-align infer] frame {current + 1}/{frames}")
    out_raw.flush()
    manifest = {
        "model": str(model_path), "source": str(source), "denoised": str(denoised),
        "output": str(output_path), "frames": frames, "height": height, "width": width,
        "stride": model_stride, "cfa": cfa.upper(),
        "architecture": architecture, "output_mode": "2dnr_plus_learned_delta",
        "alignment_mode": alignment_mode,
        "accepted_against_2dnr": checkpoint_cfg.get("accepted_against_2dnr"),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[local-align infer done] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--export_h5_input', type=str, default=None)
    parser.add_argument('--export_h5_output', type=str, default=None)
    parser.add_argument('--export_clean_key', type=str, default='clean')
    parser.add_argument('--export_noisy_key', type=str, default='noisy')
    parser.add_argument('--export_nr2d_key', type=str, default='2dnr')
    parser.add_argument('--export_nr3d_key', type=str, default='3dnr')
    parser.add_argument('--export_noisy_channel', type=int, default=None)
    parser.add_argument('--export_default_scene', type=int, default=1)
    parser.add_argument('--export_default_iso', type=int, default=1600)
    parser.add_argument('--export_noisy_id', type=int, default=1)
    parser.add_argument('--export_clip', type=float, default=4095.0)
    parser.add_argument('--export_cfa', type=str, default='GBRG', choices=['GBRG', 'RGGB', 'BGGR', 'GRBG'])
    parser.add_argument('--export_write_config', action='store_true')
    parser.add_argument('--export_raw_fusion_frames', type=int, default=7)
    parser.add_argument('--export_include_noisy', action='store_true')
    parser.add_argument('--infer_h5_input', type=str, default=None)
    parser.add_argument('--infer_h5_output', type=str, default=None)
    parser.add_argument('--infer_model_path', type=str, default=None)
    parser.add_argument('--infer_model_type', type=str, default='FusionGateNet', choices=['FusionGateNet', 'UnetGateNet', 'CheckpointUnetGateNet'])
    parser.add_argument('--infer_base_ch', type=int, default=32)
    parser.add_argument('--infer_gate_plane', type=str, default='G1', choices=['R', 'G1', 'G2', 'B'])
    parser.add_argument('--infer_cfa', type=str, default='GBRG', choices=['GBRG', 'RGGB', 'BGGR', 'GRBG'])
    parser.add_argument('--infer_nr2d_key', type=str, default='2dnr')
    parser.add_argument('--infer_nr3d_key', type=str, default='3dnr')
    parser.add_argument('--infer_output_key', type=str, default='fusion')
    parser.add_argument('--infer_first_frame_source', type=str, default='2dnr', choices=['2dnr', '3dnr'])
    parser.add_argument('--infer_scale_div', type=float, default=4095.0)
    parser.add_argument('--infer_compression', type=str, default='gzip', choices=['gzip', 'lzf', 'none'])
    parser.add_argument('--infer_device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--infer_no_copy_inputs', action='store_true')
    # New local-alignment RAW U-Net mode.  It does not use 3DNR as a target.
    parser.add_argument('--local_align_source', type=str, default=None)
    parser.add_argument('--local_align_2dnr', type=str, default=None)
    parser.add_argument('--local_align_out', type=str, default=None)
    parser.add_argument('--local_infer_source', type=str, default=None)
    parser.add_argument('--local_infer_2dnr', type=str, default=None)
    parser.add_argument('--local_infer_model', type=str, default=None)
    parser.add_argument('--local_infer_out', type=str, default=None)
    parser.add_argument('--local_infer_stride', type=int, default=None)
    parser.add_argument('--local_infer_tile', type=int, default=128)
    parser.add_argument('--local_infer_overlap', type=int, default=32)
    parser.add_argument('--local_infer_overwrite', action='store_true')
    parser.add_argument('--local_frames', type=int, default=200)
    parser.add_argument('--local_height', type=int, default=1080)
    parser.add_argument('--local_width', type=int, default=1920)
    parser.add_argument('--local_source_black', type=float, default=252.0)
    parser.add_argument('--local_2dnr_black', type=float, default=300.0)
    parser.add_argument('--local_container_scale', type=float, default=16.0)
    parser.add_argument('--local_clip', type=float, default=4095.0)
    parser.add_argument('--local_cfa', type=str, default='RGGB', choices=['GBRG', 'RGGB', 'BGGR', 'GRBG'])
    parser.add_argument('--local_alignment_mode', type=str, default='farneback', choices=['farneback', 'identity'])
    parser.add_argument('--local_stride', type=int, default=2)
    parser.add_argument('--local_patch', type=int, default=64)
    parser.add_argument('--local_steps', type=int, default=80)
    parser.add_argument('--local_base_ch', type=int, default=16)
    parser.add_argument('--local_lr', type=float, default=2e-4)
    parser.add_argument('--local_seed', type=int, default=0)
    parser.add_argument('--local_device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--local_eval_frames', type=int, default=8)
    parser.add_argument('--local_eval_patches', type=int, default=8)
    parser.add_argument('--local_validate_every', type=int, default=20)
    parser.add_argument('--local_min_improvement', type=float, default=2e-5)
    args = parser.parse_args()

    if args.local_infer_source is not None:
        required = [args.local_infer_2dnr, args.local_infer_model, args.local_infer_out]
        if any(value is None for value in required):
            raise ValueError("--local_infer_2dnr, --local_infer_model and --local_infer_out are required with --local_infer_source")
        infer_local_align_unet(
            source=args.local_infer_source,
            denoised=args.local_infer_2dnr,
            model_path=args.local_infer_model,
            output=args.local_infer_out,
            frames=args.local_frames,
            height=args.local_height,
            width=args.local_width,
            source_black=args.local_source_black,
            denoised_black=args.local_2dnr_black,
            container_scale=args.local_container_scale,
            clip=args.local_clip,
            stride=args.local_infer_stride,
            cfa=args.local_cfa,
            tile=args.local_infer_tile,
            overlap=args.local_infer_overlap,
            device=args.local_device,
            overwrite=args.local_infer_overwrite,
        )
        raise SystemExit(0)

    if args.local_align_source is not None:
        if args.local_align_2dnr is None or args.local_align_out is None:
            raise ValueError("--local_align_2dnr and --local_align_out are required with --local_align_source")
        train_local_align_unet(
            source=args.local_align_source,
            denoised=args.local_align_2dnr,
            out=args.local_align_out,
            frames=args.local_frames,
            height=args.local_height,
            width=args.local_width,
            source_black=args.local_source_black,
            denoised_black=args.local_2dnr_black,
            container_scale=args.local_container_scale,
            clip=args.local_clip,
            cfa=args.local_cfa,
            stride=args.local_stride,
            patch=args.local_patch,
            steps=args.local_steps,
            base_ch=args.local_base_ch,
            learning_rate=args.local_lr,
            seed=args.local_seed,
            device=args.local_device,
            eval_frames=args.local_eval_frames,
            eval_patches=args.local_eval_patches,
            validate_every=args.local_validate_every,
            min_improvement=args.local_min_improvement,
            alignment_mode=args.local_alignment_mode,
        )
        raise SystemExit(0)

    if args.infer_h5_input is not None:
        if args.infer_h5_output is None or args.infer_model_path is None:
            raise ValueError("--infer_h5_output and --infer_model_path are required when --infer_h5_input is used")
        infer_fusion_h5(
            input_root=args.infer_h5_input,
            output_root=args.infer_h5_output,
            model_path=args.infer_model_path,
            model_type=args.infer_model_type,
            base_ch=args.infer_base_ch,
            gate_plane=args.infer_gate_plane,
            cfa=args.infer_cfa,
            nr2d_key=args.infer_nr2d_key,
            nr3d_key=args.infer_nr3d_key,
            output_key=args.infer_output_key,
            first_frame_source=args.infer_first_frame_source,
            scale_div=args.infer_scale_div,
            compression=args.infer_compression,
            device=args.infer_device,
            copy_inputs=not args.infer_no_copy_inputs,
        )
        raise SystemExit(0)

    if args.export_h5_input is not None:
        if args.export_h5_output is None:
            raise ValueError("--export_h5_output is required when --export_h5_input is used")
        export_h5_to_raw_fusion_dataset(
            input_root=args.export_h5_input,
            output_root=args.export_h5_output,
            clean_key=args.export_clean_key,
            noisy_key=args.export_noisy_key,
            nr2d_key=args.export_nr2d_key,
            nr3d_key=args.export_nr3d_key,
            noisy_channel=args.export_noisy_channel,
            default_scene=args.export_default_scene,
            default_iso=args.export_default_iso,
            noisy_id=args.export_noisy_id,
            clip=args.export_clip,
            cfa=args.export_cfa,
            write_config=args.export_write_config,
            raw_fusion_frames=args.export_raw_fusion_frames,
            include_noisy=args.export_include_noisy,
        )
        raise SystemExit(0)

    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    # 定义要尝试的组合
    model_types = ["FusionGateNet", "UnetGateNet"]
    gate_planes = ["R", "G1", "B"]
    base_ch = 16
    epochs = 10

    for model_type in model_types:
        for gate_plane in gate_planes:
            # 复制基础配置并覆盖相关字段
            cfg_dict = base_cfg.copy()
            cfg_dict["model_type"] = model_type
            cfg_dict["gate_plane"] = gate_plane
            cfg_dict["base_ch"] = base_ch
            cfg_dict["epochs"] = epochs

            # 实例化 PipelineCFG（注意需确保所有字段都在类中定义）
            CFG = PipelineCFG(**cfg_dict)
            print(f"\n=== 开始训练: {model_type}, gate={gate_plane} ===")
            runner = FullPipeline(CFG)
            runner.run()

# if __name__ == "__main__":
#     main()

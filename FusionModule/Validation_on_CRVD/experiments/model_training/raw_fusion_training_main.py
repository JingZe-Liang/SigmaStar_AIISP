import re
import json
import random
import itertools

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
try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False
    print("Warning: mamba_ssm not installed. VmambaGateNet will not work.")


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
        if not _HAS_MAMBA:
            raise ImportError("mamba_ssm is required for VmambaGateNet. Install with: pip install mamba-ssm causal-conv1d")

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
            "noisy": str(root / self.cfg.noisy_dir),
            "gt": str(root / self.cfg.gt_dir),
            "nr2d": str(root / self.cfg.nr2d_dir),
            "nr3d": str(root / self.cfg.nr3d_dir),
        }

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

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


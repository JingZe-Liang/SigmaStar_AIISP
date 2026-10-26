"""Continuous clean RAW clips from H5, packed as R,G1,G2,B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset

from data.h5_raw import _validate_dn_frame, inspect_scene
from noise_model.raw_io import pack_bayer


@dataclass(frozen=True)
class H5ClipRef:
    scene: str
    shard: int
    start_frame: int
    path: Path


class H5RawClipDataset(Dataset[tuple[np.ndarray, dict[str, int]]]):
    """Sample clips without crossing an H5 shard or scene boundary."""

    def __init__(
        self,
        root: str | Path,
        scenes: Sequence[str],
        clip_length: int = 6,
        patch_size: int = 128,
        samples_per_epoch: int = 4096,
        seed: int = 7,
        key: str = "clean",
        cfa: str = "GBRG",
        white_level: int = 4095,
        deterministic: bool = False,
        shards: Sequence[int] | None = None,
        ref_sampling: str = "sequential",
    ) -> None:
        if clip_length < 2:
            raise ValueError("clip_length must be at least 2")
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        self.root = Path(root)
        self.clip_length = int(clip_length)
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.key = key
        self.cfa = cfa
        self.white_level = int(white_level)
        self.deterministic = bool(deterministic)
        if ref_sampling not in ("sequential", "random", "balanced"):
            raise ValueError("unsupported ref_sampling")
        self.ref_sampling = ref_sampling
        self.refs: list[H5ClipRef] = []
        self.scene_ids = {scene: i for i, scene in enumerate(scenes)}
        for scene in scenes:
            info = inspect_scene(self.root / scene, key=key)
            if info.frames_per_shard < self.clip_length:
                continue
            allowed_shards = range(info.num_shards) if shards is None else tuple(int(value) for value in shards)
            for shard in allowed_shards:
                if shard < 0 or shard >= info.num_shards:
                    raise ValueError(f"shard {shard} is outside scene {scene}")
                path = info.path / f"shard_{shard}.h5"
                for start in range(info.frames_per_shard - self.clip_length + 1):
                    self.refs.append(H5ClipRef(scene, shard, start, path))
        if not self.refs:
            raise ValueError("no clips fit the requested length")
        groups: dict[tuple[str, int], list[H5ClipRef]] = {}
        for ref in self.refs:
            groups.setdefault((ref.scene, ref.shard), []).append(ref)
        self.ref_groups = list(groups.values())
        with h5py.File(self.refs[0].path, "r") as handle:
            shape = tuple(int(x) for x in handle[key].shape[-2:])
        if shape[0] % 2 or shape[1] % 2:
            raise ValueError("RAW dimensions must be even")
        packed_shape = (4, shape[0] // 2, shape[1] // 2)
        if patch_size > packed_shape[1] or patch_size > packed_shape[2]:
            raise ValueError(f"patch_size {patch_size} exceeds packed frame {packed_shape}")
        self.packed_shape = packed_shape
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))

    def _read_clip(self, ref: H5ClipRef) -> np.ndarray:
        with h5py.File(ref.path, "r") as handle:
            frames = np.asarray(handle[self.key][ref.start_frame:ref.start_frame + self.clip_length])
        checked = np.stack([_validate_dn_frame(frame, self.white_level) for frame in frames])
        return pack_bayer(checked, cfa=self.cfa)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, int]]:
        rng = self._rng(index)
        if self.ref_sampling == "random":
            ref = self.refs[int(rng.integers(len(self.refs)))]
        elif self.ref_sampling == "balanced":
            group = self.ref_groups[int(index) % len(self.ref_groups)]
            ref = group[int(rng.integers(len(group)))]
        else:
            ref = self.refs[(self.epoch * self.samples_per_epoch + int(index)) % len(self.refs)]
        if self.deterministic:
            top = (int(index) * 37) % (self.packed_shape[1] - self.patch_size + 1)
            left = (int(index) * 53) % (self.packed_shape[2] - self.patch_size + 1)
        else:
            top = int(rng.integers(0, self.packed_shape[1] - self.patch_size + 1))
            left = int(rng.integers(0, self.packed_shape[2] - self.patch_size + 1))
        clip = self._read_clip(ref)[:, :, top:top + self.patch_size, left:left + self.patch_size]
        clip = clip / float(self.white_level)
        return clip.astype(np.float32, copy=False), {
            "frame": ref.start_frame,
            "shard": ref.shard,
            "top": top,
            "left": left,
            "scene_id": self.scene_ids[ref.scene],
        }

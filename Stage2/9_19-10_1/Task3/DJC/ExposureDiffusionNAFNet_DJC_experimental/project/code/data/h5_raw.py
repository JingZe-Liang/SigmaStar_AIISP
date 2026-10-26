"""Scene-level H5 access for the clean RAW training material.

The H5 files contain effective 12-bit DN values, rather than the left-aligned
uint16 MIS20S1 container. The project ProRes conversion notes identify the
Apple clean material as GBRG; it is decoded as GBRG here and then packed as
R,G1,G2,B. MIS20S1 .raw remains GRBG and continues to use
``noise_model.raw_io``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from noise_model.raw_io import pack_bayer


@dataclass(frozen=True)
class H5FrameRef:
    scene: str
    shard: int
    frame: int
    path: Path


@dataclass(frozen=True)
class H5SceneInfo:
    scene: str
    path: Path
    num_shards: int
    frames_per_shard: int
    total_frames: int
    height: int
    width: int


def _validate_dn_frame(frame: np.ndarray, white_level: int = 4095) -> np.ndarray:
    values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"H5 RAW frame must be [H,W], got {values.shape}")
    if values.dtype.kind not in "ui":
        raise ValueError(f"H5 RAW frame must be integer DN, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("H5 RAW frame contains non-finite values")
    if int(values.min()) < 0 or int(values.max()) > white_level:
        raise ValueError(
            f"H5 RAW DN outside [0,{white_level}]: "
            f"{values.min()}..{values.max()}"
        )
    return values.astype(np.float32, copy=False)


def inspect_scene(scene_dir: str | Path, key: str = "clean") -> H5SceneInfo:
    """Validate one scene's metadata and every shard's requested dataset."""
    root = Path(scene_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    scene = root.name
    expected_shards = int(metadata["num_shards"])
    expected_per_shard = int(metadata["frames_per_shard"])
    expected_total = int(metadata["total_frames"])
    frame_shape = tuple(int(x) for x in metadata["frame_shape"])
    if len(frame_shape) != 2 or any(x <= 0 for x in frame_shape):
        raise ValueError(f"invalid frame_shape in {metadata_path}")
    total = 0
    for shard_id in range(expected_shards):
        path = root / f"shard_{shard_id}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"{key!r} missing from {path}")
            dataset = handle[key]
            expected_shape = (expected_per_shard, *frame_shape)
            if tuple(dataset.shape) != expected_shape:
                raise ValueError(
                    f"{path}:{key} shape {dataset.shape}, expected {expected_shape}"
                )
            if dataset.dtype.kind not in "ui":
                raise ValueError(f"{path}:{key} must contain integer DN")
            # Reading two small sentinels catches truncated/corrupt gzip shards.
            _validate_dn_frame(dataset[0])
            _validate_dn_frame(dataset[-1])
        total += expected_per_shard
    if total != expected_total:
        raise ValueError(f"{scene} metadata total_frames={expected_total}, found {total}")
    return H5SceneInfo(
        scene=scene,
        path=root,
        num_shards=expected_shards,
        frames_per_shard=expected_per_shard,
        total_frames=expected_total,
        height=frame_shape[0],
        width=frame_shape[1],
    )


def inspect_root(root: str | Path, scenes: Sequence[str] | None = None) -> list[H5SceneInfo]:
    base = Path(root)
    names = list(scenes) if scenes is not None else sorted(
        path.name for path in base.glob("scene_*") if path.is_dir()
    )
    if not names:
        raise ValueError(f"no scene_* directories found below {base}")
    return [inspect_scene(base / name) for name in names]


def frame_refs(root: str | Path, scenes: Iterable[str]) -> list[H5FrameRef]:
    refs: list[H5FrameRef] = []
    for scene in scenes:
        info = inspect_scene(Path(root) / scene)
        for shard in range(info.num_shards):
            path = info.path / f"shard_{shard}.h5"
            for frame in range(info.frames_per_shard):
                refs.append(H5FrameRef(scene, shard, frame, path))
    return refs


def read_frame(ref: H5FrameRef, key: str = "clean", cfa: str = "GBRG") -> np.ndarray:
    with h5py.File(ref.path, "r") as handle:
        frame = _validate_dn_frame(handle[key][ref.frame])
    packed = pack_bayer(frame, cfa=cfa)[0]
    return packed


class H5RawPatchDataset:
    """Random packed-RAW patches sampled without crossing scene boundaries."""

    def __init__(
        self,
        root: str | Path,
        scenes: Sequence[str],
        patch_size: int = 128,
        samples_per_epoch: int = 4096,
        seed: int = 7,
        key: str = "clean",
        cfa: str = "GBRG",
        white_level: int = 4095,
        deterministic: bool = False,
        patches_per_frame: int = 1,
    ) -> None:
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        self.root = Path(root)
        self.scenes = tuple(scenes)
        self.patch_size = patch_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.key = key
        self.cfa = cfa
        self.white_level = int(white_level)
        self.deterministic = deterministic
        self.patches_per_frame = int(patches_per_frame)
        if self.patches_per_frame <= 0:
            raise ValueError("patches_per_frame must be positive")
        self.refs = frame_refs(self.root, self.scenes)
        if not self.refs:
            raise ValueError("dataset has no frames")
        first = read_frame(self.refs[0], key=key, cfa=cfa)
        if first.shape[-2] < patch_size or first.shape[-1] < patch_size:
            raise ValueError(f"patch_size {patch_size} exceeds packed frame {first.shape}")
        self.packed_shape = tuple(int(x) for x in first.shape)
        self.epoch = 0
        self._cached_ref: H5FrameRef | None = None
        self._cached_frame: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, int | str]]:
        rng = self._rng(index)
        frames_per_epoch = int(np.ceil(self.samples_per_epoch / self.patches_per_frame))
        frame_slot = self.epoch * frames_per_epoch + int(index) // self.patches_per_frame
        ref = self.refs[frame_slot % len(self.refs)]
        if self.deterministic:
            top = ((int(index) * 37) % (self.packed_shape[1] - self.patch_size + 1))
            left = ((int(index) * 53) % (self.packed_shape[2] - self.patch_size + 1))
        else:
            top = int(rng.integers(0, self.packed_shape[1] - self.patch_size + 1))
            left = int(rng.integers(0, self.packed_shape[2] - self.patch_size + 1))
        if self._cached_ref != ref:
            self._cached_frame = read_frame(ref, key=self.key, cfa=self.cfa)
            self._cached_ref = ref
        assert self._cached_frame is not None
        frame = self._cached_frame
        patch = frame[:, top : top + self.patch_size, left : left + self.patch_size]
        patch = patch / float(self.white_level)
        return patch.astype(np.float32, copy=False), {
            "scene": ref.scene,
            "shard": ref.shard,
            "frame": ref.frame,
            "top": top,
            "left": left,
        }

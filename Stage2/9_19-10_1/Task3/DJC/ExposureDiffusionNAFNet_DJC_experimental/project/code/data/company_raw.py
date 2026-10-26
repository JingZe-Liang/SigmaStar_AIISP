"""Patch dataset for a repeated company MIS20S1 RAW stream.

The stream remains in its original left-aligned uint16 container on disk.
Every access goes through ``raw_io.read_frames`` so a container value can
never silently enter a model as a normalized signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from noise_model.raw_io import RawSpec, count_frames, pack_bayer, read_frames


class CompanyRawPatchDataset(Dataset[tuple[np.ndarray, dict[str, int]]]):
    """Deterministic patches from one company RAW stream in dn12 units."""

    def __init__(
        self,
        path: str | Path,
        spec: RawSpec,
        patch_size: int,
        samples_per_epoch: int,
        seed: int = 7,
        deterministic: bool = False,
        start_frame: int = 0,
        stop_frame: int | None = None,
    ) -> None:
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        self.path = Path(path)
        self.spec = spec
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.frame_count = count_frames(self.path, self.spec)
        self.start_frame = int(start_frame)
        self.stop_frame = self.frame_count if stop_frame is None else int(stop_frame)
        if self.frame_count < 2:
            raise ValueError("company self-supervision requires a repeated RAW stream")
        if not 0 <= self.start_frame < self.stop_frame <= self.frame_count:
            raise ValueError("invalid company RAW frame range")
        packed_height, packed_width = self.spec.height // 2, self.spec.width // 2
        if self.patch_size > packed_height or self.patch_size > packed_width:
            raise ValueError("patch_size exceeds packed RAW dimensions")
        self._epoch = 0
        self._cached_frame_index: int | None = None
        self._cached_frame: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._cached_frame_index = None
        self._cached_frame = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self._epoch * 1_000_003 + int(index))

    def _load_frame(self, frame_index: int) -> np.ndarray:
        if self._cached_frame_index != frame_index:
            dn12 = read_frames(self.path, self.spec, max_frames=1, start_frame=frame_index)
            self._cached_frame = pack_bayer(dn12, cfa=self.spec.cfa)[0]
            self._cached_frame_index = frame_index
        assert self._cached_frame is not None
        return self._cached_frame

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, int]]:
        rng = self._rng(index)
        frame_index = self.start_frame + (self._epoch * self.samples_per_epoch + int(index)) % (self.stop_frame - self.start_frame)
        frame = self._load_frame(frame_index)
        max_top = frame.shape[1] - self.patch_size
        max_left = frame.shape[2] - self.patch_size
        if self.deterministic:
            top = (int(index) * 37) % (max_top + 1)
            left = (int(index) * 53) % (max_left + 1)
        else:
            top = int(rng.integers(0, max_top + 1))
            left = int(rng.integers(0, max_left + 1))
        patch = frame[:, top : top + self.patch_size, left : left + self.patch_size]
        return patch.astype(np.float32, copy=False), {
            "frame": frame_index,
            "top": top,
            "left": left,
        }


class CompanyRawClipDataset(CompanyRawPatchDataset):
    """Seven original frames, DN12, never crossing a role's frame boundaries."""

    def __init__(self, *args, fixed_crop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_crop = None if fixed_crop is None else tuple(map(int, fixed_crop))
        if self.fixed_crop is not None:
            top, left = self.fixed_crop
            if not (0 <= top <= self.spec.height//2-self.patch_size and
                    0 <= left <= self.spec.width//2-self.patch_size):
                raise ValueError("fixed packed crop exceeds RAW dimensions")

    def __getitem__(self, index):
        if self.stop_frame - self.start_frame < 7:
            raise ValueError("seven frames do not fit this role")
        rng = self._rng(index)
        start = int(rng.integers(self.start_frame, self.stop_frame - 6))
        frames = read_frames(self.path, self.spec, max_frames=7, start_frame=start)
        clip = pack_bayer(frames, self.spec.cfa)
        if self.fixed_crop is None:
            top = int(rng.integers(0, clip.shape[-2] - self.patch_size + 1))
            left = int(rng.integers(0, clip.shape[-1] - self.patch_size + 1))
        else:
            top, left = self.fixed_crop
        return clip[:, :, top:top+self.patch_size, left:left+self.patch_size], {
            "frame_start": start, "frame_end": start+6, "top": top, "left": left}

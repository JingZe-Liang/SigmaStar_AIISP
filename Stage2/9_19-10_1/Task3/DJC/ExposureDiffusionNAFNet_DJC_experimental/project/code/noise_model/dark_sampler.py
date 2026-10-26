"""Sampling real dark-frame residuals while preserving spatial structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .raw_io import RawSpec, pack_bayer, read_frames


class DarkResidualSampler:
    """Sample signal-independent residual patches from real dark frames."""

    def __init__(
        self,
        path: str | Path,
        spec: RawSpec,
        max_frames: int | None = None,
        reference: np.ndarray | None = None,
    ):
        frames = read_frames(path, spec, max_frames=max_frames)
        if reference is None:
            reference = frames.mean(axis=0)
        if reference.shape != frames.shape[1:]:
            raise ValueError("dark reference shape does not match dark frames")
        self.residual = frames - reference[None]
        self.spec = spec

    def sample_raw_patch(
        self, height: int, width: int, frame_index: int = 0, top: int = 0, left: int = 0
    ) -> np.ndarray:
        if height % 2 or width % 2:
            raise ValueError("dark patch dimensions must be even")
        frame_index %= len(self.residual)
        patch = self.residual[frame_index, top : top + height, left : left + width]
        if patch.shape != (height, width):
            raise ValueError("dark patch exceeds the available frame")
        return patch.astype(np.float32)

    def sample_packed_patch(
        self, packed_height: int, packed_width: int, frame_index: int = 0
    ) -> np.ndarray:
        raw = self.sample_raw_patch(
            packed_height * 2, packed_width * 2, frame_index=frame_index
        )
        return pack_bayer(raw, self.spec.cfa)[0]

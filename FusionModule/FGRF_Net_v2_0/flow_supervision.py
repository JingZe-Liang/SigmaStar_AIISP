"""RAFT utilities used only to construct FGRF-Net v2.0 training supervision."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_FLOW_INDEX = re.compile(r"^(\d+)")


class FlowSequence:
    """Indexed RAFT flow files. Flow is never exposed to the fusion model."""

    def __init__(
        self,
        directory: str | Path,
        target_hw: tuple[int, int],
        cache_resized: bool = False,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.target_hw = target_hw
        self.files: dict[int, Path] = {}
        for path in self.directory.glob("*.npy"):
            match = _FLOW_INDEX.match(path.stem)
            if match is not None:
                self.files[int(match.group(1))] = path
        if not self.files:
            raise FileNotFoundError(f"No indexed .npy flow files in {self.directory}")
        self.cache = {index: self._read(index) for index in sorted(self.files)} if cache_resized else None

    @property
    def pair_count(self) -> int:
        return len(self.files)

    def _read(self, index: int) -> torch.Tensor:
        if index not in self.files:
            raise IndexError(f"No flow pair {index} in {self.directory}")
        array = np.load(self.files[index], allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 2 or not np.isfinite(array).all():
            raise ValueError(f"Invalid flow file: {self.files[index]}")
        flow = torch.from_numpy(array.astype(np.float32, copy=False)).permute(2, 0, 1)
        return resize_flow(flow, self.target_hw).contiguous()

    def load(self, index: int, crop: tuple[int, int, int] | None = None) -> torch.Tensor:
        flow = self.cache[index] if self.cache is not None else self._read(index)
        if crop is None:
            return flow
        top, left, size = crop
        return flow[:, top:top + size, left:left + size]


def resize_flow(flow: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected [2, H, W], got {tuple(flow.shape)}")
    old_height, old_width = flow.shape[-2:]
    height, width = target_hw
    if (old_height, old_width) == (height, width):
        return flow
    resized = F.interpolate(flow.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=True)[0]
    resized[0] *= float(width) / old_width
    resized[1] *= float(height) / old_height
    return resized


def _sampling_grid(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected [N, 2, H, W], got {tuple(flow.shape)}")
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0) + flow[:, 0]
    y = y.unsqueeze(0) + flow[:, 1]
    valid = ((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)).unsqueeze(1)
    grid = torch.stack((2.0 * x / max(width - 1, 1) - 1.0, 2.0 * y / max(height - 1, 1) - 1.0), dim=-1)
    return grid, valid.to(flow.dtype)


def warp(source: torch.Tensor, center_to_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample source at center coordinates plus a center-to-source flow."""
    grid, valid = _sampling_grid(center_to_source)
    sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    finite = torch.isfinite(sampled).all(dim=1, keepdim=True).to(sampled.dtype)
    return sampled, valid * finite


def compose(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose p->q and q->r flows into p->r, including an in-bounds mask."""
    sampled_second, valid = warp(second.unsqueeze(0), first.unsqueeze(0))
    return first + sampled_second[0], valid[0]


def compose_offset(
    forward: FlowSequence,
    backward: FlowSequence,
    index: int,
    offset: int,
    crop: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose adjacent RAFT pairs for center frame ``index`` to ``index+offset``."""
    if offset == 0:
        size = crop[2]
        return torch.zeros(2, size, size), torch.ones(1, size, size)
    if offset > 0:
        pair_indices = range(index, index + offset)
        sequence = forward
    else:
        pair_indices = range(index - 1, index + offset - 1, -1)
        sequence = backward
    total: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    for pair_index in pair_indices:
        current = sequence.load(pair_index, crop)
        if total is None:
            total = current
            valid = torch.ones_like(current[:1])
        else:
            total, current_valid = compose(total, current)
            assert valid is not None
            valid = valid * current_valid
    assert total is not None and valid is not None
    return total, valid


def pair_confidence(
    center: torch.Tensor,
    neighbor: torch.Tensor,
    center_to_neighbor: torch.Tensor,
    neighbor_to_center: torch.Tensor,
    tau_fb: float,
    tau_rgb: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward/backward and photometric confidence for one composed pair."""
    forward = center_to_neighbor.unsqueeze(0)
    warped_neighbor, valid = warp(neighbor.unsqueeze(0), forward)
    warped_reverse, reverse_valid = warp(neighbor_to_center.unsqueeze(0), forward)
    fb_error = torch.linalg.vector_norm(forward + warped_reverse, dim=1, keepdim=True)
    photo_error = (center.unsqueeze(0) - warped_neighbor).abs().mean(dim=1, keepdim=True)
    confidence = valid * reverse_valid
    confidence = confidence * torch.exp(-fb_error / max(float(tau_fb), 1e-6))
    confidence = confidence * torch.exp(-photo_error / max(float(tau_rgb), 1e-6))
    return confidence.clamp(0.0, 1.0), valid

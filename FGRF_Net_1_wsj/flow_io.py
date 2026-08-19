"""Flow loading, resizing, warping, and forward-backward reliability."""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F


_FLOW_INDEX = re.compile(r"^(\d+)")


class FlowSequence:
    """Load one scene's indexed .npy flow files from a directory."""

    def __init__(
        self,
        directory: str | Path,
        target_hw: tuple[int, int] | None = None,
        cache_resized: bool = False,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        files = list(self.directory.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy flow files found in {self.directory}")
        indexed: dict[int, Path] = {}
        for path in files:
            match = _FLOW_INDEX.match(path.stem)
            if match:
                indexed[int(match.group(1))] = path
        if not indexed:
            raise ValueError(f"Flow file names must start with an integer: {self.directory}")
        self._files = indexed
        self.target_hw = target_hw
        self._cache: list[torch.Tensor] | None = None
        if cache_resized:
            if target_hw is None:
                raise ValueError("target_hw is required when cache_resized=True")
            self._cache = [self._load_file(index) for index in sorted(self._files)]

    @property
    def pair_count(self) -> int:
        return len(self._files)

    def load(self, pair_index: int, device: torch.device | str = "cpu") -> torch.Tensor:
        if pair_index not in self._files:
            raise IndexError(f"No flow pair {pair_index} in {self.directory}")
        if self._cache is not None:
            return self._cache[pair_index].to(device)
        return self._load_file(pair_index).to(device)

    def _load_file(self, pair_index: int) -> torch.Tensor:
        import numpy as np

        flow = np.load(self._files[pair_index], allow_pickle=False)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError(f"Expected [H, W, 2] flow, got {flow.shape}")
        if not np.isfinite(flow).all():
            raise ValueError(f"Flow contains NaN/Inf: {self._files[pair_index]}")
        tensor = torch.from_numpy(flow.astype("float32", copy=False)).permute(2, 0, 1)
        if self.target_hw is not None and tuple(tensor.shape[-2:]) != tuple(self.target_hw):
            tensor = resize_flow(tensor, self.target_hw)
        return tensor.contiguous()


def resize_flow(flow: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize [2, H, W] flow and scale displacement to the target coordinates."""
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
        squeeze = True
    elif flow.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected [2,H,W] or [N,2,H,W], got {tuple(flow.shape)}")
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = target_hw
    output = F.interpolate(flow, size=target_hw, mode="bilinear", align_corners=True)
    output[:, 0] *= float(new_w) / float(old_w)
    output[:, 1] *= float(new_h) / float(old_h)
    return output.squeeze(0) if squeeze else output


def _base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def flow_grid(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a grid for sampling a neighbor at p + flow(p)."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected [N,2,H,W] flow, got {tuple(flow.shape)}")
    batch, _, height, width = flow.shape
    coordinates = _base_grid(batch, height, width, flow.device, flow.dtype) + flow
    valid = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] <= width - 1)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] <= height - 1)
    ).unsqueeze(1)
    normal_x = 2.0 * coordinates[:, 0] / max(width - 1, 1) - 1.0
    normal_y = 2.0 * coordinates[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((normal_x, normal_y), dim=-1)
    return grid, valid.to(flow.dtype)


def warp_neighbor(source: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a neighbor image at center coordinates plus center-to-neighbor flow."""
    if source.ndim != 4 or flow.ndim != 4:
        raise ValueError("source and flow must be batched tensors")
    grid, valid = flow_grid(flow)
    warped = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    finite = torch.isfinite(warped).all(dim=1, keepdim=True).to(warped.dtype)
    return warped, valid * finite


def pair_confidence(
    center_base: torch.Tensor,
    neighbor_base: torch.Tensor,
    center_to_neighbor: torch.Tensor,
    neighbor_to_center: torch.Tensor,
    tau_fb: float = 1.5,
    tau_rgb: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return confidence and valid mask from flow cycle and photometric errors."""
    warped_neighbor, valid = warp_neighbor(neighbor_base, center_to_neighbor)
    warped_reverse, reverse_valid = warp_neighbor(neighbor_to_center, center_to_neighbor)
    fb_error = torch.linalg.vector_norm(center_to_neighbor + warped_reverse, dim=1, keepdim=True)
    rgb_error = (center_base - warped_neighbor).abs().mean(dim=1, keepdim=True)
    confidence = valid * reverse_valid
    confidence = confidence * torch.exp(-fb_error / max(tau_fb, 1e-6))
    confidence = confidence * torch.exp(-rgb_error / max(tau_rgb, 1e-6))
    return confidence.clamp(0.0, 1.0), valid


def flow_features(
    flow_prev: torch.Tensor,
    flow_next: torch.Tensor,
    confidence_prev: torch.Tensor,
    confidence_next: torch.Tensor,
    flow_scale: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized flow input channels and a continuous motion map."""
    motion = 1.0 - torch.exp(
        -torch.maximum(
            torch.linalg.vector_norm(flow_prev, dim=1, keepdim=True),
            torch.linalg.vector_norm(flow_next, dim=1, keepdim=True),
        )
        / max(flow_scale, 1e-6)
    )
    motion = motion.clamp(0.0, 1.0)
    flow_input = torch.cat(
        (
            flow_prev / max(flow_scale, 1e-6),
            flow_next / max(flow_scale, 1e-6),
        ),
        dim=1,
    )
    return flow_input, motion

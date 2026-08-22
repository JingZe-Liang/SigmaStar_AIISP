"""Training data with RAFT-derived supervision but no flow model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from flow_supervision import FlowSequence, compose_offset, pair_confidence, warp
from raw_io import RawSequence, read_packed_normalized


@dataclass(frozen=True)
class RawSpec:
    sequence: RawSequence
    black_level: float
    white_level: float
    right_shift: int


def _raw_spec(config: dict[str, Any], name: str) -> RawSpec:
    value = config[name]
    return RawSpec(
        sequence=RawSequence(value["path"], int(config["height"]), int(config["width"]), value.get("dtype", "uint16")),
        black_level=float(value["black_level"]),
        white_level=float(value["white_level"]),
        right_shift=int(value.get("right_shift", 0)),
    )


class TextureFusionDataset(Dataset):
    """Random patches with flow-derived targets; returned model inputs are only N/D2/D3."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        samples_per_epoch: int,
        training: bool,
        seed: int,
        flow_sequences: tuple[FlowSequence, FlowSequence] | None = None,
    ) -> None:
        self.config = config
        self.training = training
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.height = int(config["height"])
        self.width = int(config["width"])
        self.packed_hw = (self.height // 2, self.width // 2)
        self.noisy = _raw_spec(config, "noisy")
        self.base = _raw_spec(config, "base")
        self.temporal = _raw_spec(config, "temporal")
        counts = {self.noisy.sequence.frame_count, self.base.sequence.frame_count, self.temporal.sequence.frame_count}
        if len(counts) != 1:
            raise ValueError(f"noisy/2DNR/3DNR frame count mismatch: {counts}")
        self.frame_count = counts.pop()

        supervision = config["supervision"]
        self.radius = int(supervision.get("temporal_radius", 3))
        self.proxy_min_observations = int(supervision.get("proxy_min_observations", 5))
        self.crop_size = int(config.get("crop_size_packed", 256))
        self.flow_halo = int(supervision.get("flow_halo_packed", 16))
        if self.crop_size > min(self.packed_hw):
            raise ValueError(f"crop_size_packed exceeds packed frame: {self.crop_size}")
        if self.flow_halo < 0 or self.crop_size + 2 * self.flow_halo > min(self.packed_hw):
            raise ValueError("flow_halo_packed leaves no valid context crop")
        if self.radius < 1 or 2 * self.radius + 1 < self.proxy_min_observations:
            raise ValueError("temporal radius cannot produce the requested proxy observations")
        self.indices = list(range(self.radius, self.frame_count - self.radius))
        if not self.indices:
            raise ValueError("No valid center frames for the configured temporal radius")

        flow = config["flow"]
        if flow_sequences is None:
            cache = bool(supervision.get("cache_flows", True)) and training
            self.forward = FlowSequence(flow["forward_dir"], self.packed_hw, cache_resized=cache)
            self.backward = FlowSequence(flow["backward_dir"], self.packed_hw, cache_resized=cache)
        else:
            self.forward, self.backward = flow_sequences
        if self.forward.pair_count < self.frame_count - 1 or self.backward.pair_count < self.frame_count - 1:
            raise ValueError(
                f"Need {self.frame_count - 1} forward/backward pairs, got "
                f"{self.forward.pair_count}/{self.backward.pair_count}"
            )
        self.tau_fb = float(supervision.get("tau_fb_pixels", 1.5))
        self.tau_rgb = float(supervision.get("tau_rgb", 0.05))
        self.static_motion_threshold = float(supervision.get("static_motion_threshold_pixels", 0.75))
        self.static_confidence_threshold = float(supervision.get("static_confidence_threshold", 0.30))
        self.static_erosion_kernel = int(supervision.get("static_erosion_kernel", 3))
        self.dark_signal_threshold = float(supervision.get("dark_signal_threshold", 0.005))
        self.saturation_threshold = float(supervision.get("saturation_threshold", 0.995))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, item: int) -> np.random.Generator:
        epoch = self.epoch if self.training else 0
        return np.random.default_rng(self.seed + epoch * self.samples_per_epoch + item)

    def _choose_center_and_crop(self, item: int) -> tuple[int, tuple[int, int, int]]:
        rng = self._rng(item)
        frame_index = self.indices[int(rng.integers(0, len(self.indices)))]
        min_top = self.flow_halo
        min_left = self.flow_halo
        max_top = self.packed_hw[0] - self.crop_size - self.flow_halo
        max_left = self.packed_hw[1] - self.crop_size - self.flow_halo
        top = int(rng.integers(min_top, max_top + 1))
        left = int(rng.integers(min_left, max_left + 1))
        return frame_index, (top, left, self.crop_size)

    def _context_crop(self, crop: tuple[int, int, int]) -> tuple[int, int, int]:
        top, left, size = crop
        return top - self.flow_halo, left - self.flow_halo, size + 2 * self.flow_halo

    def _core(self, value: torch.Tensor) -> torch.Tensor:
        if self.flow_halo == 0:
            return value
        start = self.flow_halo
        stop = start + self.crop_size
        return value[..., start:stop, start:stop]

    @staticmethod
    def _raw_crop(crop: tuple[int, int, int]) -> tuple[int, int, int, int]:
        top, left, size = crop
        return 2 * top, 2 * (top + size), 2 * left, 2 * (left + size)

    def _read(self, spec: RawSpec, index: int, crop: tuple[int, int, int]) -> torch.Tensor:
        array = read_packed_normalized(
            spec.sequence,
            index,
            spec.black_level,
            spec.white_level,
            spec.right_shift,
            crop=self._raw_crop(crop),
        )
        return torch.from_numpy(np.ascontiguousarray(array))

    def _static_erosion(self, mask: torch.Tensor) -> torch.Tensor:
        if self.static_erosion_kernel <= 1:
            return mask
        if self.static_erosion_kernel % 2 == 0:
            raise ValueError("static_erosion_kernel must be odd")
        padding = self.static_erosion_kernel // 2
        eroded = F.avg_pool2d(mask.unsqueeze(0), self.static_erosion_kernel, stride=1, padding=padding)
        return (eroded >= 0.999).to(mask.dtype).squeeze(0)

    def _proxy_and_masks(
        self,
        index: int,
        context_crop: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_center = self._read(self.noisy, index, context_crop)
        base_center = self._read(self.base, index, context_crop)
        values = [noisy_center]
        reliabilities = [torch.ones_like(noisy_center[:1])]
        motions = [torch.zeros_like(noisy_center[:1])]
        for offset in range(-self.radius, self.radius + 1):
            if offset == 0:
                continue
            neighbor_index = index + offset
            noisy_neighbor = self._read(self.noisy, neighbor_index, context_crop)
            base_neighbor = self._read(self.base, neighbor_index, context_crop)
            center_to_neighbor, forward_path_valid = compose_offset(
                self.forward, self.backward, index, offset, context_crop
            )
            neighbor_to_center, reverse_path_valid = compose_offset(
                self.forward, self.backward, neighbor_index, -offset, context_crop
            )
            warped_noisy, sample_valid = warp(noisy_neighbor.unsqueeze(0), center_to_neighbor.unsqueeze(0))
            confidence, _ = pair_confidence(
                base_center,
                base_neighbor,
                center_to_neighbor,
                neighbor_to_center,
                self.tau_fb,
                self.tau_rgb,
            )
            reliable = (
                forward_path_valid
                * reverse_path_valid
                * sample_valid[0]
                * (confidence[0] >= self.static_confidence_threshold).to(torch.float32)
            )
            values.append(warped_noisy[0])
            reliabilities.append(reliable)
            motions.append(torch.linalg.vector_norm(center_to_neighbor, dim=0, keepdim=True))

        stacked_values = torch.stack(values, dim=0)
        stacked_reliable = torch.stack(reliabilities, dim=0)
        count = stacked_reliable.sum(dim=0)
        safe_min = torch.where(stacked_reliable.bool(), stacked_values, torch.full_like(stacked_values, float("inf"))).amin(dim=0)
        safe_max = torch.where(stacked_reliable.bool(), stacked_values, torch.full_like(stacked_values, float("-inf"))).amax(dim=0)
        proxy = (stacked_values * stacked_reliable).sum(dim=0)
        proxy = (proxy - safe_min - safe_max) / (count - 2.0).clamp_min(1.0)
        proxy_available = count >= float(self.proxy_min_observations)
        proxy = torch.where(proxy_available.expand_as(proxy), proxy, noisy_center)

        max_motion = torch.stack(motions, dim=0).amax(dim=0)
        valid_signal = (
            (noisy_center.mean(dim=0, keepdim=True) > self.dark_signal_threshold)
            & (noisy_center.amax(dim=0, keepdim=True) < self.saturation_threshold)
        )
        static = (
            proxy_available
            & (max_motion <= self.static_motion_threshold)
            & valid_signal
        ).to(torch.float32)
        static = self._static_erosion(static)
        motion = (valid_signal.to(torch.float32) * (1.0 - static)).clamp(0.0, 1.0)
        return (
            self._core(noisy_center),
            self._core(base_center),
            self._core(proxy),
            self._core(static),
            self._core(motion),
        )

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        index, crop = self._choose_center_and_crop(item)
        context_crop = self._context_crop(crop)
        noisy, base, proxy, static, motion = self._proxy_and_masks(index, context_crop)
        temporal = self._core(self._read(self.temporal, index, context_crop))
        # Only these three tensors are consumed by TextureGateNet.forward.
        return {
            "frame_index": index,
            "noisy": noisy,
            "base": base,
            "temporal": temporal,
            "proxy": proxy,
            "static_mask": static,
            "motion_mask": motion,
        }


def build_dataset(
    config: dict[str, Any],
    *,
    samples_per_epoch: int,
    training: bool,
    seed: int,
    flow_sequences: tuple[FlowSequence, FlowSequence] | None = None,
) -> TextureFusionDataset:
    return TextureFusionDataset(
        config,
        samples_per_epoch=samples_per_epoch,
        training=training,
        seed=seed,
        flow_sequences=flow_sequences,
    )

"""Self-supervised frame dataset for FGRF-Net."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from flow_io import FlowSequence, pair_confidence, resize_flow
from raw_io import RawSequence, read_packed_normalized, validate_same_sequence_length


def _spec(config: dict[str, Any], name: str) -> tuple[RawSequence, float, float, int]:
    value = config[name]
    sequence = RawSequence(
        value["path"],
        height=config["height"],
        width=config["width"],
        dtype=value.get("dtype", "uint16"),
    )
    return (
        sequence,
        float(value.get("black_level", config.get("denoised_black_level", 300.0))),
        float(value.get("white_level", config.get("white_level", 4095.0))),
        int(value.get("right_shift", 0)),
    )


def hard_static_mask(
    flow_previous: torch.Tensor,
    flow_next: torch.Tensor,
    confidence_previous: torch.Tensor,
    confidence_next: torch.Tensor,
    motion_threshold_pixels: float,
    confidence_threshold: float,
    erosion_kernel: int,
) -> torch.Tensor:
    """Return a binary mask where residual injection is allowed.

    A pixel is static only when its packed-resolution flow magnitude is below
    the threshold and its available forward/backward pair is reliable. A
    small binary erosion removes a ring around moving boundaries.
    """
    magnitudes = torch.maximum(
        torch.linalg.vector_norm(flow_previous, dim=0, keepdim=True),
        torch.linalg.vector_norm(flow_next, dim=0, keepdim=True),
    )
    both = (confidence_previous > 0) & (confidence_next > 0)
    pair_confidence = torch.where(
        both,
        torch.minimum(confidence_previous, confidence_next),
        torch.maximum(confidence_previous, confidence_next),
    )
    mask = ((magnitudes <= motion_threshold_pixels) & (pair_confidence >= confidence_threshold)).to(torch.float32)
    if erosion_kernel > 1:
        if erosion_kernel % 2 == 0:
            raise ValueError("static_erosion_kernel must be odd")
        padding = erosion_kernel // 2
        mask = (F.avg_pool2d(mask.unsqueeze(0), erosion_kernel, stride=1, padding=padding) >= 0.999).to(mask.dtype).squeeze(0)
    return mask


class FusionDataset(Dataset):
    """Return packed RAW inputs and flow reliability for one scene."""

    def __init__(self, config: dict[str, Any], training: bool = True) -> None:
        self.config = config
        self.training = training
        self.height = int(config["height"])
        self.width = int(config["width"])
        self.packed_hw = (self.height // 2, self.width // 2)
        self.noisy, self.noisy_black, self.noisy_white, self.noisy_shift = _spec(config, "noisy")
        self.base, self.base_black, self.base_white, self.base_shift = _spec(config, "base")
        self.temporal, self.temporal_black, self.temporal_white, self.temporal_shift = _spec(config, "temporal")
        self.pseudo_gt, self.pseudo_gt_black, self.pseudo_gt_white, self.pseudo_gt_shift = _spec(config, "pseudo_gt")
        self.frame_count = validate_same_sequence_length((self.noisy, self.base, self.temporal, self.pseudo_gt))

        flow_config = config.get("flow", {})
        forward_dir = flow_config.get("forward_dir")
        backward_dir = flow_config.get("backward_dir")
        if not forward_dir or not backward_dir:
            raise ValueError("flow.forward_dir and flow.backward_dir are required")
        cache_flows = bool(config.get("cache_flows", training)) and training
        self.forward_flow = FlowSequence(
            forward_dir, target_hw=self.packed_hw if cache_flows else None, cache_resized=cache_flows
        )
        self.backward_flow = FlowSequence(
            backward_dir, target_hw=self.packed_hw if cache_flows else None, cache_resized=cache_flows
        )
        expected_pairs = self.frame_count - 1
        if self.forward_flow.pair_count < expected_pairs or self.backward_flow.pair_count < expected_pairs:
            raise ValueError(
                f"Need {expected_pairs} forward/backward flow files, got "
                f"{self.forward_flow.pair_count}/{self.backward_flow.pair_count}"
            )

        drop_boundary = bool(config.get("drop_boundary_for_training", True)) and training
        self.indices = list(range(1, self.frame_count - 1)) if drop_boundary else list(range(self.frame_count))
        self.crop_size = int(config.get("crop_size_packed", 0)) if training else 0
        self.tau_fb = float(flow_config.get("tau_fb_pixels", 1.5))
        self.tau_rgb = float(flow_config.get("tau_rgb", 0.05))
        self.static_motion_threshold = float(flow_config.get("static_motion_threshold_pixels", 0.25))
        self.static_confidence_threshold = float(flow_config.get("static_confidence_threshold", 0.30))
        self.static_erosion_kernel = int(flow_config.get("static_erosion_kernel", 3))

    def __len__(self) -> int:
        return len(self.indices)

    def _read(
        self,
        sequence: RawSequence,
        index: int,
        black: float,
        white: float,
        shift: int,
        crop: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        array = read_packed_normalized(sequence, index, black, white, shift, crop=crop)
        return torch.from_numpy(array.copy())

    def _load_flow(
        self,
        sequence: FlowSequence,
        pair_index: int,
        crop: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        flow = sequence.load(pair_index)
        if sequence.target_hw is None:
            flow = resize_flow(flow, self.packed_hw)
        if crop is not None:
            top, left, size = crop
            flow = flow[..., top:top + size, left:left + size]
        return flow

    def _crop_box(self) -> tuple[int, int, int, int] | None:
        if not self.crop_size:
            return None
        height, width = self.packed_hw
        if self.crop_size > min(height, width):
            raise ValueError(f"crop_size_packed={self.crop_size} exceeds packed frame {self.packed_hw}")
        top = int(torch.randint(0, height - self.crop_size + 1, ()).item())
        left = int(torch.randint(0, width - self.crop_size + 1, ()).item())
        # RAW Bayer coordinates are twice the packed coordinates.
        return (2 * top, 2 * (top + self.crop_size), 2 * left, 2 * (left + self.crop_size))

    def _neighbor_flow_data(
        self,
        index: int,
        center_base: torch.Tensor,
        neighbor_base: torch.Tensor,
        center_to_neighbor: torch.Tensor,
        neighbor_to_center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        confidence, _ = pair_confidence(
            center_base.unsqueeze(0),
            neighbor_base.unsqueeze(0),
            center_to_neighbor.unsqueeze(0),
            neighbor_to_center.unsqueeze(0),
            tau_fb=self.tau_fb,
            tau_rgb=self.tau_rgb,
        )
        return center_to_neighbor, confidence.squeeze(0)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        index = self.indices[item]
        raw_crop = self._crop_box()
        packed_crop = None
        if raw_crop is not None:
            packed_crop = (raw_crop[0] // 2, raw_crop[2] // 2, self.crop_size)
        noisy = self._read(self.noisy, index, self.noisy_black, self.noisy_white, self.noisy_shift, raw_crop)
        base = self._read(self.base, index, self.base_black, self.base_white, self.base_shift, raw_crop)
        temporal = self._read(self.temporal, index, self.temporal_black, self.temporal_white, self.temporal_shift, raw_crop)
        pseudo_gt = self._read(self.pseudo_gt, index, self.pseudo_gt_black, self.pseudo_gt_white, self.pseudo_gt_shift, raw_crop)
        previous_base = (
            self._read(self.base, index - 1, self.base_black, self.base_white, self.base_shift, raw_crop)
            if index > 0
            else base.clone()
        )
        next_base = (
            self._read(self.base, index + 1, self.base_black, self.base_white, self.base_shift, raw_crop)
            if index < self.frame_count - 1
            else base.clone()
        )
        if index > 0:
            flow_previous = self._load_flow(self.backward_flow, index - 1, packed_crop)
            reverse_previous = self._load_flow(self.forward_flow, index - 1, packed_crop)
            _, confidence_previous = self._neighbor_flow_data(
                index, base, previous_base, flow_previous, reverse_previous
            )
        else:
            flow_previous = torch.zeros(2, *self.packed_hw)
            confidence_previous = torch.zeros(1, *self.packed_hw)

        if index < self.frame_count - 1:
            flow_next = self._load_flow(self.forward_flow, index, packed_crop)
            reverse_next = self._load_flow(self.backward_flow, index, packed_crop)
            _, confidence_next = self._neighbor_flow_data(index, base, next_base, flow_next, reverse_next)
        else:
            flow_next = torch.zeros(2, *self.packed_hw)
            confidence_next = torch.zeros(1, *self.packed_hw)

        static_mask = hard_static_mask(
            flow_previous,
            flow_next,
            confidence_previous,
            confidence_next,
            motion_threshold_pixels=self.static_motion_threshold,
            confidence_threshold=self.static_confidence_threshold,
            erosion_kernel=self.static_erosion_kernel,
        )
        temporal_residual = temporal - base
        noisy_residual = noisy - base
        sample: dict[str, torch.Tensor | int] = {
            "frame_index": index,
            "noisy": noisy,
            "base": base,
            "temporal_residual": temporal_residual,
            "noisy_residual": noisy_residual,
            "pseudo_gt": pseudo_gt,
            "static_mask": static_mask,
        }
        return sample


def build_dataset(config: dict[str, Any], training: bool = True) -> FusionDataset:
    return FusionDataset(config, training=training)

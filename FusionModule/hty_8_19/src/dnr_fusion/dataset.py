from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .confidence import SafetyParams, safety_confidence
from .raw_io import RawSpec, RawStream, normalize_raw, pack_rggb


@dataclass
class SceneStreams:
    source: RawStream
    denoised: RawStream
    fused: RawStream

    @property
    def frame_count(self) -> int:
        counts = {self.source.frame_count, self.denoised.frame_count, self.fused.frame_count}
        if len(counts) != 1:
            raise ValueError(f"Triplet frame counts differ: {sorted(counts)}")
        return self.source.frame_count


def open_scene_streams(config: dict[str, Any], scene_name: str) -> SceneStreams:
    raw = config["raw"]
    scene = config["scenes"][scene_name]
    candidate_spec = RawSpec(width=raw["width"], height=raw["height"])
    source_spec = RawSpec(
        width=raw["width"], height=raw["height"], shift=raw["source_shift"]
    )
    streams = SceneStreams(
        source=RawStream(scene["source"], source_spec),
        denoised=RawStream(scene["denoised"], candidate_spec),
        fused=RawStream(scene["fused"], candidate_spec),
    )
    _ = streams.frame_count
    return streams


def scene_safety_params(config: dict[str, Any], scene_name: str) -> SafetyParams:
    raw = config["raw"]
    scene = config["scenes"][scene_name]
    safety = config["safety"]
    return SafetyParams(
        motion_threshold_dn=float(scene["motion_threshold_dn"]),
        disagreement_threshold_dn=float(scene["disagreement_threshold_dn"]),
        dynamic_range_dn=float(raw["white"] - raw["candidate_black"]),
        local_kernel=int(safety["local_kernel"]),
        dilation_kernel=int(safety["dilation_kernel"]),
        soft_scale=float(safety["soft_scale"]),
        hard_motion_scale=float(safety["hard_motion_scale"]),
        hard_joint_motion_scale=float(safety["hard_joint_motion_scale"]),
        hard_joint_disagreement_scale=float(
            safety["hard_joint_disagreement_scale"]
        ),
    )


def load_packed_frame(
    streams: SceneStreams,
    frame_index: int,
    raw_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = normalize_raw(
        pack_rggb(streams.source.frame(frame_index)),
        raw_config["source_black"],
        raw_config["white"],
    )
    denoised = normalize_raw(
        pack_rggb(streams.denoised.frame(frame_index)),
        raw_config["candidate_black"],
        raw_config["white"],
    )
    fused = normalize_raw(
        pack_rggb(streams.fused.frame(frame_index)),
        raw_config["candidate_black"],
        raw_config["white"],
    )
    return source, denoised, fused


def build_features(
    current_source: torch.Tensor,
    current_2dnr: torch.Tensor,
    current_3dnr: torch.Tensor,
    previous_2dnr: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            current_source,
            current_2dnr,
            current_3dnr,
            previous_2dnr,
            current_3dnr - current_2dnr,
            current_2dnr - previous_2dnr,
        ),
        dim=1,
    )


class FusionPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        config: dict[str, Any],
        scene_name: str,
        frame_range: tuple[int, int] | list[int],
        *,
        patch_size: int,
        samples: int,
        seed: int,
        augment: bool,
    ):
        self.config = config
        self.scene_name = scene_name
        self.streams = open_scene_streams(config, scene_name)
        self.frame_start, self.frame_end = map(int, frame_range)
        if self.frame_start < 1 or self.frame_end >= self.streams.frame_count:
            raise ValueError(
                f"Frame range must be within [1, {self.streams.frame_count - 1}]"
            )
        self.patch_size = int(patch_size)
        self.samples = int(samples)
        self.seed = int(seed)
        self.augment = augment
        self.epoch = 0
        self.safety = scene_safety_params(config, scene_name)

        packed_height = config["raw"]["height"] // 2
        packed_width = config["raw"]["width"] // 2
        if patch_size > min(packed_height, packed_width):
            raise ValueError("Patch is larger than packed RAW frame")
        self.packed_height = packed_height
        self.packed_width = packed_width

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(
            self.seed + self.epoch * self.samples + int(index)
        )
        frame_index = int(rng.integers(self.frame_start, self.frame_end + 1))
        top = int(rng.integers(0, self.packed_height - self.patch_size + 1))
        left = int(rng.integers(0, self.packed_width - self.patch_size + 1))
        rows = slice(top, top + self.patch_size)
        cols = slice(left, left + self.patch_size)

        source, denoised, fused = load_packed_frame(
            self.streams, frame_index, self.config["raw"]
        )
        _, previous, _ = load_packed_frame(
            self.streams, frame_index - 1, self.config["raw"]
        )

        current_source = torch.from_numpy(source[:, rows, cols].copy()).unsqueeze(0)
        current_2dnr = torch.from_numpy(denoised[:, rows, cols].copy()).unsqueeze(0)
        current_3dnr = torch.from_numpy(fused[:, rows, cols].copy()).unsqueeze(0)
        previous_2dnr = torch.from_numpy(previous[:, rows, cols].copy()).unsqueeze(0)

        target, diagnostics = safety_confidence(
            current_2dnr, previous_2dnr, current_3dnr, self.safety
        )
        features = build_features(
            current_source, current_2dnr, current_3dnr, previous_2dnr
        )

        features = features.squeeze(0)
        target = target.squeeze(0)
        current_2dnr = current_2dnr.squeeze(0)
        current_3dnr = current_3dnr.squeeze(0)
        hard_mask = diagnostics["hard_mask"].squeeze(0)

        if self.augment:
            if rng.random() < 0.5:
                features = torch.flip(features, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
                current_2dnr = torch.flip(current_2dnr, dims=(-1,))
                current_3dnr = torch.flip(current_3dnr, dims=(-1,))
                hard_mask = torch.flip(hard_mask, dims=(-1,))
            if rng.random() < 0.5:
                features = torch.flip(features, dims=(-2,))
                target = torch.flip(target, dims=(-2,))
                current_2dnr = torch.flip(current_2dnr, dims=(-2,))
                current_3dnr = torch.flip(current_3dnr, dims=(-2,))
                hard_mask = torch.flip(hard_mask, dims=(-2,))
            rotations = int(rng.integers(0, 4))
            if rotations:
                features = torch.rot90(features, rotations, dims=(-2, -1))
                target = torch.rot90(target, rotations, dims=(-2, -1))
                current_2dnr = torch.rot90(
                    current_2dnr, rotations, dims=(-2, -1)
                )
                current_3dnr = torch.rot90(
                    current_3dnr, rotations, dims=(-2, -1)
                )
                hard_mask = torch.rot90(hard_mask, rotations, dims=(-2, -1))

        return {
            "features": features.contiguous(),
            "target_gate": target.contiguous(),
            "hard_mask": hard_mask.contiguous(),
            "denoised": current_2dnr.contiguous(),
            "fused": current_3dnr.contiguous(),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
        }


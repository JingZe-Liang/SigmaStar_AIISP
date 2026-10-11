"""Exact-pair dynamic supervision for a frozen A3 denoising backbone."""

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from raw_fusion.finetune.a3 import (
    _WHITE_LEVEL,
    _channel_vector,
    _load_json,
    _normalise_dn,
    build_rvideformer,
    set_seed,
    update_ema,
)
from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.noise import generate_noisy_frame
from raw_fusion.finetune.stream_io import open_stream, pack_frame


_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
_PACKED_HEIGHT = 540
_PACKED_WIDTH = 960


@dataclass(frozen=True, slots=True)
class A6MotionExpertConfig:
    """Configuration for frozen-backbone motion-expert training."""

    supervision_dir: str
    noise_dir: str
    a3_checkpoint: str
    output_dir: str
    old_teacher_prediction: str | None = None
    initial_expert_checkpoint: str | None = None
    steps: int = 100
    learning_rate: float = 2e-4
    temporal_frames: int = 8
    patch_size: int = 64
    train_frame_start: int = 0
    train_frame_end: int = 160
    holdout_frame_start: int = 160
    holdout_frame_end: int = 200
    temporal_strides: tuple[int, ...] = (1, 2, 3)
    gain_labels: tuple[str, ...] = ("3200", "6400", "12800", "25600")
    motion_crop_probability: float = 0.8
    teacher_distillation_fraction: float = 0.0
    procedural_motion_fraction: float = 0.0
    motion_boundary_radius: int = 2
    same_window_anchor_weight: float = 2.0
    static_anchor_weight: float = 2.0
    gate_supervision_weight: float = 0.1
    gate_static_weight: float = 0.1
    static_every: int = 1
    expert_channels: int = 32
    max_residual: float = 0.25
    ema_decay: float = 0.95
    eval_samples: int = 16
    seed: int = 20261009
    device: str = "cuda:0"


class ExactDynamicDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Low-gain moving RAW frames with independently synthesized input noise."""

    def __init__(
        self,
        *,
        supervision_dir: Path,
        noise_dir: Path,
        frame_start: int,
        frame_end: int,
        temporal_frames: int,
        temporal_strides: tuple[int, ...],
        patch_size: int,
        gain_labels: tuple[str, ...],
        samples: int,
        seed: int,
        training: bool,
        motion_crop_probability: float,
    ) -> None:
        if temporal_frames != 8:
            raise ValueError("the RViDeformer contract requires eight temporal frames")
        if patch_size != 64:
            raise ValueError("the RViDeformer contract requires 64-pixel packed patches")
        if frame_start < 0 or frame_end <= frame_start:
            raise ValueError("invalid frame range")
        if not temporal_strides or any(stride <= 0 for stride in temporal_strides):
            raise ValueError("temporal strides must be positive")
        if not 0.0 <= motion_crop_probability <= 1.0:
            raise ValueError("motion_crop_probability must be in [0, 1]")
        if samples <= 0:
            raise ValueError("samples must be positive")

        manifest = _load_json(
            supervision_dir / "stream_manifests" / "04_shdarkroom_10x.json"
        )
        if frame_end > int(manifest["frames_used"]):
            raise ValueError("frame range exceeds the 10x stream")
        if max(temporal_strides) * (temporal_frames - 1) >= frame_end - frame_start:
            raise ValueError("frame range is too short for the requested temporal strides")
        adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
        reader, _ = open_stream(Path(manifest["path"]), adapter)
        report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
        reference_manifest = _load_json(
            supervision_dir / "references" / "a1_reference_manifest.json"
        )
        noise_model = _load_json(noise_dir / "noise_model.json")
        unavailable = sorted(set(gain_labels) - set(noise_model["gain_labels"]))
        if unavailable:
            raise ValueError(f"noise model does not contain gains: {unavailable}")

        self.frame_start = frame_start
        self.frame_end = frame_end
        self.temporal_frames = temporal_frames
        self.temporal_strides = temporal_strides
        self.patch_size = patch_size
        self.gain_labels = gain_labels
        self.samples = samples
        self.seed = seed
        self.training = training
        self.motion_crop_probability = motion_crop_probability
        self.reader = reader
        self.adapter = adapter
        self.black = _channel_vector(report["black_levels"]).astype(np.float32)
        self.black_levels = {
            name: float(self.black[channel])
            for channel, name in enumerate(_CHANNEL_NAMES)
        }
        self.noise_model = noise_model
        self.fpn_offset = np.load(
            reference_manifest["fpn_paths"]["offset_dn"], mmap_mode="r"
        )
        self.motion = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")

    def __len__(self) -> int:
        return self.samples

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + index * 104729)

    def _frame_indices(self, rng: np.random.Generator) -> np.ndarray:
        stride = self.temporal_strides[int(rng.integers(0, len(self.temporal_strides)))]
        last_offset = stride * (self.temporal_frames - 1)
        latest_start = self.frame_end - last_offset - 1
        start = int(rng.integers(self.frame_start, latest_start + 1))
        return start + np.arange(self.temporal_frames) * stride

    def _crop_origin(self, motion: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        use_motion = rng.random() < self.motion_crop_probability
        positions = np.argwhere(np.any(motion >= 0.5, axis=0)) if use_motion else np.empty((0, 2))
        if positions.size:
            center_y, center_x = positions[int(rng.integers(0, len(positions)))]
            y_low = max(0, int(center_y) - self.patch_size + 1)
            y_high = min(int(center_y), _PACKED_HEIGHT - self.patch_size)
            x_low = max(0, int(center_x) - self.patch_size + 1)
            x_high = min(int(center_x), _PACKED_WIDTH - self.patch_size)
            y = int(rng.integers(y_low, y_high + 1))
            x = int(rng.integers(x_low, x_high + 1))
        else:
            y = int(rng.integers(0, _PACKED_HEIGHT - self.patch_size + 1))
            x = int(rng.integers(0, _PACKED_WIDTH - self.patch_size + 1))
        return y, x

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rng = self._rng(index)
        frame_indices = self._frame_indices(rng)
        motion_full = np.asarray(
            self.motion[frame_indices, 0, ::2, ::2], dtype=np.float32
        )
        y, x = self._crop_origin(motion_full, rng)
        crop = np.s_[:, y:y + self.patch_size, x:x + self.patch_size]
        target_dn = np.stack([
            pack_frame(self.reader.read_frame(int(frame)), self.adapter).astype(np.float32)[crop]
            for frame in frame_indices
        ])
        motion = motion_full[:, y:y + self.patch_size, x:x + self.patch_size]
        fpn = np.asarray(
            self.fpn_offset[:, y:y + self.patch_size, x:x + self.patch_size],
            dtype=np.float32,
        )
        gain = self.gain_labels[int(rng.integers(0, len(self.gain_labels)))]
        noisy_dn = np.stack([
            generate_noisy_frame(
                target_dn[frame],
                self.noise_model,
                gain,
                self.black_levels,
                fpn,
                np.random.default_rng(self.seed + index * 1009 + frame),
            )
            for frame in range(self.temporal_frames)
        ]).astype(np.float32)

        if self.training and rng.random() < 0.5:
            target_dn = target_dn[::-1].copy()
            noisy_dn = noisy_dn[::-1].copy()
            motion = motion[::-1].copy()
        if self.training and rng.random() < 0.5:
            target_dn = target_dn[..., ::-1].copy()
            noisy_dn = noisy_dn[..., ::-1].copy()
            motion = motion[..., ::-1].copy()
        if self.training and rng.random() < 0.5:
            target_dn = target_dn[..., ::-1, :].copy()
            noisy_dn = noisy_dn[..., ::-1, :].copy()
            motion = motion[..., ::-1, :].copy()

        target = np.stack([_normalise_dn(frame, self.black) for frame in target_dn])
        noisy = np.stack([_normalise_dn(frame, self.black) for frame in noisy_dn])
        valid = np.all(target_dn < 3800.0, axis=1)
        valid &= np.all(noisy_dn < 3800.0, axis=1)
        return (
            torch.from_numpy(noisy.astype(np.float32, copy=False)),
            torch.from_numpy(target.astype(np.float32, copy=False)),
            torch.from_numpy(motion[:, None].astype(np.float32, copy=False)),
            torch.from_numpy(valid[:, None].astype(np.float32, copy=False)),
        )


class StaticAnchorDataset(Dataset[tuple[Tensor, Tensor]]):
    """Static A1 clips used only to preserve the frozen A3 response."""

    def __init__(
        self,
        *,
        supervision_dir: Path,
        noise_dir: Path,
        temporal_frames: int,
        patch_size: int,
        gain_labels: tuple[str, ...],
        samples: int,
        seed: int,
    ) -> None:
        reference_manifest = _load_json(
            supervision_dir / "references" / "a1_reference_manifest.json"
        )
        report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
        self.clean = np.load(reference_manifest["clean_reference_path"], mmap_mode="r")
        self.valid = np.load(reference_manifest["valid_region_path"], mmap_mode="r")
        self.fpn_offset = np.load(
            reference_manifest["fpn_paths"]["offset_dn"], mmap_mode="r"
        )
        self.noise_model = _load_json(noise_dir / "noise_model.json")
        self.black = _channel_vector(report["black_levels"]).astype(np.float32)
        self.black_levels = {
            name: float(self.black[channel])
            for channel, name in enumerate(_CHANNEL_NAMES)
        }
        self.temporal_frames = temporal_frames
        self.patch_size = patch_size
        self.gain_labels = gain_labels
        self.samples = samples
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        rng = np.random.default_rng(self.seed + index * 104729)
        y = int(rng.integers(0, _PACKED_HEIGHT - self.patch_size + 1))
        x = int(rng.integers(0, _PACKED_WIDTH - self.patch_size + 1))
        crop = np.s_[:, y:y + self.patch_size, x:x + self.patch_size]
        clean = np.asarray(self.clean[crop], dtype=np.float32)
        fpn = np.asarray(self.fpn_offset[crop], dtype=np.float32)
        gain = self.gain_labels[int(rng.integers(0, len(self.gain_labels)))]
        noisy_dn = np.stack([
            generate_noisy_frame(
                clean,
                self.noise_model,
                gain,
                self.black_levels,
                fpn,
                np.random.default_rng(self.seed + index * 1009 + frame),
            )
            for frame in range(self.temporal_frames)
        ]).astype(np.float32)
        noisy = np.stack([_normalise_dn(frame, self.black) for frame in noisy_dn])
        valid = np.asarray(
            self.valid[y:y + self.patch_size, x:x + self.patch_size], dtype=np.float32
        )
        valid = np.broadcast_to(
            valid[None, None],
            (self.temporal_frames, 1, self.patch_size, self.patch_size),
        ).copy()
        valid *= np.all(noisy_dn < 3800.0, axis=1, keepdims=True)
        return (
            torch.from_numpy(noisy.astype(np.float32, copy=False)),
            torch.from_numpy(valid.astype(np.float32, copy=False)),
        )


class PretrainedMotionDistillationDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]
):
    """Real 645x inputs paired with same-frame official teacher predictions."""

    def __init__(
        self,
        *,
        supervision_dir: Path,
        teacher_prediction: Path,
        frame_start: int,
        frame_end: int,
        temporal_frames: int,
        temporal_strides: tuple[int, ...],
        patch_size: int,
        samples: int,
        seed: int,
        training: bool,
        motion_crop_probability: float,
    ) -> None:
        if not teacher_prediction.is_file():
            raise FileNotFoundError(teacher_prediction)
        manifest = _load_json(
            supervision_dir / "stream_manifests" / "04_shdarkroom_645x.json"
        )
        if frame_start < 0 or frame_end > int(manifest["frames_used"]):
            raise ValueError("invalid 645x frame range")
        if frame_end <= frame_start:
            raise ValueError("645x frame range is empty")
        if not temporal_strides or any(stride <= 0 for stride in temporal_strides):
            raise ValueError("temporal strides must be positive")
        if max(temporal_strides) * (temporal_frames - 1) >= frame_end - frame_start:
            raise ValueError("645x frame range is too short for the temporal strides")
        if not 0.0 <= motion_crop_probability <= 1.0:
            raise ValueError("motion_crop_probability must be in [0, 1]")
        adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
        reader, _ = open_stream(Path(manifest["path"]), adapter)
        report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
        teacher = np.load(teacher_prediction, mmap_mode="r")
        if teacher.ndim != 4 or teacher.shape[1:] != (
            4,
            _PACKED_HEIGHT,
            _PACKED_WIDTH,
        ):
            raise ValueError(f"invalid old teacher prediction shape {teacher.shape}")
        if teacher.shape[0] < frame_end:
            raise ValueError("old teacher prediction does not cover the frame range")

        self.frame_start = frame_start
        self.frame_end = frame_end
        self.temporal_frames = temporal_frames
        self.temporal_strides = temporal_strides
        self.patch_size = patch_size
        self.samples = samples
        self.seed = seed
        self.training = training
        self.motion_crop_probability = motion_crop_probability
        self.reader = reader
        self.adapter = adapter
        self.teacher = teacher
        self.motion = np.load(Path(manifest["motion"]["mask_path"]), mmap_mode="r")
        self.black = _channel_vector(report["black_levels"]).astype(np.float32)
        fit = manifest["exposure_fit"]
        self.slopes = _channel_vector(fit["slopes"]).astype(np.float32)
        self.intercepts = _channel_vector(fit["intercepts"]).astype(np.float32)
        self.shift_x = int(manifest["registration"]["integer_shift_x"])
        self.shift_y = int(manifest["registration"]["integer_shift_y"])

    def __len__(self) -> int:
        return self.samples

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + index * 104729)

    def _frame_indices(self, rng: np.random.Generator) -> np.ndarray:
        stride = self.temporal_strides[int(rng.integers(0, len(self.temporal_strides)))]
        last_offset = stride * (self.temporal_frames - 1)
        latest_start = self.frame_end - last_offset - 1
        start = int(rng.integers(self.frame_start, latest_start + 1))
        return start + np.arange(self.temporal_frames) * stride

    def _aligned_motion(self, frame_indices: np.ndarray) -> np.ndarray:
        source = np.asarray(
            self.motion[frame_indices, 0, ::2, ::2], dtype=np.float32
        )
        aligned = np.ones_like(source)
        y0 = max(0, -self.shift_y)
        y1 = min(_PACKED_HEIGHT, _PACKED_HEIGHT - self.shift_y)
        x0 = max(0, -self.shift_x)
        x1 = min(_PACKED_WIDTH, _PACKED_WIDTH - self.shift_x)
        aligned[:, y0:y1, x0:x1] = source[
            :, y0 + self.shift_y:y1 + self.shift_y,
            x0 + self.shift_x:x1 + self.shift_x,
        ]
        return aligned

    def _crop_origin(self, motion: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        positions = np.empty((0, 2))
        if rng.random() < self.motion_crop_probability:
            positions = np.argwhere(np.any(motion >= 0.5, axis=0))
        if positions.size:
            center_y, center_x = positions[int(rng.integers(0, len(positions)))]
            y_low = max(0, int(center_y) - self.patch_size + 1)
            y_high = min(int(center_y), _PACKED_HEIGHT - self.patch_size)
            x_low = max(0, int(center_x) - self.patch_size + 1)
            x_high = min(int(center_x), _PACKED_WIDTH - self.patch_size)
            y = int(rng.integers(y_low, y_high + 1))
            x = int(rng.integers(x_low, x_high + 1))
        else:
            y = int(rng.integers(0, _PACKED_HEIGHT - self.patch_size + 1))
            x = int(rng.integers(0, _PACKED_WIDTH - self.patch_size + 1))
        return y, x

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rng = self._rng(index)
        frame_indices = self._frame_indices(rng)
        motion_full = self._aligned_motion(frame_indices)
        y, x = self._crop_origin(motion_full, rng)
        source_y = y + self.shift_y
        source_x = x + self.shift_x
        inputs_dn = np.stack([
            self.slopes[:, None, None]
            * pack_frame(self.reader.read_frame(int(frame)), self.adapter).astype(np.float32)[
                :, source_y:source_y + self.patch_size,
                source_x:source_x + self.patch_size,
            ]
            + self.intercepts[:, None, None]
            for frame in frame_indices
        ])
        inputs = np.stack([_normalise_dn(frame, self.black) for frame in inputs_dn])
        target = np.asarray(
            self.teacher[
                frame_indices,
                :,
                y:y + self.patch_size,
                x:x + self.patch_size,
            ],
            dtype=np.float32,
        )
        motion = motion_full[:, y:y + self.patch_size, x:x + self.patch_size]
        valid = np.all(inputs_dn < 3800.0, axis=1)

        if self.training and rng.random() < 0.5:
            inputs = inputs[::-1].copy()
            target = target[::-1].copy()
            motion = motion[::-1].copy()
            valid = valid[::-1].copy()
        if self.training and rng.random() < 0.5:
            inputs = inputs[..., ::-1].copy()
            target = target[..., ::-1].copy()
            motion = motion[..., ::-1].copy()
            valid = valid[..., ::-1].copy()
        if self.training and rng.random() < 0.5:
            inputs = inputs[..., ::-1, :].copy()
            target = target[..., ::-1, :].copy()
            motion = motion[..., ::-1, :].copy()
            valid = valid[..., ::-1, :].copy()

        return (
            torch.from_numpy(inputs.astype(np.float32, copy=False)),
            torch.from_numpy(target.astype(np.float32, copy=False)),
            torch.from_numpy(motion[:, None].astype(np.float32, copy=False)),
            torch.from_numpy(valid[:, None].astype(np.float32, copy=False)),
        )


class ProceduralMotionDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Exact local motion synthesized from clean A1 RAW patches."""

    def __init__(
        self,
        *,
        supervision_dir: Path,
        noise_dir: Path,
        temporal_frames: int,
        patch_size: int,
        gain_labels: tuple[str, ...],
        samples: int,
        seed: int,
    ) -> None:
        if temporal_frames != 8 or patch_size != 64:
            raise ValueError("procedural motion requires the 8x64 RViDeformer contract")
        reference_manifest = _load_json(
            supervision_dir / "references" / "a1_reference_manifest.json"
        )
        report = _load_json(supervision_dir / "references" / "a1_reference_report.json")
        self.clean = np.load(reference_manifest["clean_reference_path"], mmap_mode="r")
        self.reference_valid = np.load(
            reference_manifest["valid_region_path"], mmap_mode="r"
        )
        self.fpn_offset = np.load(
            reference_manifest["fpn_paths"]["offset_dn"], mmap_mode="r"
        )
        self.noise_model = _load_json(noise_dir / "noise_model.json")
        self.black = _channel_vector(report["black_levels"]).astype(np.float32)
        self.black_levels = {
            name: float(self.black[channel])
            for channel, name in enumerate(_CHANNEL_NAMES)
        }
        self.temporal_frames = temporal_frames
        self.patch_size = patch_size
        self.gain_labels = gain_labels
        self.samples = samples
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def _valid_crop(
        self,
        rng: np.random.Generator,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        for _ in range(32):
            y = int(rng.integers(0, _PACKED_HEIGHT - height + 1))
            x = int(rng.integers(0, _PACKED_WIDTH - width + 1))
            if float(np.mean(self.reference_valid[y:y + height, x:x + width])) >= 0.9:
                return y, x
        raise ValueError("could not sample a sufficiently valid A1 crop")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rng = np.random.default_rng(self.seed + index * 104729)
        background_y, background_x = self._valid_crop(
            rng, self.patch_size, self.patch_size
        )
        background_crop = np.s_[
            :,
            background_y:background_y + self.patch_size,
            background_x:background_x + self.patch_size,
        ]
        background = np.asarray(self.clean[background_crop], dtype=np.float32)
        fpn = np.asarray(self.fpn_offset[background_crop], dtype=np.float32)

        object_size = int(rng.choice(np.asarray((20, 24, 28, 32))))
        object_y, object_x = self._valid_crop(rng, object_size, object_size)
        moving_object = np.asarray(
            self.clean[
                :,
                object_y:object_y + object_size,
                object_x:object_x + object_size,
            ],
            dtype=np.float32,
        )
        velocity_choices = np.asarray((-3, -2, -1, 1, 2, 3))
        velocity_y = int(rng.choice(velocity_choices))
        velocity_x = int(rng.choice(velocity_choices))
        span_y = abs(velocity_y) * (self.temporal_frames - 1)
        span_x = abs(velocity_x) * (self.temporal_frames - 1)
        if velocity_y > 0:
            start_y = int(rng.integers(0, self.patch_size - object_size - span_y + 1))
        else:
            start_y = int(rng.integers(span_y, self.patch_size - object_size + 1))
        if velocity_x > 0:
            start_x = int(rng.integers(0, self.patch_size - object_size - span_x + 1))
        else:
            start_x = int(rng.integers(span_x, self.patch_size - object_size + 1))

        alpha_1d = np.hanning(object_size).astype(np.float32)
        alpha = np.outer(alpha_1d, alpha_1d)[None]
        target_dn: list[np.ndarray] = []
        motion_masks: list[np.ndarray] = []
        for frame in range(self.temporal_frames):
            y = start_y + frame * velocity_y
            x = start_x + frame * velocity_x
            target_frame = background.copy()
            region = target_frame[:, y:y + object_size, x:x + object_size]
            region *= 1.0 - alpha
            region += moving_object * alpha
            motion = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            motion[y:y + object_size, x:x + object_size] = alpha[0] >= 0.1
            target_dn.append(target_frame)
            motion_masks.append(motion)
        target_array = np.stack(target_dn)
        motion_array = np.stack(motion_masks)

        gain = self.gain_labels[int(rng.integers(0, len(self.gain_labels)))]
        noisy_dn = np.stack([
            generate_noisy_frame(
                target_array[frame],
                self.noise_model,
                gain,
                self.black_levels,
                fpn,
                np.random.default_rng(self.seed + index * 1009 + frame),
            )
            for frame in range(self.temporal_frames)
        ]).astype(np.float32)
        target = np.stack([_normalise_dn(frame, self.black) for frame in target_array])
        noisy = np.stack([_normalise_dn(frame, self.black) for frame in noisy_dn])
        valid = np.all(target_array < 3800.0, axis=1)
        valid &= np.all(noisy_dn < 3800.0, axis=1)
        return (
            torch.from_numpy(noisy.astype(np.float32, copy=False)),
            torch.from_numpy(target.astype(np.float32, copy=False)),
            torch.from_numpy(motion_array[:, None]),
            torch.from_numpy(valid[:, None].astype(np.float32, copy=False)),
        )


class MotionResidualExpert(nn.Module):
    """Predict a gated residual while leaving the A3 backbone untouched."""

    def __init__(self, channels: int = 32, max_residual: float = 0.25) -> None:
        super().__init__()
        if channels <= 0 or max_residual <= 0.0:
            raise ValueError("expert channels and max residual must be positive")
        self.max_residual = max_residual
        self.features = nn.Sequential(
            nn.Conv3d(12, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.GELU(),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.GELU(),
        )
        self.residual_head = nn.Conv3d(
            channels, 4, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.gate_head = nn.Conv3d(
            channels, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.2)

    def forward(
        self,
        inputs: Tensor,
        baseline: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if inputs.shape != baseline.shape or inputs.ndim != 5 or inputs.shape[2] != 4:
            raise ValueError(
                f"expected matching [B,T,4,H,W] inputs, got {inputs.shape} and {baseline.shape}"
            )
        previous = torch.cat((baseline[:, :1], baseline[:, :-1]), dim=1)
        following = torch.cat((baseline[:, 1:], baseline[:, -1:]), dim=1)
        temporal_difference = 0.5 * (
            torch.abs(baseline - previous) + torch.abs(following - baseline)
        )
        features = torch.cat((inputs, baseline, temporal_difference), dim=2)
        features = self.features(features.transpose(1, 2))
        residual = torch.tanh(self.residual_head(features)).transpose(1, 2)
        residual = residual * self.max_residual
        gate = torch.sigmoid(self.gate_head(features)).transpose(1, 2)
        candidate = baseline + gate * residual
        return candidate, gate, residual


class FrozenA3MotionModel(nn.Module):
    """Inference wrapper with an immutable A3 backbone and trainable expert."""

    def __init__(self, backbone: nn.Module, expert: MotionResidualExpert) -> None:
        super().__init__()
        self.backbone = backbone
        self.expert = expert
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenA3MotionModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_with_aux(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            baseline = self.backbone(inputs).clamp(0.0, 1.0)
        candidate, gate, residual = self.expert(inputs, baseline)
        return candidate.clamp(0.0, 1.0), baseline, gate, residual

    def forward(self, inputs: Tensor) -> Tensor:
        candidate, _, _, _ = self.forward_with_aux(inputs)
        return candidate


def _load_a3_backbone(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise ValueError("A6 requires an A3 checkpoint containing EMA weights")
    backbone = build_rvideformer()
    backbone.load_state_dict(payload["ema"], strict=True)
    return backbone.to(device).eval()


def build_motion_expert_model(
    *,
    a3_checkpoint: Path,
    device: torch.device,
    expert_channels: int,
    max_residual: float,
) -> FrozenA3MotionModel:
    backbone = _load_a3_backbone(a3_checkpoint, device)
    expert = MotionResidualExpert(expert_channels, max_residual).to(device)
    return FrozenA3MotionModel(backbone, expert).to(device)


def load_motion_expert_checkpoint(
    checkpoint: Path,
    device: torch.device,
    weights: str = "ema",
) -> FrozenA3MotionModel:
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be 'model' or 'ema'")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = build_motion_expert_model(
        a3_checkpoint=Path(config["a3_checkpoint"]),
        device=device,
        expert_channels=int(config["expert_channels"]),
        max_residual=float(config["max_residual"]),
    )
    state_key = "ema_expert" if weights == "ema" else "expert"
    model.expert.load_state_dict(payload[state_key], strict=True)
    return model.eval()


def dilate_dynamic_mask(mask: Tensor, radius: int) -> Tensor:
    if radius < 0:
        raise ValueError("motion boundary radius must be non-negative")
    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError(f"expected [B,T,1,H,W] mask, got {tuple(mask.shape)}")
    if radius == 0:
        return mask.clone()
    frames = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(frames, kernel_size=kernel, stride=1, padding=radius)
    return dilated.reshape_as(mask)


def _masked_charbonnier(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.expand_as(prediction)
    pixels = expanded.sum()
    if float(pixels.detach().cpu()) <= 0.0:
        return prediction.sum() * 0.0
    robust_error = torch.sqrt((prediction - target).square() + 1e-6) - 1e-3
    return (robust_error * expanded).sum() / pixels


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.expand_as(values)
    pixels = expanded.sum()
    if float(pixels.detach().cpu()) <= 0.0:
        return values.sum() * 0.0
    return (values * expanded).sum() / pixels


def _balanced_gate_loss(gate: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    positive = (target * valid).sum()
    negative = ((1.0 - target) * valid).sum()
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    weights = valid * (1.0 + target * (positive_weight - 1.0))
    loss = F.binary_cross_entropy(gate, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _rmse_dn(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    scale: Tensor,
) -> float:
    expanded = mask.expand_as(prediction)
    pixels = expanded.sum()
    if float(pixels.detach().cpu()) <= 0.0:
        return float("nan")
    squared = ((prediction - target) * scale).square() * expanded
    return float(torch.sqrt(squared.sum() / pixels).detach().cpu())


@torch.no_grad()
def evaluate_exact_dynamic(
    model: FrozenA3MotionModel,
    dataset: ExactDynamicDataset,
    device: torch.device,
    motion_boundary_radius: int,
) -> dict[str, float]:
    model.eval()
    scale = torch.from_numpy((_WHITE_LEVEL - dataset.black).astype(np.float32)).to(device)
    scale = scale.view(1, 1, 4, 1, 1)
    rows: list[dict[str, float]] = []
    for index in range(len(dataset)):
        inputs, target, motion, valid = dataset[index]
        inputs_device = inputs[None].to(device)
        target_device = target[None].to(device)
        motion_device = dilate_dynamic_mask(
            motion[None].to(device), motion_boundary_radius
        )
        valid_device = valid[None].to(device)
        candidate, baseline, gate, _ = model.forward_with_aux(inputs_device)
        dynamic = motion_device * valid_device
        static = (1.0 - motion_device) * valid_device
        candidate_rmse = _rmse_dn(candidate, target_device, dynamic, scale)
        baseline_rmse = _rmse_dn(baseline, target_device, dynamic, scale)
        static_shift = _rmse_dn(candidate, baseline, static, scale)
        gate_binary = gate >= 0.5
        motion_binary = motion_device >= 0.5
        valid_binary = valid_device >= 0.5
        intersection = torch.count_nonzero(gate_binary & motion_binary & valid_binary)
        union = torch.count_nonzero((gate_binary | motion_binary) & valid_binary)

        candidate_diff = torch.abs(torch.diff(candidate * scale, dim=1))
        baseline_diff = torch.abs(torch.diff(baseline * scale, dim=1))
        target_diff = torch.abs(torch.diff(target_device * scale, dim=1))
        transition = torch.maximum(motion_device[:, 1:], motion_device[:, :-1])
        transition *= valid_device[:, 1:] * valid_device[:, :-1]
        candidate_motion = _masked_mean(candidate_diff, transition)
        baseline_motion = _masked_mean(baseline_diff, transition)
        target_motion = _masked_mean(target_diff, transition)
        rows.append({
            "candidate_dynamic_rmse_dn": candidate_rmse,
            "baseline_dynamic_rmse_dn": baseline_rmse,
            "candidate_dynamic_psnr_db": float(
                20.0 * np.log10(_WHITE_LEVEL / max(candidate_rmse, 1e-8))
            ),
            "baseline_dynamic_psnr_db": float(
                20.0 * np.log10(_WHITE_LEVEL / max(baseline_rmse, 1e-8))
            ),
            "static_shift_from_a3_rmse_dn": static_shift,
            "gate_motion_mean": float(_masked_mean(gate, dynamic).cpu()),
            "gate_static_mean": float(_masked_mean(gate, static).cpu()),
            "gate_iou_0p5": float(intersection / union.clamp_min(1)),
            "motion_to_target_temporal_ratio": float(
                candidate_motion / target_motion.clamp_min(1e-8)
            ),
            "baseline_to_target_temporal_ratio": float(
                baseline_motion / target_motion.clamp_min(1e-8)
            ),
        })
    return {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0]
    } | {"samples": float(len(rows))}


def train_motion_expert(config: A6MotionExpertConfig) -> dict[str, object]:
    if config.steps <= 0 or config.learning_rate <= 0.0:
        raise ValueError("steps and learning rate must be positive")
    if config.static_every <= 0:
        raise ValueError("static_every must be positive")
    if not 0.0 < config.ema_decay < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    if not 0.0 <= config.teacher_distillation_fraction <= 1.0:
        raise ValueError("teacher_distillation_fraction must be in [0, 1]")
    if not 0.0 <= config.procedural_motion_fraction <= 1.0:
        raise ValueError("procedural_motion_fraction must be in [0, 1]")
    if (
        config.teacher_distillation_fraction + config.procedural_motion_fraction
        > 1.0
    ):
        raise ValueError("teacher and procedural fractions must sum to at most 1")
    if config.teacher_distillation_fraction > 0.0 and config.old_teacher_prediction is None:
        raise ValueError(
            "old_teacher_prediction is required when teacher distillation is enabled"
        )
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A6 motion-expert training requires CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    set_seed(config.seed)

    supervision_dir = Path(config.supervision_dir)
    noise_dir = Path(config.noise_dir)
    train_dynamic = ExactDynamicDataset(
        supervision_dir=supervision_dir,
        noise_dir=noise_dir,
        frame_start=config.train_frame_start,
        frame_end=config.train_frame_end,
        temporal_frames=config.temporal_frames,
        temporal_strides=config.temporal_strides,
        patch_size=config.patch_size,
        gain_labels=config.gain_labels,
        samples=max(config.steps, 64),
        seed=config.seed,
        training=True,
        motion_crop_probability=config.motion_crop_probability,
    )
    holdout_dynamic = ExactDynamicDataset(
        supervision_dir=supervision_dir,
        noise_dir=noise_dir,
        frame_start=config.holdout_frame_start,
        frame_end=config.holdout_frame_end,
        temporal_frames=config.temporal_frames,
        temporal_strides=(1, 2, 3),
        patch_size=config.patch_size,
        gain_labels=config.gain_labels,
        samples=config.eval_samples,
        seed=config.seed + 1_000_003,
        training=False,
        motion_crop_probability=1.0,
    )
    train_evaluation = ExactDynamicDataset(
        supervision_dir=supervision_dir,
        noise_dir=noise_dir,
        frame_start=config.train_frame_start,
        frame_end=config.train_frame_end,
        temporal_frames=config.temporal_frames,
        temporal_strides=(1, 2, 3),
        patch_size=config.patch_size,
        gain_labels=config.gain_labels,
        samples=config.eval_samples,
        seed=config.seed + 3_000_007,
        training=False,
        motion_crop_probability=1.0,
    )
    teacher_train: PretrainedMotionDistillationDataset | None = None
    teacher_holdout: PretrainedMotionDistillationDataset | None = None
    if config.old_teacher_prediction is not None:
        teacher_path = Path(config.old_teacher_prediction)
        teacher_train = PretrainedMotionDistillationDataset(
            supervision_dir=supervision_dir,
            teacher_prediction=teacher_path,
            frame_start=config.train_frame_start,
            frame_end=config.train_frame_end,
            temporal_frames=config.temporal_frames,
            temporal_strides=config.temporal_strides,
            patch_size=config.patch_size,
            samples=max(config.steps, 64),
            seed=config.seed + 4_000_009,
            training=True,
            motion_crop_probability=config.motion_crop_probability,
        )
        teacher_holdout = PretrainedMotionDistillationDataset(
            supervision_dir=supervision_dir,
            teacher_prediction=teacher_path,
            frame_start=config.holdout_frame_start,
            frame_end=config.holdout_frame_end,
            temporal_frames=config.temporal_frames,
            temporal_strides=(1, 2, 3),
            patch_size=config.patch_size,
            samples=config.eval_samples,
            seed=config.seed + 5_000_011,
            training=False,
            motion_crop_probability=1.0,
        )
    procedural_train: ProceduralMotionDataset | None = None
    procedural_holdout: ProceduralMotionDataset | None = None
    if config.procedural_motion_fraction > 0.0:
        procedural_train = ProceduralMotionDataset(
            supervision_dir=supervision_dir,
            noise_dir=noise_dir,
            temporal_frames=config.temporal_frames,
            patch_size=config.patch_size,
            gain_labels=config.gain_labels,
            samples=max(config.steps, 64),
            seed=config.seed + 7_000_019,
        )
        procedural_holdout = ProceduralMotionDataset(
            supervision_dir=supervision_dir,
            noise_dir=noise_dir,
            temporal_frames=config.temporal_frames,
            patch_size=config.patch_size,
            gain_labels=config.gain_labels,
            samples=config.eval_samples,
            seed=config.seed + 8_000_023,
        )
    static_anchor = StaticAnchorDataset(
        supervision_dir=supervision_dir,
        noise_dir=noise_dir,
        temporal_frames=config.temporal_frames,
        patch_size=config.patch_size,
        gain_labels=config.gain_labels,
        samples=max(config.steps, 64),
        seed=config.seed + 2_000_003,
    )
    model = build_motion_expert_model(
        a3_checkpoint=Path(config.a3_checkpoint),
        device=device,
        expert_channels=config.expert_channels,
        max_residual=config.max_residual,
    )
    if config.initial_expert_checkpoint is not None:
        initial_payload = torch.load(
            config.initial_expert_checkpoint, map_location="cpu", weights_only=True
        )
        model.expert.load_state_dict(initial_payload["ema_expert"], strict=True)
    ema_expert = copy.deepcopy(model.expert).to(device).eval()
    for parameter in ema_expert.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.expert.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    before = evaluate_exact_dynamic(
        model, holdout_dynamic, device, config.motion_boundary_radius
    )
    teacher_before = None
    if teacher_holdout is not None:
        teacher_before = evaluate_exact_dynamic(
            model, teacher_holdout, device, config.motion_boundary_radius
        )
    procedural_before = None
    if procedural_holdout is not None:
        procedural_before = evaluate_exact_dynamic(
            model, procedural_holdout, device, config.motion_boundary_radius
        )
    source_rng = np.random.default_rng(config.seed + 6_000_013)
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, config.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            source_draw = source_rng.random()
            use_teacher = (
                teacher_train is not None
                and source_draw < config.teacher_distillation_fraction
            )
            use_procedural = (
                not use_teacher
                and procedural_train is not None
                and source_draw
                < config.teacher_distillation_fraction
                + config.procedural_motion_fraction
            )
            if use_teacher:
                active_dataset = teacher_train
                source_name = "old_teacher_645x"
            elif use_procedural:
                active_dataset = procedural_train
                source_name = "procedural_a1_motion"
            else:
                active_dataset = train_dynamic
                source_name = "exact_10x"
            if active_dataset is None:
                raise RuntimeError("the selected A6 training dataset is unavailable")
            inputs, target, motion, valid = active_dataset[
                (step - 1) % len(active_dataset)
            ]
            inputs_device = inputs[None].to(device)
            target_device = target[None].to(device)
            valid_device = valid[None].to(device)
            motion_device = dilate_dynamic_mask(
                motion[None].to(device), config.motion_boundary_radius
            )
            candidate, baseline, gate, _ = model.forward_with_aux(inputs_device)
            dynamic_mask = motion_device * valid_device
            static_mask = (1.0 - motion_device) * valid_device
            motion_loss = _masked_charbonnier(candidate, target_device, dynamic_mask)
            same_window_anchor = _masked_charbonnier(candidate, baseline, static_mask)
            gate_supervision = _balanced_gate_loss(gate, motion_device, valid_device)
            gate_static = _masked_mean(gate, static_mask)

            static_loss = candidate.sum() * 0.0
            static_gate = candidate.sum() * 0.0
            if step % config.static_every == 0:
                static_inputs, static_valid = static_anchor[(step - 1) % len(static_anchor)]
                static_candidate, static_baseline, static_gate_values, _ = model.forward_with_aux(
                    static_inputs[None].to(device)
                )
                static_valid_device = static_valid[None].to(device)
                static_loss = _masked_charbonnier(
                    static_candidate, static_baseline, static_valid_device
                )
                static_gate = _masked_mean(static_gate_values, static_valid_device)

            total = (
                motion_loss
                + config.same_window_anchor_weight * same_window_anchor
                + config.static_anchor_weight * static_loss
                + config.gate_supervision_weight * gate_supervision
                + config.gate_static_weight * (gate_static + static_gate)
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite A6 loss at step {step}")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.expert.parameters(), max_norm=1.0
            )
            optimizer.step()
            update_ema(ema_expert, model.expert, decay=config.ema_decay)
            record = {
                "step": step,
                "source": source_name,
                "total": float(total.detach().cpu()),
                "motion": float(motion_loss.detach().cpu()),
                "same_window_anchor": float(same_window_anchor.detach().cpu()),
                "static_anchor": float(static_loss.detach().cpu()),
                "gate_supervision": float(gate_supervision.detach().cpu()),
                "gate_static": float(gate_static.detach().cpu()),
                "static_gate": float(static_gate.detach().cpu()),
                "dynamic_mask_fraction": float(dynamic_mask.mean().detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(
                    f"[a6-motion] step={step} total={record['total']:.6f} "
                    f"motion={record['motion']:.6f} static={record['static_anchor']:.6f}",
                    flush=True,
                )

    after_model = evaluate_exact_dynamic(
        model, holdout_dynamic, device, config.motion_boundary_radius
    )
    train_after_model = evaluate_exact_dynamic(
        model, train_evaluation, device, config.motion_boundary_radius
    )
    teacher_after_model = None
    if teacher_holdout is not None:
        teacher_after_model = evaluate_exact_dynamic(
            model, teacher_holdout, device, config.motion_boundary_radius
        )
    procedural_after_model = None
    if procedural_holdout is not None:
        procedural_after_model = evaluate_exact_dynamic(
            model, procedural_holdout, device, config.motion_boundary_radius
        )
    live_state = copy.deepcopy(model.expert.state_dict())
    model.expert.load_state_dict(ema_expert.state_dict(), strict=True)
    after_ema = evaluate_exact_dynamic(
        model, holdout_dynamic, device, config.motion_boundary_radius
    )
    train_after_ema = evaluate_exact_dynamic(
        model, train_evaluation, device, config.motion_boundary_radius
    )
    teacher_after_ema = None
    if teacher_holdout is not None:
        teacher_after_ema = evaluate_exact_dynamic(
            model, teacher_holdout, device, config.motion_boundary_radius
        )
    procedural_after_ema = None
    if procedural_holdout is not None:
        procedural_after_ema = evaluate_exact_dynamic(
            model, procedural_holdout, device, config.motion_boundary_radius
        )
    model.expert.load_state_dict(live_state, strict=True)
    checkpoint_path = output_dir / f"checkpoint_step_{config.steps:06d}.pth"
    torch.save({
        "step": config.steps,
        "expert": model.expert.state_dict(),
        "ema_expert": ema_expert.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "initialization": {
            "a3_checkpoint": str(Path(config.a3_checkpoint).resolve()),
            "a3_weights": "ema",
            "backbone_frozen": True,
            "initial_expert_checkpoint": config.initial_expert_checkpoint,
        },
    }, checkpoint_path)
    result: dict[str, object] = {
        "protocol": "A6 frozen A3 motion expert with exact low-gain dynamic targets",
        "checkpoint": str(checkpoint_path.resolve()),
        "train_frame_range": [config.train_frame_start, config.train_frame_end],
        "holdout_frame_range": [config.holdout_frame_start, config.holdout_frame_end],
        "exact_holdout_before": before,
        "exact_holdout_after_model": after_model,
        "exact_holdout_after_ema": after_ema,
        "exact_train_after_model": train_after_model,
        "exact_train_after_ema": train_after_ema,
        "old_teacher_holdout_before": teacher_before,
        "old_teacher_holdout_after_model": teacher_after_model,
        "old_teacher_holdout_after_ema": teacher_after_ema,
        "procedural_holdout_before": procedural_before,
        "procedural_holdout_after_model": procedural_after_model,
        "procedural_holdout_after_ema": procedural_after_ema,
        "backbone_frozen": True,
        "real_video_promotion_status": "not evaluated",
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result

"""Stateless A12 training sampling with fresh crops, windows, and patch noise.

Use ``A12SamplingDataset(config)`` with ``A12StepSampler(dataset, start_step)``
and a DataLoader using ``batch_size=config.batch_size`` and ``shuffle=False``.
``start_step`` counts completed optimizer updates, with one loader batch per
update and no gradient accumulation. Save the sampling config with checkpoints.
Existing A3 datasets and evaluation protocols remain unchanged.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import operator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler

from raw_fusion.finetune.a3 import (
    A3Config,
    PhaseADataset,
    _CHANNEL_NAMES,
    _PACKED_HEIGHT,
    _PACKED_WIDTH,
    _WHITE_LEVEL,
    _load_json,
    _load_reference,
    _normalise_dn,
    _real_records,
    _synthetic_records,
)
from raw_fusion.finetune.noise import generate_noisy_frame


@dataclass(frozen=True, slots=True)
class A12SamplePlan:
    """Serializable identity of one sample, independent of worker or access order."""

    sample_index: int
    epoch: int
    epoch_index: int
    source: Literal["real", "synthetic"]
    record_index: int
    gain: str
    crop_y: int
    crop_x: int
    frame_start: int | None
    geometry_seed: int
    noise_seed: int | None


class A12SamplingDataset(PhaseADataset):
    """Address every training sample by its absolute position in the run.

    A complete epoch contains ``round(samples_per_epoch * real_fraction)`` real
    samples distributed evenly among synthetic samples. Noise is generated only
    for the requested patch; the old full-frame noise cache is never built.
    Reuse A3's real loading, registration, exposure mapping, and mask semantics.
    This sampler changes data coverage, not the supervision or noise model.
    """

    def __init__(self, config: A3Config, samples_per_epoch: int = 512) -> None:
        if config.temporal_frames != 8 or config.patch_size != 64:
            raise ValueError("A12 requires the A3 input shape: 8 frames and 64 packed pixels")
        if config.steps <= 0 or config.batch_size <= 0 or samples_per_epoch < 2:
            raise ValueError("steps and batch_size must be positive; samples_per_epoch must be >= 2")
        if config.seed < 0 or not 0.0 < config.real_fraction < 1.0:
            raise ValueError("seed must be nonnegative and real_fraction must be in (0, 1)")
        real_count = round(samples_per_epoch * config.real_fraction)
        if not 0 < real_count < samples_per_epoch:
            raise ValueError("the epoch must contain both real and synthetic samples")

        self.config = config
        self.samples = config.steps * config.batch_size
        self.samples_per_epoch = samples_per_epoch
        self.real_samples_per_epoch = real_count
        self.supervision_dir = Path(config.supervision_dir)
        self.noise_dir = Path(config.noise_dir)
        self.clean_reference, self.valid_region, self.reference_black = _load_reference(self.supervision_dir)
        if self.clean_reference.shape != (4, _PACKED_HEIGHT, _PACKED_WIDTH):
            raise ValueError("A12 reuses A3's 4x540x960 reference and registration geometry")
        self.clean_normalised = _normalise_dn(self.clean_reference, self.reference_black)
        self.real = _real_records(self.supervision_dir, config.holdout_gain)
        self.synthetic = _synthetic_records(self.noise_dir)
        if any(record.frame_count < config.temporal_frames for record in self.real):
            raise ValueError("each real stream must contain at least eight frames")
        self._readers: dict[Path, tuple[object, object]] = {}
        self._motion_masks: dict[Path, np.ndarray] = {}

        self.noise_model = _load_json(self.noise_dir / "noise_model.json")
        reference_manifest = _load_json(self.supervision_dir / "references" / "a1_reference_manifest.json")
        self.fpn_offset = np.load(reference_manifest["fpn_paths"]["offset_dn"], mmap_mode="r")
        if self.fpn_offset.shape != self.clean_reference.shape:
            raise ValueError("FPN and clean reference must have identical shapes")
        self.black_levels = {name: float(self.reference_black[c]) for c, name in enumerate(_CHANNEL_NAMES)}

    def sample_plan(self, sample_index: int) -> A12SamplePlan:
        """Return crop/window/noise identity without reading any RAW frames."""

        sample_index = operator.index(sample_index)
        if not 0 <= sample_index < self.samples:
            raise IndexError(f"sample_index {sample_index} outside [0, {self.samples})")
        epoch, position = divmod(sample_index, self.samples_per_epoch)
        count = self.real_samples_per_epoch
        period = self.samples_per_epoch
        real_before = (position * count + period - 1) // period
        real_after = ((position + 1) * count + period - 1) // period
        real_seen = epoch * count + real_before
        geometry_seed = int(np.random.SeedSequence(
            [self.config.seed, epoch, position, 12, 0],
        ).generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(geometry_seed)
        patch = self.config.patch_size

        if real_after > real_before:
            source = "real"
            record_index = real_seen % len(self.real)
            record = self.real[record_index]
            low_y, high_y = max(0, -record.shift_y), min(_PACKED_HEIGHT - patch, _PACKED_HEIGHT - patch - record.shift_y)
            low_x, high_x = max(0, -record.shift_x), min(_PACKED_WIDTH - patch, _PACKED_WIDTH - patch - record.shift_x)
            if high_y < low_y or high_x < low_x:
                raise ValueError(f"registration shift leaves no patch overlap for {record.gain}")
            y = int(rng.integers(low_y, high_y + 1))
            x = int(rng.integers(low_x, high_x + 1))
            frame_start = int(rng.integers(0, record.frame_count - self.config.temporal_frames + 1))
            gain = record.gain
            noise_seed = None
        else:
            source = "synthetic"
            record_index = (sample_index - real_seen) % len(self.synthetic)
            gain = self.synthetic[record_index].gain
            y = int(rng.integers(0, _PACKED_HEIGHT - patch + 1))
            x = int(rng.integers(0, _PACKED_WIDTH - patch + 1))
            frame_start = None
            noise_seed = int(np.random.SeedSequence(
                [self.config.seed, epoch, position, 12, 1],
            ).generate_state(1, dtype=np.uint64)[0])

        return A12SamplePlan(sample_index, epoch, position, source, record_index,
                             gain, y, x, frame_start, geometry_seed, noise_seed)

    def __getitem__(self, sample_index: int) -> tuple[Tensor, Tensor, Tensor]:
        plan = self.sample_plan(sample_index)
        if plan.source == "real":
            arrays = self._real_sample(self.real[plan.record_index], np.random.default_rng(plan.geometry_seed))
        else:
            patch = self.config.patch_size
            y, x = plan.crop_y, plan.crop_x
            clean = self.clean_reference[:, y : y + patch, x : x + patch]
            fpn = self.fpn_offset[:, y : y + patch, x : x + patch]
            rng = np.random.default_rng(plan.noise_seed)
            noisy = np.stack([
                generate_noisy_frame(clean, self.noise_model, plan.gain, self.black_levels, fpn, rng)
                for _ in range(self.config.temporal_frames)
            ])
            inputs = _normalise_dn(noisy, self.reference_black)
            target_frame = self.clean_normalised[:, y : y + patch, x : x + patch]
            targets = np.broadcast_to(target_frame, inputs.shape).copy()
            valid = self.valid_region[y : y + patch, x : x + patch].copy()
            threshold = ((3800.0 - self.reference_black) / (_WHITE_LEVEL - self.reference_black))[:, None, None]
            valid &= np.all(inputs < threshold, axis=(0, 1))
            arrays = inputs, targets, np.repeat(valid[None], self.config.temporal_frames, axis=0)

        return tuple(torch.from_numpy(array.astype(np.float32, copy=False)) for array in arrays)


class A12StepSampler(Sampler[int]):
    """Resume at the next unconsumed batch, regardless of earlier prefetching.

    Pass the number of completed updates as start_step. The DataLoader must use
    the same batch_size as dataset.config and must not shuffle or drop samples.
    Increasing config.steps extends a run without changing earlier sample IDs.
    """

    def __init__(self, dataset: A12SamplingDataset, start_step: int = 0) -> None:
        start_step = operator.index(start_step)
        if not 0 <= start_step <= dataset.config.steps:
            raise ValueError("start_step must be between zero and total configured steps")
        self.start_sample = start_step * dataset.config.batch_size
        self.stop_sample = len(dataset)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start_sample, self.stop_sample))

    def __len__(self) -> int:
        return self.stop_sample - self.start_sample

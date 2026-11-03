"""Paired sample-refresh ablation with an identical repeating source schedule."""

from dataclasses import asdict, replace
import hashlib
import json
import operator
from pathlib import Path
from typing import Literal

import numpy as np

from raw_fusion.finetune.a3 import A3Config, _PACKED_HEIGHT, _PACKED_WIDTH
from raw_fusion.finetune.a12_sampling import A12SamplePlan, A12SamplingDataset


def paired_sampling_config_identity(config: A3Config) -> dict:
    """Identify data and sampling settings independently of run location."""
    fields = ("seed", "temporal_frames", "patch_size", "batch_size", "real_fraction",
              "holdout_gain", "synthetic_temporal_variants")
    identity = {name: getattr(config, name) for name in fields}
    for name in ("dataset_root", "supervision_dir", "noise_dir"):
        identity[name] = str(Path(getattr(config, name)).resolve())
    return identity


class A12PairedSamplingDataset(A12SamplingDataset):
    """Repeat source/record/gain identities while isolating sample refresh.

    Both modes preserve the original A12 first cycle. Fixed mode reuses its
    complete sample identities. Refresh mode changes only geometry and noise
    seeds in later cycles, retaining the first cycle's source schedule.
    """

    def __init__(
        self, config: A3Config, sampling_mode: Literal["fixed", "refresh"],
        samples_per_epoch: int = 512, plan_overrides_path: Path | None = None,
    ) -> None:
        if sampling_mode not in {"fixed", "refresh"}:
            raise ValueError("sampling_mode must be fixed or refresh")
        self.sampling_mode = sampling_mode
        super().__init__(config, samples_per_epoch=samples_per_epoch)
        self._plan_overrides: dict[int, A12SamplePlan] = {}
        if plan_overrides_path is not None:
            manifest = json.loads(Path(plan_overrides_path).read_text(encoding="utf-8"))
            if manifest.get("schema") != "a12_empty_plan_overrides_v1":
                raise ValueError("unknown empty-plan override schema")
            if manifest.get("status") != "resolved":
                raise ValueError("empty-plan override manifest is not resolved")
            if manifest["sampling_config"] != paired_sampling_config_identity(config):
                raise ValueError("empty-plan overrides have different sampling configuration")
            if manifest["samples_per_epoch"] != samples_per_epoch:
                raise ValueError("empty-plan overrides have a different cycle length")
            for key in ("source_audit", "source_sample_manifest"):
                source_path = Path(manifest[key])
                if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest[f"{key}_sha256"]:
                    raise ValueError(f"empty-plan override source changed: {key}")
            audit = json.loads(Path(manifest["source_audit"]).read_text(encoding="utf-8"))
            source_samples = json.loads(Path(manifest["source_sample_manifest"]).read_text(encoding="utf-8"))
            if audit["sample_manifest_sha256"] != manifest["source_sample_manifest_sha256"]:
                raise ValueError("empty-plan audit and sample manifest identities differ")
            if audit["coverage"]["fixed"]["zero_valid_sample_indices"]:
                raise ValueError("this override protocol cannot alter the fixed group")
            indices = [entry["sample_index"] for entry in manifest["overrides"]]
            if len(set(indices)) != len(indices) or sorted(indices) != sorted(audit["coverage"]["refresh"]["zero_valid_sample_indices"]):
                raise ValueError("overrides must cover exactly the audited refresh empty samples")
            validation_dataset = self
            if self.samples < samples_per_epoch:
                validation_config = replace(config, steps=(samples_per_epoch + config.batch_size - 1) // config.batch_size)
                validation_dataset = A12PairedSamplingDataset(validation_config, "refresh", samples_per_epoch)
            for entry in manifest["overrides"]:
                index = entry["sample_index"]
                original = validation_dataset._unmodified_plan(index, "refresh")
                audited = source_samples["modes"]["refresh"][index]
                if (entry["sampling_mode"] != "refresh" or original.source != "real"
                        or not samples_per_epoch <= index < 2 * samples_per_epoch):
                    raise ValueError("only second-cycle refresh real samples can be overridden")
                if (audited["valid_fraction"] != 0.0 or entry["original_valid_fraction"] != 0.0
                        or audited["plan"] != asdict(original) or entry["original_plan"] != asdict(original)
                        or entry["original_tensor_sha256"] != audited["tensor_sha256"]):
                    raise ValueError("override original sample is not the audited empty plan")
                replacement = A12SamplePlan(**entry["replacement_plan"])
                immutable = ("source", "record_index", "gain", "sample_index", "epoch", "epoch_index", "noise_seed")
                if any(getattr(replacement, key) != getattr(original, key) for key in immutable):
                    raise ValueError("override changes source identity or sample position")
                attempt = entry["accepted_attempt"]
                if not isinstance(attempt, int) or not 0 <= attempt < 64:
                    raise ValueError("override accepted attempt is outside the finite search budget")
                expected_seed = int(np.random.SeedSequence(
                    [config.seed, original.epoch, original.epoch_index, 12, 2, attempt],
                ).generate_state(1, dtype=np.uint64)[0])
                if replacement.geometry_seed != expected_seed:
                    raise ValueError("override seed does not match the deterministic candidate sequence")
                record = self.real[replacement.record_index]
                rng = np.random.default_rng(replacement.geometry_seed)
                patch = config.patch_size
                y = int(rng.integers(max(0, -record.shift_y), min(_PACKED_HEIGHT - patch, _PACKED_HEIGHT - patch - record.shift_y) + 1))
                x = int(rng.integers(max(0, -record.shift_x), min(_PACKED_WIDTH - patch, _PACKED_WIDTH - patch - record.shift_x) + 1))
                frame_start = int(rng.integers(0, record.frame_count - config.temporal_frames + 1))
                if (replacement.crop_y, replacement.crop_x, replacement.frame_start) != (y, x, frame_start):
                    raise ValueError("override geometry does not match its recorded seed")
                if not 0.0 < entry["replacement_valid_fraction"] <= 1.0:
                    raise ValueError("override replacement must have nonempty audited supervision")
                if sampling_mode == "refresh":
                    self._plan_overrides[index] = replacement

    def sample_plan(self, sample_index: int) -> A12SamplePlan:
        sample_index = operator.index(sample_index)
        if not 0 <= sample_index < self.samples:
            raise IndexError(f"sample_index {sample_index} outside [0, {self.samples})")
        if sample_index in self._plan_overrides:
            return self._plan_overrides[sample_index]
        else:
            return self._unmodified_plan(sample_index, self.sampling_mode)

    def _unmodified_plan(self, sample_index: int, sampling_mode: str) -> A12SamplePlan:
        epoch, position = divmod(sample_index, self.samples_per_epoch)
        template = super().sample_plan(position)
        if sampling_mode == "fixed" or epoch == 0:
            return replace(template, sample_index=sample_index, epoch=epoch)
        else:
            geometry_seed = int(np.random.SeedSequence(
                [self.config.seed, epoch, position, 12, 0],
            ).generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(geometry_seed)
            patch = self.config.patch_size
            if template.source == "real":
                record = self.real[template.record_index]
                low_y = max(0, -record.shift_y)
                high_y = min(_PACKED_HEIGHT - patch, _PACKED_HEIGHT - patch - record.shift_y)
                low_x = max(0, -record.shift_x)
                high_x = min(_PACKED_WIDTH - patch, _PACKED_WIDTH - patch - record.shift_x)
                if high_y < low_y or high_x < low_x:
                    raise ValueError(f"registration shift leaves no patch overlap for {record.gain}")
                crop_y = int(rng.integers(low_y, high_y + 1))
                crop_x = int(rng.integers(low_x, high_x + 1))
                frame_start = int(rng.integers(0, record.frame_count - self.config.temporal_frames + 1))
                noise_seed = None
            else:
                crop_y = int(rng.integers(0, _PACKED_HEIGHT - patch + 1))
                crop_x = int(rng.integers(0, _PACKED_WIDTH - patch + 1))
                frame_start = None
                noise_seed = int(np.random.SeedSequence(
                    [self.config.seed, epoch, position, 12, 1],
                ).generate_state(1, dtype=np.uint64)[0])
            return replace(template, sample_index=sample_index, epoch=epoch,
                           crop_y=crop_y, crop_x=crop_x, frame_start=frame_start,
                           geometry_seed=geometry_seed, noise_seed=noise_seed)

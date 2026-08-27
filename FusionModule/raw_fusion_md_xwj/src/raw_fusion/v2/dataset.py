"""Dataset/split value objects and ancestry checks for V2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from collections.abc import Mapping, Sequence

import numpy as np

from .artifacts import sha256_file
from .schemas.common import ContractError, expect_number, expect_string
from .schemas.dataset import validate_dataset_v2, validate_split_v2


@dataclass(frozen=True, slots=True)
class RawAssetV2:
    path: Path
    sha256: str
    frame_count: int


@dataclass(frozen=True, slots=True)
class TargetSequenceV2:
    noisy: RawAssetV2
    denoised: RawAssetV2
    fused: RawAssetV2
    white_balance_rgb: tuple[float, float, float]
    isp_gain: float


@dataclass(frozen=True, slots=True)
class SupportSequenceV2:
    noisy: RawAssetV2
    usable_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DatasetV2:
    raw_contract: Mapping[str, object]
    targets: Mapping[str, TargetSequenceV2]
    support: Mapping[str, SupportSequenceV2]
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SplitPartV2:
    pre_roll_frames: tuple[int, ...]
    target_frames: tuple[int, ...]
    post_guard_frames: tuple[int, ...]
    references: Mapping[str, tuple[int, int]]

    @property
    def guard_frames(self) -> tuple[int, ...]:
        return self.post_guard_frames

    @property
    def metric_frames(self) -> tuple[int, ...]:
        return self.target_frames


@dataclass(frozen=True, slots=True)
class SplitV2:
    train: SplitPartV2
    validation: SplitPartV2
    audit_rois: Mapping[str, Mapping[str, tuple[int, int]]]
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TargetAncestryV2:
    source_frame: int
    history_frames: tuple[int, ...]
    query_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SplitAncestryPartV2:
    target_frames: tuple[int, ...]
    preroll_frames: tuple[int, ...]
    guard_frames: tuple[int, ...]
    metric_frames: tuple[int, ...]
    references: Mapping[str, tuple[int, int]]
    target_ancestry: tuple[TargetAncestryV2, ...]

    @property
    def post_guard_frames(self) -> tuple[int, ...]:
        return self.guard_frames


@dataclass(frozen=True, slots=True)
class SplitAncestryV2:
    train: SplitAncestryPartV2
    validation: SplitAncestryPartV2
    all_loss_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PackedRoi:
    top: int
    left: int
    height: int
    width: int


def _asset(mapping: Mapping[str, object], context: str) -> RawAssetV2:
    path = Path(expect_string(mapping["path"], f"{context}.path"))
    digest = expect_string(mapping["sha256"], f"{context}.sha256")
    count = int(mapping["frame_count"])
    return RawAssetV2(path=path, sha256=digest, frame_count=count)


def dataset_from_mapping(mapping: Mapping[str, object], *, manifest_path: Path | None = None) -> DatasetV2:
    validate_dataset_v2(mapping)
    targets: dict[str, TargetSequenceV2] = {}
    for condition, value in mapping["target_sequences"].items():
        targets[condition] = TargetSequenceV2(
            noisy=_asset(value["noisy"], f"DatasetV2.target_sequences.{condition}.noisy"),
            denoised=_asset(value["denoised"], f"DatasetV2.target_sequences.{condition}.denoised"),
            fused=_asset(value["fused"], f"DatasetV2.target_sequences.{condition}.fused"),
            white_balance_rgb=tuple(float(x) for x in value["preview"]["white_balance_rgb"]),
            isp_gain=float(value["preview"]["isp_gain"]),
        )
    support: dict[str, SupportSequenceV2] = {}
    for condition, value in mapping["support_sequences"].items():
        start, end = value["used_frame_range"]
        support[condition] = SupportSequenceV2(
            noisy=_asset(value["noisy"], f"DatasetV2.support_sequences.{condition}.noisy"),
            usable_frames=tuple(range(start, end + 1)),
        )
    return DatasetV2(raw_contract=mapping["raw_contract"], targets=targets, support=support, manifest_path=manifest_path)


def _resolve_asset(asset: RawAssetV2, owner: Path, allowed_root: Path) -> RawAssetV2:
    root = allowed_root.resolve()
    candidate = (owner.parent / asset.path).resolve()
    # Dataset source paths are rooted at the declared data/workspace root,
    # while derived manifests remain owner-relative.  Prefer owner-relative
    # resolution, then accept the explicit source-root-relative form.
    if not candidate.is_file() and not asset.path.is_absolute():
        candidate = (root / asset.path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ContractError(f"dataset asset outside allowed root: {asset.path}") from error
    if not candidate.is_file():
        raise ContractError(f"dataset asset missing: {candidate}")
    if sha256_file(candidate) != asset.sha256:
        raise ContractError(f"dataset asset SHA-256 mismatch: {candidate}")
    return RawAssetV2(candidate, asset.sha256, asset.frame_count)


def load_dataset_v2(path: Path, *, allowed_root: Path | None = None, validate_assets: bool = False) -> DatasetV2:
    from .artifacts import load_json_object

    manifest = Path(path).resolve()
    mapping = load_json_object(manifest)
    dataset = dataset_from_mapping(mapping, manifest_path=manifest)
    root = Path(allowed_root).resolve() if allowed_root is not None else manifest.parents[2]
    targets = {
        key: TargetSequenceV2(
            noisy=_resolve_asset(value.noisy, manifest, root),
            denoised=_resolve_asset(value.denoised, manifest, root),
            fused=_resolve_asset(value.fused, manifest, root),
            white_balance_rgb=value.white_balance_rgb,
            isp_gain=value.isp_gain,
        )
        for key, value in dataset.targets.items()
    }
    support = {
        key: SupportSequenceV2(_resolve_asset(value.noisy, manifest, root), value.usable_frames)
        for key, value in dataset.support.items()
    }
    loaded = DatasetV2(dataset.raw_contract, targets, support, manifest)
    if validate_assets:
        validate_dataset_assets(loaded)
    return loaded


def validate_dataset_assets(dataset: DatasetV2) -> None:
    width = int(dataset.raw_contract["sensor_width"])
    height = int(dataset.raw_contract["sensor_height"])
    expected_per_frame = width * height * 2
    signals = dataset.raw_contract["signals"]
    for condition, sequence in dataset.targets.items():
        for name, asset in (("noisy", sequence.noisy), ("denoised", sequence.denoised), ("fused", sequence.fused)):
            if asset.path.stat().st_size != expected_per_frame * asset.frame_count:
                raise ContractError(f"{condition}.{name}: byte size does not match frame_count")
            values = np.memmap(asset.path, dtype="<u2", mode="r", shape=(asset.frame_count, height, width))
            maximum = int(np.max(values >> int(signals[name]["right_shift"])))
            if maximum > int(dataset.raw_contract["white_level"]):
                raise ContractError(f"{condition}.{name}: post-shift sample exceeds white level")
    for condition, sequence in dataset.support.items():
        if sequence.noisy.path.stat().st_size != expected_per_frame * sequence.noisy.frame_count:
            raise ContractError(f"{condition}: support byte size does not match frame_count")
        if sequence.usable_frames != tuple(range(200)):
            raise ContractError(f"{condition}: support usable range must be 0..199")


def split_from_mapping(mapping: Mapping[str, object], *, manifest_path: Path | None = None) -> SplitV2:
    validate_split_v2(mapping)

    def part(value: Mapping[str, object]) -> SplitPartV2:
        refs = {key: tuple(value["references"][key]) for key in ("g", "e", "v")}
        def inclusive(pair: Sequence[int]) -> tuple[int, ...]:
            start, end = (int(pair[0]), int(pair[1]))
            return tuple(range(start, end + 1))
        return SplitPartV2(
            inclusive(value["pre_roll_frames"]), inclusive(value["target_frames"]),
            inclusive(value["post_guard_frames"]), refs,
        )

    return SplitV2(
        train=part(mapping["train"]), validation=part(mapping["validation"]),
        audit_rois={key: {axis: tuple(value[axis]) for axis in ("x_half_open", "y_half_open")} for key, value in mapping["audit_rois"].items()},
        manifest_path=manifest_path,
    )


def load_split_v2(path: Path) -> SplitV2:
    from .artifacts import load_json_object

    manifest = Path(path).resolve()
    return split_from_mapping(load_json_object(manifest), manifest_path=manifest)


def label_query_frames(frame: int) -> tuple[int, ...]:
    frame = int(frame)
    if not 0 <= frame <= 199:
        raise ContractError(f"source frame out of range: {frame}")
    return tuple(range(max(0, frame - 2), min(199, frame + 2) + 1))


def _ancestry_part(part: SplitPartV2) -> SplitAncestryPartV2:
    targets = tuple(part.target_frames)
    return SplitAncestryPartV2(
        target_frames=targets,
        preroll_frames=part.pre_roll_frames,
        guard_frames=part.post_guard_frames,
        metric_frames=targets,
        references=part.references,
        target_ancestry=tuple(TargetAncestryV2(frame, (frame - 1, frame), label_query_frames(frame)) for frame in targets),
    )


def expand_split_ancestry(split: SplitV2) -> SplitAncestryV2:
    train = _ancestry_part(split.train)
    validation = _ancestry_part(split.validation)
    train_ref = set(range(128, 176))
    for item in validation.target_ancestry:
        overlap = train_ref.intersection(item.query_frames)
        if overlap:
            raise ContractError(f"frame {min(overlap)} validation query crosses train reference")
    return SplitAncestryV2(train=train, validation=validation, all_loss_frames=tuple(train.target_frames + validation.target_frames))


def expand_audit_rois_to_cells(rois: Sequence[PackedRoi], *, cell_size: int = 32) -> set[tuple[int, int]]:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    cells: set[tuple[int, int]] = set()
    for roi in rois:
        if roi.height <= 0 or roi.width <= 0:
            continue
        for y in range(roi.top // cell_size, (roi.top + roi.height - 1) // cell_size + 1):
            for x in range(roi.left // cell_size, (roi.left + roi.width - 1) // cell_size + 1):
                cells.add((y, x))
    return cells

"""Current-data reader and training sample adapter for data-first V2."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .data_first_contracts import DataFirstInputBatch, derive_input_condition
from .data_first_supervision import DataFirstSupervision, MOG2SupervisionConfig, MOG2SupervisionGenerator
from .dataset import DatasetV2, SplitV2, load_dataset_v2, load_split_v2
from .raw import normalize_signal, pack_rggb_v2
from .schemas.common import ContractError


NETWORK_INPUT_SIZE = 320
CELL_SIZE = 32
PACKED_SHAPE = (540, 960)


@dataclass(frozen=True, slots=True)
class DataFirstSampleRow:
    condition: str
    split: str
    source_frame: int
    cell_y: int
    cell_x: int


@dataclass(frozen=True, slots=True)
class DataFirstSample:
    model_inputs: DataFirstInputBatch
    supervision: DataFirstSupervision
    row: DataFirstSampleRow


class DataFirstDataset:
    def __init__(
        self,
        frame_provider,
        *,
        supervision_config: MOG2SupervisionConfig,
        target_frames: Mapping[str, tuple[int, ...]] | None = None,
        md_mask_provider=None,
        shape: tuple[int, int] = PACKED_SHAPE,
    ) -> None:
        if shape != PACKED_SHAPE:
            raise ContractError("data-first dataset requires packed 540x960 frames")
        self._provider = frame_provider
        self._supervision = MOG2SupervisionGenerator(frame_provider, config=supervision_config)
        self._md_mask_provider = md_mask_provider
        self.target_frames = dict(target_frames or {"128x": tuple(range(58, 94)), "645x": tuple(range(58, 94))})
        self.cell_shape = (17, 30)
        self.shape = shape

    @classmethod
    def from_arrays(
        cls,
        frames: Mapping[tuple[str, str, int], np.ndarray],
        *,
        supervision_config: MOG2SupervisionConfig,
        target_frames: Mapping[str, tuple[int, ...]] | None = None,
        md_mask_provider=None,
    ) -> "DataFirstDataset":
        def provider(condition: str, frame: int) -> np.ndarray:
            try:
                return frames[(condition, "denoised", int(frame))]
            except KeyError as error:
                raise ContractError(f"missing denoised frame: {condition}/{frame}") from error

        def read(condition: str, kind: str, frame: int) -> np.ndarray:
            try:
                return np.asarray(frames[(condition, kind, int(frame))])
            except KeyError as error:
                raise ContractError(f"missing {kind} frame: {condition}/{frame}") from error

        instance = cls(provider, supervision_config=supervision_config, target_frames=target_frames, md_mask_provider=md_mask_provider)
        instance._read = read  # type: ignore[attr-defined]
        return instance

    @classmethod
    def from_paths(
        cls,
        dataset_path: Path,
        split_path: Path,
        supervision_config: MOG2SupervisionConfig,
        *,
        allowed_root: Path = Path("/data1/wangzepu/Jaime"),
        mog2_cache: Path | None = None,
    ) -> "DataFirstDataset":
        dataset = load_dataset_v2(Path(dataset_path), allowed_root=allowed_root, validate_assets=False)
        split = load_split_v2(Path(split_path))
        signals = dataset.raw_contract["signals"]
        height = int(dataset.raw_contract["sensor_height"])
        width = int(dataset.raw_contract["sensor_width"])

        def read(condition: str, kind: str, frame: int) -> np.ndarray:
            if condition not in dataset.targets:
                raise ContractError(f"unknown target condition: {condition}")
            sequence = dataset.targets[condition]
            asset = getattr(sequence, kind)
            if frame < 0 or frame >= asset.frame_count:
                raise ContractError(f"frame {frame} is outside {condition}/{kind}")
            mosaic = np.memmap(asset.path, dtype="<u2", mode="r", shape=(asset.frame_count, height, width))[frame]
            packed = pack_rggb_v2(np.right_shift(mosaic, int(signals[kind]["right_shift"])).astype(np.uint16))
            return np.ascontiguousarray(packed)

        def provider(condition: str, frame: int) -> np.ndarray:
            return read(condition, "denoised", frame)

        target_frames = {
            "128x": tuple(split.train.target_frames),
            "645x": tuple(split.train.target_frames),
        }
        md_mask_provider = None
        if mog2_cache is not None:
            from .data_first_mog2_cache import MOG2MaskCache
            cache = MOG2MaskCache.open(mog2_cache)
            expected_hashes = {condition: dataset.targets[condition].denoised.sha256 for condition in target_frames}
            if cache.manifest.get("source_sha256") != expected_hashes or cache.manifest.get("target_frames") != {condition: list(frames) for condition, frames in target_frames.items()}:
                raise ContractError("MOG2 cache does not match dataset or training split")
            md_mask_provider = cache.read_mask
        instance = cls(provider, supervision_config=supervision_config, target_frames=target_frames, md_mask_provider=md_mask_provider)
        instance._read = read  # type: ignore[attr-defined]
        return instance

    def _read(self, condition: str, kind: str, frame: int) -> np.ndarray:
        raise ContractError("data-first frame reader is not configured")

    def _crop(self, value: np.ndarray, cell_y: int, cell_x: int) -> np.ndarray:
        if value.shape != (4, *PACKED_SHAPE):
            raise ContractError("data-first packed frame must have shape [4,540,960]")
        if not (0 <= cell_y < self.cell_shape[0] and 0 <= cell_x < self.cell_shape[1]):
            raise ContractError("data-first cell coordinates are outside the real cell grid")
        origin_y = min(max(cell_y * CELL_SIZE - 64, 0), PACKED_SHAPE[0] - NETWORK_INPUT_SIZE)
        origin_x = min(max(cell_x * CELL_SIZE - 64, 0), PACKED_SHAPE[1] - NETWORK_INPUT_SIZE)
        return np.ascontiguousarray(value[:, origin_y : origin_y + NETWORK_INPUT_SIZE, origin_x : origin_x + NETWORK_INPUT_SIZE])

    def sample(self, row: DataFirstSampleRow) -> DataFirstSample:
        if row.condition not in self.target_frames or row.source_frame not in self.target_frames[row.condition]:
            raise ContractError("data-first sample frame is not in the selected split")
        if row.source_frame < 2:
            raise ContractError("data-first sample requires two previous frames")
        prev = self._crop(self._read(row.condition, "noisy", row.source_frame - 1), row.cell_y, row.cell_x)
        curr = self._crop(self._read(row.condition, "noisy", row.source_frame), row.cell_y, row.cell_x)
        denoised = self._crop(self._read(row.condition, "denoised", row.source_frame), row.cell_y, row.cell_x)
        fused = self._crop(self._read(row.condition, "fused", row.source_frame), row.cell_y, row.cell_x)
        image_tensors = {
            name: torch.from_numpy(normalize_signal(value, 252 if "noisy" in name else 300, 4095)).unsqueeze(0)
            for name, value in (("prev_noisy", prev), ("curr_noisy", curr), ("denoised", denoised), ("fused", fused))
        }
        condition = derive_input_condition(**image_tensors)
        inputs = DataFirstInputBatch(**image_tensors, c_tilde=condition)
        supervision = self._supervision.supervise_from_mask(row.condition, row.source_frame, self._md_mask_provider(row.condition, row.source_frame)) if self._md_mask_provider is not None else self._supervision.supervise(row.condition, row.source_frame)
        return DataFirstSample(inputs, supervision, row)

    def policy_alpha_class(self, condition: str, frame: int) -> np.ndarray:
        """Return cached-or-online policy classes for deterministic sampling."""
        supervision = self._supervision.supervise_from_mask(condition, frame, self._md_mask_provider(condition, frame)) if self._md_mask_provider is not None else self._supervision.supervise(condition, frame)
        return supervision.policy_alpha_class[0]


__all__ = ["DataFirstDataset", "DataFirstSample", "DataFirstSampleRow"]

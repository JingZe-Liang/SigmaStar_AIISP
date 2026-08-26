"""Parallel, resumable MOG2 supervision cache generation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Mapping

import cv2
import numpy as np

from .dataset import load_dataset_v2, load_split_v2
from .md import Mog2ConfigV2, create_mog2
from .schemas.common import ContractError


_CACHE_PROTOCOL = "raw_fusion_v2_data_first_mog2_cache"
_WORKER_ASSETS: dict[str, tuple[Path, int, int, int, int]] = {}


@dataclass(frozen=True, slots=True)
class MOG2CacheResult:
    root: Path
    completed: int
    total: int
    elapsed_seconds: float


class MOG2MaskCache:
    def __init__(self, root: Path, manifest: Mapping[str, object]) -> None:
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.progress_path = self.root / "progress.jsonl"

    @classmethod
    def create(cls, root: Path, *, source_sha256: Mapping[str, str], target_frames: Mapping[str, tuple[int, ...]], mog2_config: Mapping[str, object]) -> "MOG2MaskCache":
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "protocol": _CACHE_PROTOCOL,
            "source_sha256": dict(source_sha256),
            "target_frames": {name: list(frames) for name, frames in target_frames.items()},
            "mog2_config": dict(mog2_config),
        }
        path = destination / "manifest.json"
        if path.exists():
            existing = cls.open(destination)
            if existing.manifest != manifest:
                raise ContractError("existing MOG2 cache manifest is incompatible")
            return existing
        path.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
        return cls(destination, manifest)

    @classmethod
    def open(cls, root: Path) -> "MOG2MaskCache":
        path = Path(root) / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="ascii"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ContractError("cannot read MOG2 cache manifest") from error
        if not isinstance(manifest, dict) or manifest.get("protocol") != _CACHE_PROTOCOL:
            raise ContractError("cache is not a data-first MOG2 cache")
        return cls(Path(root), manifest)

    def _path(self, condition: str, frame: int) -> Path:
        return self.root / condition / f"frame_{int(frame):04d}.npy"

    def is_complete(self, condition: str, frame: int) -> bool:
        path = self._path(condition, frame)
        if not path.is_file():
            return False
        try:
            value = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            return False
        return value.shape == (540, 960) and value.dtype == np.uint8

    def read_mask(self, condition: str, frame: int) -> np.ndarray:
        path = self._path(condition, frame)
        if not self.is_complete(condition, frame):
            raise ContractError(f"MOG2 cache mask is missing or invalid: {condition}/{frame}")
        return np.ascontiguousarray(np.load(path), dtype=np.uint8)

    def write_mask(self, condition: str, frame: int, mask: np.ndarray) -> Path:
        value = np.asarray(mask, dtype=np.uint8)
        if value.shape != (540, 960):
            raise ContractError("MOG2 cache masks must be uint8 [540,960]")
        destination = self._path(condition, frame)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(value))
        temporary.replace(destination)
        return destination

    def write_progress(self, *, completed: int, total: int, elapsed_seconds: float) -> None:
        rate = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        record = {
            "completed": completed,
            "total": total,
            "elapsed_seconds": elapsed_seconds,
            "tasks_per_second": rate,
            "eta_seconds": (total - completed) / rate if rate > 0.0 else None,
        }
        with self.progress_path.open("a", encoding="ascii") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            stream.flush()


def _initialize_worker(assets: Mapping[str, tuple[str, int, int, int, int]]) -> None:
    global _WORKER_ASSETS
    cv2.setNumThreads(1)
    _WORKER_ASSETS = {name: (Path(path), frames, height, width, shift) for name, (path, frames, height, width, shift) in assets.items()}


def _green_frame(condition: str, frame: int) -> np.ndarray:
    path, count, height, width, shift = _WORKER_ASSETS[condition]
    source = np.memmap(path, dtype="<u2", mode="r", shape=(count, height, width))[frame]
    gr = np.right_shift(source[0::2, 1::2], shift).astype(np.uint32)
    gb = np.right_shift(source[1::2, 0::2], shift).astype(np.uint32)
    return np.ascontiguousarray(((gr + gb) // 2).astype(np.float32))


def _generate_one(task: tuple[str, int, int, float, bool]) -> tuple[str, int, np.ndarray, float]:
    condition, frame, history, threshold, shadows = task
    started = time.monotonic()
    subtractor = create_mog2(Mog2ConfigV2(history=history, var_threshold=threshold, detect_shadows=shadows))
    for index in range(50):
        subtractor.apply(_green_frame(condition, index), learningRate=-1.0)
    raw_mask = subtractor.apply(_green_frame(condition, frame), learningRate=0.0)
    mask = cv2.medianBlur(np.ascontiguousarray(raw_mask, dtype=np.uint8), 3)
    return condition, frame, np.ascontiguousarray(mask), time.monotonic() - started


def generate_mog2_cache(dataset_path: Path, split_path: Path, output_dir: Path, *, workers: int | None = None, allowed_root: Path = Path("/data1/wangzepu/Jaime")) -> MOG2CacheResult:
    dataset = load_dataset_v2(dataset_path, allowed_root=allowed_root, validate_assets=False)
    split = load_split_v2(split_path)
    config = Mog2ConfigV2(history=50)
    targets = {condition: tuple(split.train.target_frames) for condition in ("128x", "645x")}
    cache = MOG2MaskCache.create(
        output_dir,
        source_sha256={condition: dataset.targets[condition].denoised.sha256 for condition in targets},
        target_frames=targets,
        mog2_config={"history": config.history, "var_threshold": config.var_threshold, "detect_shadows": config.detect_shadows},
    )
    raw = dataset.raw_contract
    assets = {
        condition: (str(dataset.targets[condition].denoised.path), dataset.targets[condition].denoised.frame_count, int(raw["sensor_height"]), int(raw["sensor_width"]), int(raw["signals"]["denoised"]["right_shift"]))
        for condition in targets
    }
    pending = [(condition, frame, config.history, config.var_threshold, config.detect_shadows) for condition, frames in targets.items() for frame in frames if not cache.is_complete(condition, frame)]
    total = sum(len(frames) for frames in targets.values())
    completed = total - len(pending)
    started = time.monotonic()
    if pending:
        count = int(workers) if workers is not None else min(32, max(1, os.cpu_count() or 1))
        if count <= 0:
            raise ContractError("MOG2 cache workers must be positive")
        with ProcessPoolExecutor(max_workers=min(count, len(pending)), initializer=_initialize_worker, initargs=(assets,)) as pool:
            futures = [pool.submit(_generate_one, task) for task in pending]
            for future in as_completed(futures):
                condition, frame, mask, _worker_seconds = future.result()
                cache.write_mask(condition, frame, mask)
                completed += 1
                cache.write_progress(completed=completed, total=total, elapsed_seconds=time.monotonic() - started)
    return MOG2CacheResult(Path(output_dir), completed, total, time.monotonic() - started)


__all__ = ["MOG2CacheResult", "MOG2MaskCache", "generate_mog2_cache"]

"""MD-free inference for the data-first V2 checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .bands import b2
from .data_first_checkpoint import load_data_first_checkpoint
from .data_first_contracts import DataFirstInputBatch, derive_input_condition
from .data_first_dataset import DataFirstDataset
from .data_first_fusion import limit_q_to_raw_range
from .model import FrequencyFusionConfigV2, FrequencyFusionCore
from .selector import select_q
from .schemas.common import ContractError


@dataclass(frozen=True, slots=True)
class DataFirstInferenceResult:
    output_dir: Path
    frames: int
    fallback_fraction: float


def _input(dataset: DataFirstDataset, condition: str, frame: int) -> tuple[DataFirstInputBatch, np.ndarray]:
    prev = dataset._crop(dataset._read(condition, "noisy", frame - 1), 4, 5)
    curr = dataset._crop(dataset._read(condition, "noisy", frame), 4, 5)
    denoised = dataset._crop(dataset._read(condition, "denoised", frame), 4, 5)
    fused = dataset._crop(dataset._read(condition, "fused", frame), 4, 5)
    tensors = {
        name: torch.from_numpy((value.astype(np.float32) - (252.0 if "noisy" in name else 300.0)) / 3795.0).unsqueeze(0)
        for name, value in (("prev_noisy", prev), ("curr_noisy", curr), ("denoised", denoised), ("fused", fused))
    }
    return DataFirstInputBatch(**tensors, c_tilde=derive_input_condition(**tensors)), denoised


def run_data_first_inference(
    checkpoint: Path,
    dataset: DataFirstDataset,
    output_dir: Path,
    *,
    conditions: tuple[str, ...] = ("128x", "645x"),
    max_frames: int | None = None,
) -> DataFirstInferenceResult:
    restored = load_data_first_checkpoint(checkpoint)
    model = FrequencyFusionCore(FrequencyFusionConfigV2.production()).eval()
    model.load_state_dict(restored.payload["model_state_dict"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    fallback = 0
    manifest: dict[str, object] = {"protocol": "raw_fusion_v2_data_first", "conditions": {}}
    for condition in conditions:
        frames = tuple(dataset.target_frames[condition])
        if max_frames is not None:
            frames = frames[: int(max_frames)]
        outputs: list[np.ndarray] = []
        traces: list[dict[str, object]] = []
        for frame in frames:
            original = dataset._read(condition, "denoised", frame)
            output = original.copy()
            source = "model"
            try:
                inputs, denoised_crop = _input(dataset, condition, frame)
                with torch.inference_mode():
                    logits = model.forward_data_first(inputs).q_logits_pixel_core
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("q logits are non-finite")
                    selected = select_q(logits, torch.ones((logits.shape[0], 1, *logits.shape[-2:]), dtype=torch.bool))
                    safe_q = limit_q_to_raw_range(inputs.denoised[..., 32:-32, 32:-32], b2(inputs.fused[..., 32:-32, 32:-32] - inputs.denoised[..., 32:-32, 32:-32]), selected.q)
                    q_map = safe_q[0, 0].numpy()
                    q_classes, q_counts = np.unique(selected.class_index[0].numpy(), return_counts=True)
                    delta = b2(inputs.fused[..., 32:-32, 32:-32] - inputs.denoised[..., 32:-32, 32:-32])[0].numpy()
                candidate = denoised_crop[:, 32:-32, 32:-32].astype(np.float32) + np.float32(3795.0) * q_map * delta
                if not np.all(np.isfinite(candidate)) or np.any(candidate < 0.0) or np.any(candidate > 4095.0):
                    raise FloatingPointError("data-first output is outside range")
                origin_y, origin_x = 4 * 32 - 64, 5 * 32 - 64
                output[:, origin_y + 32 : origin_y + 288, origin_x + 32 : origin_x + 288] = np.rint(candidate).astype(np.uint16)
            except (ContractError, FloatingPointError, RuntimeError, ValueError, TypeError, KeyError) as error:
                source = "denoised_bypass"
                fallback += 1
                traces.append({"frame": frame, "output_source": source, "reason": str(error)})
            else:
                traces.append({"frame": frame, "output_source": source, "q_class_counts": {str(int(value)): int(count) for value, count in zip(q_classes, q_counts)}, "q_nonzero_fraction": float(np.mean(q_map > 0.0))})
            outputs.append(output)
            total += 1
        np.save(destination / f"inference_{condition}.npy", np.stack(outputs, axis=0))
        (destination / f"inference_{condition}.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in traces), encoding="ascii")
        manifest["conditions"][condition] = {"frames": len(outputs), "trace": f"inference_{condition}.jsonl"}  # type: ignore[index]
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="ascii")
    return DataFirstInferenceResult(destination, total, fallback / total if total else 0.0)


__all__ = ["DataFirstInferenceResult", "run_data_first_inference"]

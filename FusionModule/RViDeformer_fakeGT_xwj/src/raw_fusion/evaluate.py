"""CPU evaluation for packed RAW fusion models and fixed candidate baselines."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import operator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint import load_checkpoint_strict
from .config import (
    DatasetConfig,
    ExperimentConfig,
    config_fingerprint,
    load_dataset_config,
    load_experiment_config,
)
from .losses import FusionLoss, valid_target_mask
from .metrics import MetricAccumulator, baseline_predictions, compute_frame_metrics
from .model import CausalRawFusionNet, FusionOutput
from .raw import RawFrameDirectoryReader, RawStreamReader, normalize_raw, pack_rggb


_BASELINE_NAMES = frozenset(("denoised", "fused", "average"))
_FRAME_INPUT_NAMES = ("prev_noisy", "curr_noisy", "denoised", "fused", "target")
_DIAGNOSTIC_NAMES = (
    "gate_mean",
    "gate_p10",
    "gate_p50",
    "gate_p90",
    "correction_abs_mean",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One named model configuration and its checkpoint path."""

    name: str
    config_path: Path
    checkpoint_path: Path


def parse_model_specs(specifications: Sequence[str]) -> dict[str, ModelSpec]:
    """Parse repeated ``NAME=CONFIG,CHECKPOINT`` values without duplicate names."""
    parsed: dict[str, ModelSpec] = {}
    for specification in specifications:
        if not isinstance(specification, str):
            raise TypeError("model specification must be a string")
        name, separator, paths = specification.partition("=")
        if not separator or not name or not paths:
            raise ValueError("model must use NAME=CONFIG,CHECKPOINT")
        config, comma, checkpoint = paths.partition(",")
        if not comma or not config or not checkpoint or "," in checkpoint:
            raise ValueError("model must use NAME=CONFIG,CHECKPOINT")
        if name in parsed:
            raise ValueError(f"duplicate model name: {name}")
        parsed[name] = ModelSpec(name, Path(config), Path(checkpoint))
    if not parsed:
        raise ValueError("at least one --model is required")
    return parsed


def validate_saturation_margin_dn(model_margins: Mapping[str, int]) -> int:
    """Return the shared saturation margin or reject model configurations that differ."""
    if not isinstance(model_margins, Mapping) or not model_margins:
        raise ValueError("at least one model saturation_margin_dn is required")
    normalized: dict[str, int] = {}
    for name, value in model_margins.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model margin names must be non-empty strings")
        if isinstance(value, bool):
            raise TypeError("saturation_margin_dn must be an integer")
        try:
            normalized[name] = operator.index(value)
        except TypeError as error:
            raise TypeError("saturation_margin_dn must be an integer") from error
    margins = set(normalized.values())
    if len(margins) != 1:
        details = ", ".join(f"{name}={value}" for name, value in normalized.items())
        raise ValueError(
            "all models must use the same loss.saturation_margin_dn; got " + details
        )
    return next(iter(margins))


def evaluate_models(
    models: Mapping[str, nn.Module],
    frames: Iterable[Mapping[str, Tensor | int]],
    *,
    saturation_threshold: float,
) -> dict[str, object]:
    """Evaluate named models with denoised, fused, and 0.5-average baselines.

    ``frames`` contains normalized packed RAW tensors with shape ``[B, 4, H, W]``.
    Predictions are scored directly in that representation, with no display exposure,
    clipping, or visualization normalization in the metric path.
    """
    threshold = _validate_threshold(saturation_threshold)
    named_models = _validate_models(models)
    method_names = ("denoised", "fused", "average", *named_models)
    accumulators = {name: MetricAccumulator() for name in method_names}
    frame_reports: dict[str, list[dict[str, object]]] = {name: [] for name in method_names}
    diagnostic_frames: dict[str, list[dict[str, object]]] = {
        name: [] for name in named_models
    }
    diagnostic_values: dict[str, list[dict[str, float]]] = {
        name: [] for name in named_models
    }

    previous_training = {name: model.training for name, model in named_models.items()}
    for model in named_models.values():
        model.eval()
    seen_frames = 0
    try:
        with torch.no_grad():
            for position, frame in enumerate(frames):
                inputs, frame_index = _validate_frame(frame, position)
                target = inputs["target"]
                valid_mask = valid_target_mask(target, threshold)
                if not bool(valid_mask.any()):
                    raise ValueError(f"frame {frame_index} has an empty valid target mask")

                predictions = baseline_predictions(inputs["denoised"], inputs["fused"])
                for name, model in named_models.items():
                    output = model(
                        prev_noisy=inputs["prev_noisy"],
                        curr_noisy=inputs["curr_noisy"],
                        denoised=inputs["denoised"],
                        fused=inputs["fused"],
                    )
                    _validate_model_output(name, output, target)
                    predictions[name] = output.prediction
                    values = _frame_diagnostics(output)
                    diagnostic_values[name].append(values)
                    diagnostic_frames[name].append({"frame_index": frame_index, **values})

                for name, prediction in predictions.items():
                    metrics = compute_frame_metrics(prediction, target, valid_mask)
                    accumulators[name].add(metrics)
                    frame_reports[name].append({"frame_index": frame_index, **_json_safe(metrics)})
                seen_frames += 1
    finally:
        for name, model in named_models.items():
            model.train(previous_training[name])

    if seen_frames == 0:
        raise ValueError("evaluation requires at least one frame")

    methods: dict[str, object] = {}
    for name in method_names:
        method: dict[str, object] = {
            "frames": frame_reports[name],
            "aggregate": _aggregate_metrics(accumulators[name]),
        }
        if name in named_models:
            method["diagnostics"] = {
                "frames": diagnostic_frames[name],
                "aggregate": _aggregate_diagnostics(diagnostic_values[name]),
            }
        methods[name] = method
    return {"methods": methods}


def evaluate_synthetic_case(models: Mapping[str, nn.Module]) -> dict[str, object]:
    """Evaluate a one-frame, 4x4 packed zero case for pure-function smoke tests."""
    packed = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    frame: dict[str, Tensor | int] = {name: packed for name in _FRAME_INPUT_NAMES}
    frame["frame_index"] = 0
    return evaluate_models(models, [frame], saturation_threshold=1.0)


def write_evaluation_json(path: Path, report: Mapping[str, object]) -> None:
    """Serialize a finite-or-string report while forbidding JSON NaN/Infinity values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(report), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_from_checkpoints(
    model_specs: Mapping[str, ModelSpec],
    *,
    sequence: str,
    frame_range: tuple[int, int],
) -> dict[str, object]:
    """Strictly load compatible named checkpoints and evaluate their shared RAW frames."""
    specs = _validate_model_specs(model_specs)
    configured: dict[str, tuple[ModelSpec, ExperimentConfig, DatasetConfig]] = {}
    reference_dataset: DatasetConfig | None = None
    reference_experiment: ExperimentConfig | None = None

    for name, spec in specs.items():
        experiment = load_experiment_config(spec.config_path)
        dataset = load_dataset_config(experiment.dataset_path)
        if reference_dataset is None:
            reference_dataset = dataset
            reference_experiment = experiment
        elif dataset != reference_dataset:
            raise ValueError("all model configurations must use the same dataset")
        if sequence not in dataset.sequences:
            raise ValueError(f"unknown sequence for model {name}: {sequence}")
        _validate_frame_range(frame_range, dataset.layout.frame_count)
        configured[name] = (spec, experiment, dataset)

    if reference_dataset is None or reference_experiment is None:
        raise ValueError("at least one model is required")
    validate_saturation_margin_dn(
        {name: experiment.loss.saturation_margin_dn for name, (_, experiment, _) in configured.items()}
    )

    loaded_models: dict[str, nn.Module] = {}
    for name, (spec, experiment, dataset) in configured.items():
        model = CausalRawFusionNet(experiment.model).to("cpu")
        load_checkpoint_strict(
            spec.checkpoint_path,
            expected_fingerprint=config_fingerprint(dataset, experiment),
            model=model,
            device="cpu",
        )
        loaded_models[name] = model

    threshold = FusionLoss(
        reference_experiment.loss,
        white_level=reference_dataset.layout.white_level,
        target_black_level=reference_dataset.layout.target_black_level,
    ).saturation_threshold
    report = evaluate_models(
        loaded_models,
        _iter_dataset_frames(reference_dataset, sequence, frame_range),
        saturation_threshold=threshold,
    )
    return {"sequence": sequence, "frame_range": list(frame_range), "methods": report["methods"]}


def _iter_dataset_frames(
    dataset: DatasetConfig,
    sequence_name: str,
    frame_range: tuple[int, int],
) -> Iterable[dict[str, Tensor | int]]:
    layout = dataset.layout
    selected = dataset.sequences[sequence_name]
    noisy = RawStreamReader(
        selected.noisy_stream,
        layout.width,
        layout.height,
        layout.frame_count,
        layout.noisy_shift,
    )
    denoised = RawStreamReader(
        selected.denoised_stream, layout.width, layout.height, layout.frame_count, 0
    )
    fused = RawStreamReader(
        selected.fused_stream, layout.width, layout.height, layout.frame_count, 0
    )
    target = RawFrameDirectoryReader(
        selected.pseudo_gt_dir,
        layout.pseudo_gt_pattern,
        layout.width,
        layout.height,
        0,
    )
    start, end = _validate_frame_range(frame_range, layout.frame_count)
    for index in range(start, end + 1):
        previous_index = index if index == 0 else index - 1
        yield {
            "frame_index": index,
            "prev_noisy": _packed_tensor(
                noisy.read_frame(previous_index), layout.noisy_black_level, layout.white_level
            ),
            "curr_noisy": _packed_tensor(
                noisy.read_frame(index), layout.noisy_black_level, layout.white_level
            ),
            "denoised": _packed_tensor(
                denoised.read_frame(index), layout.candidate_black_level, layout.white_level
            ),
            "fused": _packed_tensor(
                fused.read_frame(index), layout.candidate_black_level, layout.white_level
            ),
            "target": _packed_tensor(
                target.read_frame(index), layout.target_black_level, layout.white_level
            ),
        }


def _packed_tensor(raw: Any, black_level: int, white_level: int) -> Tensor:
    return torch.from_numpy(pack_rggb(normalize_raw(raw, black_level, white_level))).unsqueeze(0)


def _validate_models(models: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
    if not isinstance(models, Mapping):
        raise TypeError("models must be a mapping")
    result: dict[str, nn.Module] = {}
    for name, model in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model names must be non-empty strings")
        if name in _BASELINE_NAMES:
            raise ValueError(f"model name is reserved for a fixed baseline: {name}")
        if not isinstance(model, nn.Module):
            raise TypeError(f"model {name} must be an nn.Module")
        result[name] = model
    return result


def _validate_frame(
    frame: Mapping[str, Tensor | int], position: int
) -> tuple[dict[str, Tensor], int]:
    if not isinstance(frame, Mapping):
        raise TypeError("each frame must be a mapping")
    missing = set(_FRAME_INPUT_NAMES) - set(frame)
    if missing:
        raise ValueError(f"frame is missing inputs: {', '.join(sorted(missing))}")
    values = {name: frame[name] for name in _FRAME_INPUT_NAMES}
    if any(not isinstance(value, Tensor) for value in values.values()):
        raise TypeError("frame inputs must be tensors")
    tensors = {name: value for name, value in values.items() if isinstance(value, Tensor)}
    reference = tensors["target"]
    if reference.ndim != 4 or reference.shape[1] != 4:
        raise ValueError("frame inputs must have shape [B, 4, H, W]")
    if any(value.shape != reference.shape for value in tensors.values()):
        raise ValueError("all frame inputs must have the same shape")
    if any(not value.is_floating_point() for value in tensors.values()):
        raise TypeError("frame inputs must be floating point")
    if any(value.device.type != "cpu" for value in tensors.values()):
        raise ValueError("evaluation only supports CPU tensors")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("frame inputs must be finite")
    frame_index = frame.get("frame_index", position)
    try:
        index = operator.index(frame_index)
    except TypeError as error:
        raise TypeError("frame_index must be an integer") from error
    return tensors, index


def _validate_model_output(name: str, output: object, target: Tensor) -> None:
    if not isinstance(output, FusionOutput):
        raise TypeError(f"model {name} must return FusionOutput")
    expected = {
        "prediction": target.shape,
        "gate": (target.shape[0], 1, target.shape[2], target.shape[3]),
        "correction": target.shape,
    }
    for field, shape in expected.items():
        value = getattr(output, field)
        if not isinstance(value, Tensor) or value.shape != shape:
            raise ValueError(f"model {name} {field} has invalid shape")
        if value.device != target.device or value.dtype != target.dtype:
            raise ValueError(f"model {name} {field} must match target device and dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"model {name} {field} must be finite")


def _frame_diagnostics(output: FusionOutput) -> dict[str, float]:
    gate = output.gate.detach().cpu().reshape(-1).float()
    correction = output.correction.detach().cpu().abs().reshape(-1).float()
    return {
        "gate_mean": float(gate.mean().item()),
        "gate_p10": float(torch.quantile(gate, 0.10).item()),
        "gate_p50": float(torch.quantile(gate, 0.50).item()),
        "gate_p90": float(torch.quantile(gate, 0.90).item()),
        "correction_abs_mean": float(correction.mean().item()),
    }


def _aggregate_diagnostics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("model diagnostics are empty")
    return {
        f"{name}_frame_mean": float(sum(item[name] for item in values) / len(values))
        for name in _DIAGNOSTIC_NAMES
    }


def _aggregate_metrics(accumulator: MetricAccumulator) -> dict[str, object]:
    aggregate = accumulator.compute()
    mse = aggregate.get("mse")
    if not isinstance(mse, float) or not math.isfinite(mse) or mse < 0.0:
        raise ValueError("aggregate MSE must be finite and non-negative")
    aggregate["psnr"] = math.inf if mse == 0.0 else -10.0 * math.log10(mse)
    return _json_safe(aggregate)


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError("saturation_threshold must be finite and in (0, 1]")
    return threshold


def _validate_frame_range(frame_range: tuple[int, int], frame_count: int) -> tuple[int, int]:
    if len(frame_range) != 2:
        raise ValueError("frame range must have two entries")
    start, end = (operator.index(value) for value in frame_range)
    if start < 0 or end < start or end >= frame_count:
        raise ValueError("frame range is outside the dataset")
    return start, end


def _validate_model_specs(model_specs: Mapping[str, ModelSpec]) -> dict[str, ModelSpec]:
    if not isinstance(model_specs, Mapping) or not model_specs:
        raise ValueError("at least one model specification is required")
    result: dict[str, ModelSpec] = {}
    for name, spec in model_specs.items():
        if not isinstance(spec, ModelSpec) or name != spec.name:
            raise ValueError("model specification name mismatch")
        result[name] = spec
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("evaluation report contains NaN")
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_frame_range(value: str) -> tuple[int, int]:
    start, separator, end = value.partition(":")
    if not separator or not start or not end:
        raise ValueError("--frames must use START:END")
    try:
        return int(start), int(end)
    except ValueError as error:
        raise ValueError("--frames must use integer START:END") from error


def _require_comparison_models(model_specs: Mapping[str, ModelSpec]) -> None:
    missing = {"candidate", "full"} - set(model_specs)
    if missing:
        raise ValueError(
            "formal evaluation requires both candidate and full models; missing "
            + ", ".join(sorted(missing))
        )


def main(argv: list[str] | None = None) -> None:
    """Run strict named-model evaluation and save an ``allow_nan=False`` JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CONFIG,CHECKPOINT")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", required=True, metavar="START:END")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    model_specs = parse_model_specs(arguments.model)
    _require_comparison_models(model_specs)
    report = evaluate_from_checkpoints(
        model_specs,
        sequence=arguments.sequence,
        frame_range=_parse_frame_range(arguments.frames),
    )
    write_evaluation_json(arguments.output, report)


if __name__ == "__main__":
    main()

"""Frozen, strictly parsed configuration contracts for RAW fusion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RawLayout:
    width: int
    height: int
    frame_count: int
    dtype: str
    white_level: int
    noisy_black_level: int
    candidate_black_level: int
    target_black_level: int
    noisy_shift: int
    cfa_pattern: str
    pseudo_gt_pattern: str


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    name: str
    noisy_stream: Path
    denoised_stream: Path
    fused_stream: Path
    pseudo_gt_dir: Path
    white_balance: tuple[float, float, float]
    isp_gain: float


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    layout: RawLayout
    sequences: dict[str, SequenceConfig]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_sequence: str
    train_frames: tuple[int, int]
    validation_sequence: str
    validation_frames: tuple[int, int]
    test_sequence: str
    test_frames: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    channels: tuple[int, int, int]
    residual_scale: float
    use_temporal: bool


@dataclass(frozen=True, slots=True)
class LossConfig:
    gradient_weight: float
    gate_weight: float
    residual_weight: float
    range_weight: float
    charbonnier_epsilon: float
    gate_temperature: float
    gate_margin: float
    saturation_margin_dn: int


@dataclass(frozen=True, slots=True)
class TrainConfig:
    patch_size_packed: int
    samples_per_epoch: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    amp: bool
    device: str


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    tile_size_packed: int
    overlap_packed: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    dataset_path: Path
    split: SplitConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    inference: InferenceConfig


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load a dataset configuration and resolve its file paths."""
    source_path, document = _load_json(path)
    _check_keys(document, {"layout", "sequences"}, "dataset")
    layout = _parse_layout(document["layout"])
    sequences_document = _mapping(document["sequences"], "dataset.sequences")
    sequences = {
        name: _parse_sequence(name, value, source_path.parent)
        for name, value in sequences_document.items()
    }
    return DatasetConfig(layout=layout, sequences=sequences)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration and resolve its dataset path."""
    source_path, document = _load_json(path)
    _check_keys(
        document,
        {"dataset", "split", "model", "loss", "train", "inference"},
        "experiment",
    )
    dataset_path = _resolve_path(document["dataset"], source_path.parent, "experiment.dataset")
    return ExperimentConfig(
        dataset_path=dataset_path,
        split=_parse_split(document["split"]),
        model=_parse_model(document["model"]),
        loss=_parse_loss(document["loss"]),
        train=_parse_train(document["train"]),
        inference=_parse_inference(document["inference"]),
    )


def config_fingerprint(dataset: DatasetConfig, experiment: ExperimentConfig) -> str:
    """Return the stable compatibility hash for normalization and model choices."""
    layout = dataset.layout
    payload = {
        "cfa_pattern": layout.cfa_pattern,
        "normalization": {
            "white_level": layout.white_level,
            "noisy_black_level": layout.noisy_black_level,
            "candidate_black_level": layout.candidate_black_level,
            "target_black_level": layout.target_black_level,
            "noisy_shift": layout.noisy_shift,
        },
        "model": {
            "channels": experiment.model.channels,
            "use_temporal": experiment.model.use_temporal,
            "residual_scale": experiment.model.residual_scale,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> tuple[Path, dict[str, Any]]:
    source_path = Path(path).resolve()
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 JSON 配置: {source_path}") from error
    return source_path, _mapping(document, str(source_path))


def _parse_layout(value: Any) -> RawLayout:
    document = _mapping(value, "dataset.layout")
    _check_keys(
        document,
        {
            "width",
            "height",
            "frame_count",
            "dtype",
            "white_level",
            "noisy_black_level",
            "candidate_black_level",
            "target_black_level",
            "noisy_shift",
            "cfa_pattern",
            "pseudo_gt_pattern",
        },
        "dataset.layout",
    )
    return RawLayout(
        width=_integer(document["width"], "dataset.layout.width"),
        height=_integer(document["height"], "dataset.layout.height"),
        frame_count=_integer(document["frame_count"], "dataset.layout.frame_count"),
        dtype=_string(document["dtype"], "dataset.layout.dtype"),
        white_level=_integer(document["white_level"], "dataset.layout.white_level"),
        noisy_black_level=_integer(document["noisy_black_level"], "dataset.layout.noisy_black_level"),
        candidate_black_level=_integer(
            document["candidate_black_level"], "dataset.layout.candidate_black_level"
        ),
        target_black_level=_integer(document["target_black_level"], "dataset.layout.target_black_level"),
        noisy_shift=_integer(document["noisy_shift"], "dataset.layout.noisy_shift"),
        cfa_pattern=_cfa_pattern(document["cfa_pattern"]),
        pseudo_gt_pattern=_string(document["pseudo_gt_pattern"], "dataset.layout.pseudo_gt_pattern"),
    )


def _parse_sequence(name: str, value: Any, base_path: Path) -> SequenceConfig:
    document = _mapping(value, f"dataset.sequences.{name}")
    _check_keys(
        document,
        {
            "name",
            "noisy_stream",
            "denoised_stream",
            "fused_stream",
            "pseudo_gt_dir",
            "white_balance",
            "isp_gain",
        },
        f"dataset.sequences.{name}",
    )
    white_balance = _fixed_float_tuple(
        document["white_balance"], 3, f"dataset.sequences.{name}.white_balance"
    )
    return SequenceConfig(
        name=_string(document["name"], f"dataset.sequences.{name}.name"),
        noisy_stream=_resolve_path(document["noisy_stream"], base_path, "noisy_stream"),
        denoised_stream=_resolve_path(document["denoised_stream"], base_path, "denoised_stream"),
        fused_stream=_resolve_path(document["fused_stream"], base_path, "fused_stream"),
        pseudo_gt_dir=_resolve_path(document["pseudo_gt_dir"], base_path, "pseudo_gt_dir"),
        white_balance=(white_balance[0], white_balance[1], white_balance[2]),
        isp_gain=_number(document["isp_gain"], f"dataset.sequences.{name}.isp_gain"),
    )


def _parse_split(value: Any) -> SplitConfig:
    document = _mapping(value, "experiment.split")
    _check_keys(
        document,
        {
            "train_sequence",
            "train_frames",
            "validation_sequence",
            "validation_frames",
            "test_sequence",
            "test_frames",
        },
        "experiment.split",
    )
    return SplitConfig(
        train_sequence=_string(document["train_sequence"], "experiment.split.train_sequence"),
        train_frames=_frame_range(document["train_frames"], "experiment.split.train_frames"),
        validation_sequence=_string(
            document["validation_sequence"], "experiment.split.validation_sequence"
        ),
        validation_frames=_frame_range(
            document["validation_frames"], "experiment.split.validation_frames"
        ),
        test_sequence=_string(document["test_sequence"], "experiment.split.test_sequence"),
        test_frames=_frame_range(document["test_frames"], "experiment.split.test_frames"),
    )


def _parse_model(value: Any) -> ModelConfig:
    document = _mapping(value, "experiment.model")
    _check_keys(document, {"channels", "residual_scale", "use_temporal"}, "experiment.model")
    channels = _fixed_int_tuple(document["channels"], 3, "experiment.model.channels")
    return ModelConfig(
        channels=(channels[0], channels[1], channels[2]),
        residual_scale=_number(document["residual_scale"], "experiment.model.residual_scale"),
        use_temporal=_boolean(document["use_temporal"], "experiment.model.use_temporal"),
    )


def _parse_loss(value: Any) -> LossConfig:
    document = _mapping(value, "experiment.loss")
    _check_keys(
        document,
        {
            "gradient_weight",
            "gate_weight",
            "residual_weight",
            "range_weight",
            "charbonnier_epsilon",
            "gate_temperature",
            "gate_margin",
            "saturation_margin_dn",
        },
        "experiment.loss",
    )
    return LossConfig(
        gradient_weight=_number(document["gradient_weight"], "experiment.loss.gradient_weight"),
        gate_weight=_number(document["gate_weight"], "experiment.loss.gate_weight"),
        residual_weight=_number(document["residual_weight"], "experiment.loss.residual_weight"),
        range_weight=_number(document["range_weight"], "experiment.loss.range_weight"),
        charbonnier_epsilon=_number(
            document["charbonnier_epsilon"], "experiment.loss.charbonnier_epsilon"
        ),
        gate_temperature=_number(document["gate_temperature"], "experiment.loss.gate_temperature"),
        gate_margin=_number(document["gate_margin"], "experiment.loss.gate_margin"),
        saturation_margin_dn=_integer(
            document["saturation_margin_dn"], "experiment.loss.saturation_margin_dn"
        ),
    )


def _parse_train(value: Any) -> TrainConfig:
    document = _mapping(value, "experiment.train")
    _check_keys(
        document,
        {
            "patch_size_packed",
            "samples_per_epoch",
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "num_workers",
            "seed",
            "amp",
            "device",
        },
        "experiment.train",
    )
    return TrainConfig(
        patch_size_packed=_integer(
            document["patch_size_packed"], "experiment.train.patch_size_packed"
        ),
        samples_per_epoch=_integer(
            document["samples_per_epoch"], "experiment.train.samples_per_epoch"
        ),
        batch_size=_integer(document["batch_size"], "experiment.train.batch_size"),
        epochs=_integer(document["epochs"], "experiment.train.epochs"),
        learning_rate=_number(document["learning_rate"], "experiment.train.learning_rate"),
        weight_decay=_number(document["weight_decay"], "experiment.train.weight_decay"),
        num_workers=_integer(document["num_workers"], "experiment.train.num_workers"),
        seed=_integer(document["seed"], "experiment.train.seed"),
        amp=_boolean(document["amp"], "experiment.train.amp"),
        device=_string(document["device"], "experiment.train.device"),
    )


def _parse_inference(value: Any) -> InferenceConfig:
    document = _mapping(value, "experiment.inference")
    _check_keys(document, {"tile_size_packed", "overlap_packed"}, "experiment.inference")
    return InferenceConfig(
        tile_size_packed=_integer(
            document["tile_size_packed"], "experiment.inference.tile_size_packed"
        ),
        overlap_packed=_integer(document["overlap_packed"], "experiment.inference.overlap_packed"),
    )


def _check_keys(document: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(document)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"{context} 存在未知字段: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{context} 缺少必需字段: {', '.join(sorted(missing))}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} 必须是对象")
    return value


def _resolve_path(value: Any, base_path: Path, context: str) -> Path:
    return (base_path / _string(value, context)).resolve()


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} 必须是字符串")
    return value


def _cfa_pattern(value: Any) -> str:
    context = "dataset.layout.cfa_pattern"
    pattern = _string(value, context)
    if pattern != "RGGB":
        raise ValueError(f"{context} 必须为 RGGB")
    return pattern


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} 必须是整数")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} 必须是数字")
    return float(value)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} 必须是布尔值")
    return value


def _fixed_int_tuple(value: Any, length: int, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} 必须是包含 {length} 个元素的数组")
    return tuple(_integer(item, context) for item in value)


def _fixed_float_tuple(value: Any, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} 必须是包含 {length} 个元素的数组")
    return tuple(_number(item, context) for item in value)


def _frame_range(value: Any, context: str) -> tuple[int, int]:
    values = _fixed_int_tuple(value, 2, context)
    start, end = values
    if start < 0 or start > end:
        raise ValueError(f"{context} 必须是非负闭区间 [start, end]")
    return start, end

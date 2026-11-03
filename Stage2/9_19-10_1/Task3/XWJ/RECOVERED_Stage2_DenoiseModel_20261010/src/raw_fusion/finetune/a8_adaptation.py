"""Conservative static-reference sensor adaptation from the immutable A3 model."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from raw_fusion.finetune.a3 import (
    A3Config,
    PhaseADataset,
    build_rvideformer,
    compute_loss,
    set_seed,
)
from raw_fusion.finetune.cross_sensor import (
    NativeSequenceSpec,
    SATURATION_LIMIT_DN,
    TEMPORAL_FRAMES,
    channel_vector,
    normalise_native_dn,
    open_sequence,
)
from raw_fusion.finetune.stream_io import pack_frame


@dataclass(frozen=True, slots=True)
class A8AdaptationConfig:
    """Reproducible configuration for one static-reference sensor adaptation."""

    dataset_root: str
    source: NativeSequenceSpec
    reference_dir: str
    a3_checkpoint: str
    a3_checkpoint_sha256: str
    source_sha256: str
    replay_supervision_dir: str
    replay_noise_dir: str
    output_dir: str
    seed: int = 20261013
    steps: int = 500
    batch_size: int = 1
    learning_rate: float = 1e-5
    sensor_fraction: float = 0.5
    patch_size: int = 64
    gradient_weight: float = 0.1
    highpass_weight: float = 0.1
    min_valid_fraction: float = 0.25
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    checkpoint_interval: int = 100
    precision: str = "bfloat16"
    trainable_scope: str = "output_head"
    device: str = "cuda:0"

    def validate(self) -> None:
        dataset_root = Path(self.dataset_root).resolve()
        source_path = self.source.path.resolve()
        if not source_path.is_relative_to(dataset_root):
            raise ValueError("adaptation source must be inside the authoritative dataset root")
        if self.source.reference_mode != "motion_aware_split_half":
            raise ValueError("static adaptation requires motion_aware_split_half reference mode")
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.patch_size != 64:
            raise ValueError("RViDeformer adaptation requires packed patch size 64")
        if not 0.0 < self.sensor_fraction < 1.0:
            raise ValueError("sensor_fraction must be in (0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.gradient_clip_norm <= 0.0 or self.checkpoint_interval <= 0:
            raise ValueError("gradient clip norm and checkpoint interval must be positive")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if self.trainable_scope != "output_head":
            raise ValueError("trainable_scope must be output_head")

    def with_runtime_overrides(
        self,
        *,
        steps: int | None = None,
        output_dir: Path | None = None,
        device: str | None = None,
    ) -> "A8AdaptationConfig":
        return replace(
            self,
            steps=self.steps if steps is None else steps,
            output_dir=self.output_dir if output_dir is None else str(output_dir),
            device=self.device if device is None else device,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adaptation_config(path: Path) -> A8AdaptationConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_payload = dict(payload["source"])
    source = NativeSequenceSpec.from_mapping(
        str(source_payload.pop("name")), source_payload
    )
    config = A8AdaptationConfig(
        dataset_root=str(payload["dataset_root"]),
        source=source,
        reference_dir=str(payload["reference_dir"]),
        a3_checkpoint=str(payload["a3_checkpoint"]),
        a3_checkpoint_sha256=str(payload["a3_checkpoint_sha256"]),
        source_sha256=str(payload["source_sha256"]),
        replay_supervision_dir=str(payload["replay_supervision_dir"]),
        replay_noise_dir=str(payload["replay_noise_dir"]),
        output_dir=str(payload["output_dir"]),
        seed=int(payload.get("seed", 20261013)),
        steps=int(payload.get("steps", 500)),
        batch_size=int(payload.get("batch_size", 1)),
        learning_rate=float(payload.get("learning_rate", 1e-5)),
        sensor_fraction=float(payload.get("sensor_fraction", 0.5)),
        patch_size=int(payload.get("patch_size", 64)),
        gradient_weight=float(payload.get("gradient_weight", 0.1)),
        highpass_weight=float(payload.get("highpass_weight", 0.1)),
        min_valid_fraction=float(payload.get("min_valid_fraction", 0.25)),
        ema_decay=float(payload.get("ema_decay", 0.999)),
        gradient_clip_norm=float(payload.get("gradient_clip_norm", 1.0)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 100)),
        precision=str(payload.get("precision", "bfloat16")),
        trainable_scope=str(payload.get("trainable_scope", "output_head")),
        device=str(payload.get("device", "cuda:0")),
    )
    config.validate()
    return config


def config_mapping(config: A8AdaptationConfig) -> dict[str, object]:
    mapping = asdict(config)
    source = mapping.pop("source")
    if not isinstance(source, dict):
        raise TypeError("serialized source must be a mapping")
    source["path"] = str(source["path"])
    source["name"] = config.source.name
    mapping["source"] = source
    return mapping


class StaticSplitHalfDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Static sensor patches with mutually exclusive input/target frame parity."""

    def __init__(
        self,
        config: A8AdaptationConfig,
        *,
        samples: int = 1024,
    ) -> None:
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.config = config
        self.samples = samples
        self.reader = open_sequence(config.source)
        self.adapter = config.source.adapter
        self.black = channel_vector(self._load_report()["black_levels"])
        reference_dir = Path(config.reference_dir)
        self.full_mean = np.load(reference_dir / "full_mean_dn.npy", mmap_mode="r")
        self.even_mean = np.load(reference_dir / "even_mean_dn.npy", mmap_mode="r")
        self.odd_mean = np.load(reference_dir / "odd_mean_dn.npy", mmap_mode="r")
        self.even_confidence = np.load(
            reference_dir / "even_confidence.npy", mmap_mode="r"
        )
        self.odd_confidence = np.load(
            reference_dir / "odd_confidence.npy", mmap_mode="r"
        )
        self.valid_region = np.load(reference_dir / "valid_region.npy", mmap_mode="r")
        self.motion_thresholds = np.load(
            reference_dir / "motion_threshold_per_frame.npy", mmap_mode="r"
        )
        expected_shape = (4, *config.source.packed_shape)
        for name, array in (
            ("full_mean", self.full_mean),
            ("even_mean", self.even_mean),
            ("odd_mean", self.odd_mean),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
        if self.valid_region.shape != config.source.packed_shape:
            raise ValueError("valid region shape does not match the adaptation source")
        if len(self.motion_thresholds) < config.source.frames_used:
            raise ValueError("motion threshold array does not cover all source frames")

    def _load_report(self) -> dict[str, object]:
        path = Path(self.config.reference_dir) / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        provenance = report["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("reference report provenance must be a mapping")
        if Path(str(provenance["source_path"])).resolve() != self.config.source.path.resolve():
            raise ValueError("reference report source path does not match adaptation source")
        if str(provenance["source_sha256"]) != self.config.source_sha256:
            raise ValueError("reference report source SHA-256 does not match configuration")
        return report

    def __len__(self) -> int:
        return self.samples

    def _frame_indices(
        self, parity: int, rng: np.random.Generator
    ) -> np.ndarray:
        last_first = self.config.source.frames_used - 1 - 2 * (TEMPORAL_FRAMES - 1)
        candidates = np.arange(parity, last_first + 1, 2, dtype=np.int64)
        if candidates.size == 0:
            raise ValueError("source has too few frames for split-half temporal windows")
        first = int(candidates[int(rng.integers(0, len(candidates)))])
        return first + np.arange(TEMPORAL_FRAMES, dtype=np.int64) * 2

    @staticmethod
    def _augment(
        inputs: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            inputs = inputs[..., ::-1]
            target = target[..., ::-1]
            mask = mask[..., ::-1]
        if rng.random() < 0.5:
            inputs = inputs[..., ::-1, :]
            target = target[..., ::-1, :]
            mask = mask[..., ::-1, :]
        if rng.random() < 0.5:
            inputs = inputs.swapaxes(-2, -1)
            target = target.swapaxes(-2, -1)
            mask = mask.swapaxes(-2, -1)
        return inputs, target, mask

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = np.random.default_rng(self.config.seed + index * 104729)
        parity = index % 2
        indices = self._frame_indices(parity, rng)
        target_mean = self.odd_mean if parity == 0 else self.even_mean
        target_confidence = (
            self.odd_confidence if parity == 0 else self.even_confidence
        )
        patch = self.config.patch_size
        height, width = self.config.source.packed_shape

        for _ in range(32):
            y = int(rng.integers(0, height - patch + 1))
            x = int(rng.integers(0, width - patch + 1))
            input_dn = np.stack(
                [
                    pack_frame(self.reader.read_frame(int(frame_index)), self.adapter)[
                        :, y : y + patch, x : x + patch
                    ]
                    for frame_index in indices
                ],
                axis=0,
            ).astype(np.float32)
            reference_gray = np.asarray(
                self.full_mean[:, y : y + patch, x : x + patch], dtype=np.float32
            ).mean(axis=0)
            input_gray = input_dn.mean(axis=1)
            thresholds = np.asarray(
                self.motion_thresholds[indices], dtype=np.float32
            )[:, None, None]
            static = np.all(np.abs(input_gray - reference_gray[None]) <= thresholds, axis=0)
            unsaturated = np.all(input_dn < SATURATION_LIMIT_DN, axis=(0, 1))
            valid = np.asarray(
                self.valid_region[y : y + patch, x : x + patch], dtype=bool
            ).copy()
            valid &= np.asarray(
                target_confidence[y : y + patch, x : x + patch] >= 0.9,
                dtype=bool,
            )
            valid &= static & unsaturated
            if float(valid.mean()) >= self.config.min_valid_fraction:
                break
        else:
            raise RuntimeError("failed to sample a sufficiently valid static sensor patch")

        inputs = normalise_native_dn(input_dn, self.black)
        target_frame = normalise_native_dn(
            np.asarray(
                target_mean[:, y : y + patch, x : x + patch], dtype=np.float32
            ),
            self.black,
        )
        target = np.broadcast_to(
            target_frame, (TEMPORAL_FRAMES, 4, patch, patch)
        ).copy()
        mask = np.broadcast_to(valid, (TEMPORAL_FRAMES, patch, patch)).copy()
        inputs, target, mask = self._augment(inputs, target, mask, rng)
        arrays = (inputs, target, mask.astype(np.float32))
        return tuple(
            torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
            for array in arrays
        )


class MixedSensorReplayDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Deterministically mix target-sensor supervision and MIS20S1 replay."""

    def __init__(self, config: A8AdaptationConfig, *, samples: int = 1024) -> None:
        self.config = config
        self.samples = samples
        self.sensor = StaticSplitHalfDataset(config, samples=samples)
        replay_config = A3Config(
            dataset_root=config.dataset_root,
            supervision_dir=config.replay_supervision_dir,
            noise_dir=config.replay_noise_dir,
            output_dir=config.output_dir,
            checkpoint=config.a3_checkpoint,
            seed=config.seed + 1_000_003,
            steps=config.steps,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            holdout_gain="1000x",
            device=config.device,
        )
        self.replay = PhaseADataset(replay_config, samples=samples)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        selector = np.random.default_rng(
            self.config.seed + index * 32452843
        ).random()
        if selector < self.config.sensor_fraction:
            inputs, target, mask = self.sensor[index]
            source_id = 1
        else:
            inputs, target, mask = self.replay[index]
            source_id = 0
        return inputs, target, mask, torch.tensor(source_id, dtype=torch.int64)


def _load_a3_state(checkpoint: Path) -> dict[str, Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise ValueError("A8 initialization requires an A3 training checkpoint with EMA")
    state = payload["ema"]
    if not isinstance(state, dict):
        raise TypeError("A3 EMA state must be a mapping")
    return state


def _build_from_a3(checkpoint: Path, device: torch.device) -> nn.Module:
    model = build_rvideformer()
    model.load_state_dict(_load_a3_state(checkpoint), strict=True)
    return model.to(device)


def configure_output_head_training(
    model: nn.Module,
) -> tuple[list[nn.Parameter], dict[str, object]]:
    """Freeze the trunk and expose only the final reconstruction head."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_prefixes = ("norm.", "conv_after_body.", "conv_last.")
    trainable_names: list[str] = []
    trainable_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad_(True)
            trainable_names.append(name)
            trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise RuntimeError("RViDeformer output head did not expose trainable parameters")
    return trainable_parameters, {
        "scope": "output_head",
        "prefixes": list(trainable_prefixes),
        "parameter_tensors": len(trainable_parameters),
        "parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "names": trainable_names,
    }


@torch.no_grad()
def update_selected_ema(
    ema: nn.Module,
    model: nn.Module,
    trainable_names: set[str],
    decay: float,
) -> None:
    """Update only authorized EMA parameters and keep the frozen trunk exact."""

    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema.named_parameters():
        if name in trainable_names:
            ema_parameter.mul_(decay).add_(
                model_parameters[name], alpha=1.0 - decay
            )
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema.named_buffers():
        ema_buffer.copy_(model_buffers[name])


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: A8AdaptationConfig,
    source_counts: dict[str, int],
    trainable_report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "a8-static-sensor-adaptation-v1",
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_mapping(config),
            "source_counts": source_counts,
            "trainable": trainable_report,
            "initialization": {
                "checkpoint": str(Path(config.a3_checkpoint).resolve()),
                "checkpoint_sha256": config.a3_checkpoint_sha256,
                "weights": "ema",
            },
        },
        path,
    )


def train_static_sensor(config: A8AdaptationConfig) -> dict[str, object]:
    """Fine-tune A3 on one static sensor with MIS20S1 anti-forgetting replay."""

    config.validate()
    checkpoint = Path(config.a3_checkpoint)
    if sha256_file(checkpoint) != config.a3_checkpoint_sha256:
        raise ValueError("A3 checkpoint SHA-256 does not match the adaptation protocol")
    if sha256_file(config.source.path) != config.source_sha256:
        raise ValueError("sensor source SHA-256 does not match the adaptation protocol")
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"adaptation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A8 requires CUDA; torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    set_seed(config.seed)

    config_payload = config_mapping(config)
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="ascii"
    )
    dataset = MixedSensorReplayDataset(
        config, samples=max(1024, config.steps * config.batch_size)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    model = _build_from_a3(checkpoint, device)
    ema = _build_from_a3(checkpoint, device)
    trainable_parameters, trainable_report = configure_output_head_training(model)
    trainable_names = {
        str(name) for name in trainable_report["names"]
    }
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)

    source_counts = {"sensor": 0, "mis20s1_replay": 0}
    iterator = iter(loader)
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, config.steps + 1):
            try:
                inputs, targets, masks, source_ids = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets, masks, source_ids = next(iterator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device).unsqueeze(2)
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and config.precision == "bfloat16",
            ):
                prediction = model(inputs).clamp(0.0, 1.0)
            loss, metrics = compute_loss(
                prediction.float(),
                targets,
                masks,
                config.gradient_weight,
                config.highpass_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {metrics}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip_norm
                ).detach().cpu()
            )
            optimizer.step()
            update_selected_ema(
                ema,
                model,
                trainable_names,
                config.ema_decay,
            )

            sensor_items = int(source_ids.sum().item())
            source_counts["sensor"] += sensor_items
            source_counts["mis20s1_replay"] += int(source_ids.numel()) - sensor_items
            record = {
                "step": step,
                "source": "sensor" if sensor_items == source_ids.numel() else "mis20s1_replay",
                "gradient_norm": gradient_norm,
                **metrics,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(
                    f"[a8:{config.source.sensor}] step={step}/{config.steps} "
                    f"source={record['source']} loss={metrics['loss']:.6f} "
                    f"grad={gradient_norm:.4f}",
                    flush=True,
                )
            if step % config.checkpoint_interval == 0 or step == config.steps:
                _save_checkpoint(
                    output_dir / f"checkpoint_step_{step:06d}.pth",
                    model,
                    ema,
                    optimizer,
                    step,
                    config,
                    source_counts,
                    trainable_report,
                )

    summary = {
        "status": "completed",
        "sensor": config.source.sensor,
        "steps": config.steps,
        "source_counts": source_counts,
        "trainable": trainable_report,
        "checkpoint": str(
            (output_dir / f"checkpoint_step_{config.steps:06d}.pth").resolve()
        ),
        "log": str(log_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    return summary

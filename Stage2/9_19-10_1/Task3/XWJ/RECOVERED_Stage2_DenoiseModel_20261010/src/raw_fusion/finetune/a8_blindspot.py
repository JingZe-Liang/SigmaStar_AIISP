"""Blind-spot sensor adaptation for RAW streams without a clean reference."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from raw_fusion.finetune.a3 import A3Config, PhaseADataset, compute_loss, set_seed
from raw_fusion.finetune.a8_adaptation import (
    _build_from_a3,
    configure_output_head_training,
    sha256_file,
    update_selected_ema,
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
class A8BlindSpotConfig:
    """Reproducible single-frame blind-spot adaptation configuration."""

    dataset_root: str
    source: NativeSequenceSpec
    profile_dir: str
    a3_checkpoint: str
    a3_checkpoint_sha256: str
    source_sha256: str
    replay_supervision_dir: str
    replay_noise_dir: str
    output_dir: str
    train_frame_start: int
    train_frame_stop: int
    validation_frame_start: int
    validation_frame_stop: int
    seed: int = 20261013
    steps: int = 500
    batch_size: int = 1
    learning_rate: float = 1e-5
    sensor_fraction: float = 0.5
    patch_size: int = 64
    blindspot_fraction: float = 0.05
    blindspot_radius: int = 2
    gradient_weight: float = 0.1
    highpass_weight: float = 0.1
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
        if self.source.reference_mode != "stress_only":
            raise ValueError("blind-spot adaptation requires a stress-only source")
        train_range = range(self.train_frame_start, self.train_frame_stop)
        validation_range = range(
            self.validation_frame_start, self.validation_frame_stop
        )
        if not 0 <= self.train_frame_start < self.train_frame_stop <= self.source.frames_used:
            raise ValueError("training frame range is invalid")
        if not (
            0
            <= self.validation_frame_start
            < self.validation_frame_stop
            <= self.source.frames_used
        ):
            raise ValueError("validation frame range is invalid")
        if set(train_range).intersection(validation_range):
            raise ValueError("training and validation frame ranges must be disjoint")
        if self.steps <= 0 or self.batch_size != 1:
            raise ValueError("steps must be positive and batch_size must be one")
        if self.patch_size != 64:
            raise ValueError("RViDeformer adaptation requires packed patch size 64")
        if not 0.0 < self.sensor_fraction < 1.0:
            raise ValueError("sensor_fraction must be in (0, 1)")
        if not 0.0 < self.blindspot_fraction <= 0.25:
            raise ValueError("blindspot_fraction must be in (0, 0.25]")
        if not 0 < self.blindspot_radius < self.patch_size // 2:
            raise ValueError("blindspot_radius is invalid for the patch size")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
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
    ) -> "A8BlindSpotConfig":
        return replace(
            self,
            steps=self.steps if steps is None else steps,
            output_dir=self.output_dir if output_dir is None else str(output_dir),
            device=self.device if device is None else device,
        )


def load_blindspot_config(path: Path) -> A8BlindSpotConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_payload = dict(payload["source"])
    source = NativeSequenceSpec.from_mapping(
        str(source_payload.pop("name")), source_payload
    )
    config = A8BlindSpotConfig(
        dataset_root=str(payload["dataset_root"]),
        source=source,
        profile_dir=str(payload["profile_dir"]),
        a3_checkpoint=str(payload["a3_checkpoint"]),
        a3_checkpoint_sha256=str(payload["a3_checkpoint_sha256"]),
        source_sha256=str(payload["source_sha256"]),
        replay_supervision_dir=str(payload["replay_supervision_dir"]),
        replay_noise_dir=str(payload["replay_noise_dir"]),
        output_dir=str(payload["output_dir"]),
        train_frame_start=int(payload["train_frame_start"]),
        train_frame_stop=int(payload["train_frame_stop"]),
        validation_frame_start=int(payload["validation_frame_start"]),
        validation_frame_stop=int(payload["validation_frame_stop"]),
        seed=int(payload.get("seed", 20261013)),
        steps=int(payload.get("steps", 500)),
        batch_size=int(payload.get("batch_size", 1)),
        learning_rate=float(payload.get("learning_rate", 1e-5)),
        sensor_fraction=float(payload.get("sensor_fraction", 0.5)),
        patch_size=int(payload.get("patch_size", 64)),
        blindspot_fraction=float(payload.get("blindspot_fraction", 0.05)),
        blindspot_radius=int(payload.get("blindspot_radius", 2)),
        gradient_weight=float(payload.get("gradient_weight", 0.1)),
        highpass_weight=float(payload.get("highpass_weight", 0.1)),
        ema_decay=float(payload.get("ema_decay", 0.999)),
        gradient_clip_norm=float(payload.get("gradient_clip_norm", 1.0)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 100)),
        precision=str(payload.get("precision", "bfloat16")),
        trainable_scope=str(payload.get("trainable_scope", "output_head")),
        device=str(payload.get("device", "cuda:0")),
    )
    config.validate()
    return config


def blindspot_config_mapping(config: A8BlindSpotConfig) -> dict[str, object]:
    mapping = asdict(config)
    source = mapping.pop("source")
    if not isinstance(source, dict):
        raise TypeError("serialized source must be a mapping")
    source["path"] = str(source["path"])
    source["name"] = config.source.name
    mapping["source"] = source
    return mapping


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


class BlindSpotSingleFrameDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Create J-invariant patches without temporal averaging or teacher targets."""

    def __init__(
        self,
        config: A8BlindSpotConfig,
        *,
        samples: int = 1024,
    ) -> None:
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.config = config
        self.samples = samples
        self.reader = open_sequence(config.source)
        report_path = Path(config.profile_dir) / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        provenance = report["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("profile provenance must be a mapping")
        if Path(str(provenance["source_path"])).resolve() != config.source.path.resolve():
            raise ValueError("profile source path does not match adaptation source")
        if str(provenance["source_sha256"]) != config.source_sha256:
            raise ValueError("profile source SHA-256 does not match configuration")
        self.black = channel_vector(
            {str(name): float(value) for name, value in report["black_levels"].items()}
        )

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = np.random.default_rng(self.config.seed + index * 104729)
        frame_index = int(
            rng.integers(self.config.train_frame_start, self.config.train_frame_stop)
        )
        patch = self.config.patch_size
        height, width = self.config.source.packed_shape
        y = int(rng.integers(0, height - patch + 1))
        x = int(rng.integers(0, width - patch + 1))
        frame_dn = pack_frame(
            self.reader.read_frame(frame_index), self.config.source.adapter
        )[:, y : y + patch, x : x + patch].astype(np.float32)
        target_frame = normalise_native_dn(frame_dn, self.black)

        radius = self.config.blindspot_radius
        eligible = np.all(frame_dn < SATURATION_LIMIT_DN, axis=0)
        eligible[:radius] = False
        eligible[-radius:] = False
        eligible[:, :radius] = False
        eligible[:, -radius:] = False
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            raise RuntimeError("blind-spot patch has no eligible target pixels")
        masked_count = max(
            1, int(round(eligible_indices.size * self.config.blindspot_fraction))
        )
        selected = rng.choice(eligible_indices, size=masked_count, replace=False)
        rows, columns = np.unravel_index(selected, eligible.shape)
        offsets = rng.integers(
            -radius, radius + 1, size=(masked_count, 2), dtype=np.int64
        )
        zero_offsets = np.all(offsets == 0, axis=1)
        while bool(zero_offsets.any()):
            offsets[zero_offsets] = rng.integers(
                -radius,
                radius + 1,
                size=(int(zero_offsets.sum()), 2),
                dtype=np.int64,
            )
            zero_offsets = np.all(offsets == 0, axis=1)

        corrupted = target_frame.copy()
        replacement_rows = rows + offsets[:, 0]
        replacement_columns = columns + offsets[:, 1]
        corrupted[:, rows, columns] = target_frame[
            :, replacement_rows, replacement_columns
        ]
        spatial_mask = np.zeros((patch, patch), dtype=np.float32)
        spatial_mask[rows, columns] = 1.0
        inputs = np.broadcast_to(
            corrupted, (TEMPORAL_FRAMES, 4, patch, patch)
        ).copy()
        target = np.broadcast_to(
            target_frame, (TEMPORAL_FRAMES, 4, patch, patch)
        ).copy()
        mask = np.broadcast_to(
            spatial_mask, (TEMPORAL_FRAMES, patch, patch)
        ).copy()
        inputs, target, mask = _augment(inputs, target, mask, rng)
        return tuple(
            torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
            for array in (inputs, target, mask)
        )


class MixedBlindSpotReplayDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Deterministically mix blind-spot sensor patches and MIS20S1 replay."""

    def __init__(self, config: A8BlindSpotConfig, *, samples: int = 1024) -> None:
        self.config = config
        self.samples = samples
        self.sensor = BlindSpotSingleFrameDataset(config, samples=samples)
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


def blindspot_loss(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> tuple[Tensor, dict[str, float]]:
    """Apply reconstruction loss only where every temporal input was hidden."""

    expanded_mask = mask.expand_as(prediction)
    loss = (
        torch.sqrt((prediction - target).square() + 1e-6) * expanded_mask
    ).sum() / expanded_mask.sum().clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "charbonnier": float(loss.detach().cpu()),
        "gradient": 0.0,
        "highpass": 0.0,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: A8BlindSpotConfig,
    source_counts: dict[str, int],
    trainable_report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "a8-blindspot-sensor-adaptation-v1",
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": blindspot_config_mapping(config),
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


def train_blindspot_sensor(config: A8BlindSpotConfig) -> dict[str, object]:
    """Adapt the A3 output head using blind-spot and replay supervision."""

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

    (output_dir / "config.json").write_text(
        json.dumps(blindspot_config_mapping(config), indent=2) + "\n",
        encoding="ascii",
    )
    dataset = MixedBlindSpotReplayDataset(
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
    trainable_names = {str(name) for name in trainable_report["names"]}
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)

    source_counts = {"sensor_blindspot": 0, "mis20s1_replay": 0}
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
            source_id = int(source_ids.item())
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and config.precision == "bfloat16",
            ):
                prediction = model(inputs).clamp(0.0, 1.0)
            if source_id == 1:
                loss, metrics = blindspot_loss(prediction.float(), targets, masks)
                source_name = "sensor_blindspot"
            else:
                loss, metrics = compute_loss(
                    prediction.float(),
                    targets,
                    masks,
                    config.gradient_weight,
                    config.highpass_weight,
                )
                source_name = "mis20s1_replay"
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {metrics}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip_norm
                )
                .detach()
                .cpu()
            )
            optimizer.step()
            update_selected_ema(ema, model, trainable_names, config.ema_decay)

            source_counts[source_name] += 1
            record = {
                "step": step,
                "source": source_name,
                "gradient_norm": gradient_norm,
                **metrics,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(
                    f"[a8:{config.source.sensor}] step={step}/{config.steps} "
                    f"source={source_name} loss={metrics['loss']:.6f} "
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

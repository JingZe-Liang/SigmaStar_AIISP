"""Phase A RViDeformer fine-tuning data, model loading, and training."""

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.noise import generate_noisy_frame
from raw_fusion.finetune.stream_io import open_stream, pack_frame
from raw_fusion.finetune.rvideformer_arch import RViDeformer


_REFERENCE_ROOT = Path("/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer")
_RECOVERED_OFFICIAL_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "artifacts/pretrained/RViDeformer-L_CRVD.pth"
)
_CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
_WHITE_LEVEL = 4095.0
_PACKED_HEIGHT = 540
_PACKED_WIDTH = 960


@dataclass(frozen=True, slots=True)
class A3Config:
    """Reproducible Phase A training configuration."""

    dataset_root: str
    supervision_dir: str
    noise_dir: str
    output_dir: str
    checkpoint: str
    seed: int = 20260921
    steps: int = 200
    temporal_frames: int = 8
    patch_size: int = 64
    batch_size: int = 1
    learning_rate: float = 1e-4
    real_fraction: float = 0.5
    holdout_gain: str = "1000x"
    gradient_weight: float = 0.1
    highpass_weight: float = 0.1
    device: str = "cuda"
    synthetic_temporal_variants: int = 8
    resume_checkpoint: str | None = None


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_path(supervision_dir: Path, scene: str, gain: str) -> Path:
    if scene == "shdarkroom":
        return supervision_dir / "stream_manifests" / f"04_shdarkroom_{gain}.json"
    return supervision_dir / "stream_manifests" / f"04_mis20s1_outdoor_{gain}.json"


def _channel_vector(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[name] for name in _CHANNEL_NAMES], dtype=np.float32)


def _normalise_dn(values: np.ndarray, black: np.ndarray) -> np.ndarray:
    return np.clip((values.astype(np.float32) - black[:, None, None]) / (_WHITE_LEVEL - black[:, None, None]), 0.0, 1.0)


def _load_reference(supervision_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_dir = supervision_dir / "references"
    manifest = _load_json(reference_dir / "a1_reference_manifest.json")
    report = _load_json(reference_dir / "a1_reference_report.json")
    if not manifest["quality_passed"]:
        raise ValueError("A1 reference quality gate is not passed")
    clean = np.load(manifest["clean_reference_path"], mmap_mode="r")
    valid = np.load(manifest["valid_region_path"], mmap_mode="r")
    black = _channel_vector(report["black_levels"])
    return np.asarray(clean), np.asarray(valid, dtype=bool), black


@dataclass(frozen=True, slots=True)
class _RealRecord:
    manifest_path: Path
    gain: str
    frame_count: int
    shift_x: int
    shift_y: int
    slopes: np.ndarray
    intercepts: np.ndarray
    black: np.ndarray
    motion_mask_path: Path


@dataclass(frozen=True, slots=True)
class _SyntheticRecord:
    noisy_path: Path
    gain: str


def _real_records(supervision_dir: Path, holdout_gain: str) -> list[_RealRecord]:
    records: list[_RealRecord] = []
    for gain in ("128x", "313x", "632x", "645x", "1000x"):
        if gain == holdout_gain:
            continue
        path = _manifest_path(supervision_dir, "shdarkroom", gain)
        manifest = _load_json(path)
        fit = manifest["exposure_fit"]
        records.append(_RealRecord(
            manifest_path=path,
            gain=gain,
            frame_count=int(manifest["frames_used"]),
            shift_x=int(manifest["registration"]["integer_shift_x"]),
            shift_y=int(manifest["registration"]["integer_shift_y"]),
            slopes=_channel_vector(fit["slopes"]),
            intercepts=_channel_vector(fit["intercepts"]),
            black=_channel_vector(manifest["black_levels"]),
            motion_mask_path=Path(manifest["motion"]["mask_path"]),
        ))
    if not records:
        raise ValueError("no real cross-gain records remain after holdout")
    return records


def _synthetic_records(noise_dir: Path) -> list[_SyntheticRecord]:
    manifest = _load_json(noise_dir / "synthetic_pairs_manifest.json")
    records = [
        _SyntheticRecord(Path(pair["noisy_path"]), pair["gain_label"])
        for pair in manifest["pairs"]
    ]
    if not records:
        raise ValueError("A2 synthetic pair manifest is empty")
    return records


class PhaseADataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Random patch windows for real cross-gain and A2 synthetic supervision."""

    def __init__(self, config: A3Config, samples: int = 512) -> None:
        if config.temporal_frames != 8:
            raise ValueError("the official RViDeformer-L checkpoint requires 8 temporal frames")
        if config.patch_size != 64:
            raise ValueError("the official RViDeformer-L checkpoint requires packed patch size 64")
        if config.synthetic_temporal_variants < config.temporal_frames:
            raise ValueError("synthetic_temporal_variants must cover the temporal window")
        if not 0.0 < config.real_fraction < 1.0:
            raise ValueError("real_fraction must be in (0, 1)")

        self.config = config
        self.samples = samples
        self.supervision_dir = Path(config.supervision_dir)
        self.noise_dir = Path(config.noise_dir)
        self.clean_reference, self.valid_region, self.reference_black = _load_reference(self.supervision_dir)
        self.clean_normalised = _normalise_dn(self.clean_reference, self.reference_black)
        self.real = _real_records(self.supervision_dir, config.holdout_gain)
        self.synthetic = _synthetic_records(self.noise_dir)
        self._readers: dict[Path, tuple[object, object]] = {}
        self._motion_masks: dict[Path, np.ndarray] = {}
        self._synthetic_temporal = self._build_synthetic_temporal_cache()

    def __len__(self) -> int:
        return self.samples

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.config.seed + index * 104729)

    def _reader(self, record: _RealRecord) -> tuple[object, object]:
        loaded = self._readers.get(record.manifest_path)
        if loaded is None:
            manifest = _load_json(record.manifest_path)
            adapter = resolve_adapter(manifest["group"], manifest["scene"], manifest["gain_label"])
            reader, _ = open_stream(Path(manifest["path"]), adapter)
            loaded = (reader, adapter)
            self._readers[record.manifest_path] = loaded
        return loaded

    def _real_sample(self, record: _RealRecord, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reader, adapter = self._reader(record)
        height = _PACKED_HEIGHT
        width = _PACKED_WIDTH
        patch = self.config.patch_size
        start_y = max(0, -record.shift_y)
        end_y = min(height - patch, height - patch - record.shift_y)
        start_x = max(0, -record.shift_x)
        end_x = min(width - patch, width - patch - record.shift_x)
        if end_y < start_y or end_x < start_x:
            raise ValueError(f"registration shift leaves no patch overlap for {record.gain}")
        y = int(rng.integers(start_y, end_y + 1))
        x = int(rng.integers(start_x, end_x + 1))
        moving_y = y + record.shift_y
        moving_x = x + record.shift_x
        frame_start = int(rng.integers(0, record.frame_count - self.config.temporal_frames + 1))

        inputs: list[np.ndarray] = []
        static_masks: list[np.ndarray] = []
        motion = self._motion_masks.get(record.motion_mask_path)
        if motion is None:
            motion = np.load(record.motion_mask_path, mmap_mode="r")
            self._motion_masks[record.motion_mask_path] = motion
        for frame_index in range(frame_start, frame_start + self.config.temporal_frames):
            packed = pack_frame(reader.read_frame(frame_index), adapter).astype(np.float32)
            moving_patch = packed[:, moving_y : moving_y + patch, moving_x : moving_x + patch]
            mapped_dn = record.slopes[:, None, None] * moving_patch + record.intercepts[:, None, None]
            inputs.append(_normalise_dn(mapped_dn, self.reference_black))
            static_masks.append(np.asarray(motion[frame_index, 0, moving_y * 2 : (moving_y + patch) * 2 : 2, moving_x * 2 : (moving_x + patch) * 2 : 2] < 0.5))

        input_array = np.stack(inputs, axis=0)
        target = np.broadcast_to(self.clean_normalised[:, y : y + patch, x : x + patch], (self.config.temporal_frames, 4, patch, patch)).copy()
        valid = self.valid_region[y : y + patch, x : x + patch].copy()
        valid &= np.all(input_array < ((3800.0 - self.reference_black) / (_WHITE_LEVEL - self.reference_black))[:, None, None], axis=(0, 1))
        valid &= np.all(np.stack(static_masks), axis=0)
        return input_array, target, valid[None].repeat(self.config.temporal_frames, axis=0)

    def _build_synthetic_temporal_cache(self) -> dict[str, np.ndarray]:
        noise_model = _load_json(self.noise_dir / "noise_model.json")
        reference_manifest = _load_json(self.supervision_dir / "references" / "a1_reference_manifest.json")
        fpn_offset = np.asarray(np.load(reference_manifest["fpn_paths"]["offset_dn"], mmap_mode="r"))
        black_levels = {
            name: float(self.reference_black[channel])
            for channel, name in enumerate(_CHANNEL_NAMES)
        }
        cache: dict[str, np.ndarray] = {}
        for record_index, record in enumerate(self.synthetic):
            frames = [
                generate_noisy_frame(
                    self.clean_reference,
                    noise_model,
                    record.gain,
                    black_levels,
                    fpn_offset,
                    np.random.default_rng(self.config.seed + record_index * 1009 + variant),
                )
                for variant in range(self.config.synthetic_temporal_variants)
            ]
            cache[record.gain] = np.stack(frames, axis=0)
        return cache

    def _synthetic_sample(self, record: _SyntheticRecord, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        noisy_variants = self._synthetic_temporal[record.gain]
        patch = self.config.patch_size
        y = int(rng.integers(0, _PACKED_HEIGHT - patch + 1))
        x = int(rng.integers(0, _PACKED_WIDTH - patch + 1))
        variant_start = int(rng.integers(0, len(noisy_variants)))
        selected = np.stack([
            noisy_variants[(variant_start + offset) % len(noisy_variants), :, y : y + patch, x : x + patch]
            for offset in range(self.config.temporal_frames)
        ])
        input_array = _normalise_dn(selected, self.reference_black)
        target_frame = self.clean_normalised[:, y : y + patch, x : x + patch]
        target = np.broadcast_to(target_frame, (self.config.temporal_frames, 4, patch, patch)).copy()
        valid = self.valid_region[y : y + patch, x : x + patch].copy()
        valid &= np.all(input_array < ((3800.0 - self.reference_black) / (_WHITE_LEVEL - self.reference_black))[:, None, None], axis=(0, 1))
        return input_array, target, valid[None].repeat(self.config.temporal_frames, axis=0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = self._rng(index)
        real_count = int(round(self.samples * self.config.real_fraction))
        if index < real_count:
            item = self._real_sample(self.real[index % len(self.real)], rng)
        else:
            item = self._synthetic_sample(self.synthetic[(index - real_count) % len(self.synthetic)], rng)
        return tuple(torch.from_numpy(array.astype(np.float32, copy=False)) for array in item)


def build_rvideformer() -> RViDeformer:
    """Construct the full training architecture before loading its checkpoint."""

    return RViDeformer(
        upscale=1,
        img_size=[8, 64, 64],
        window_size=[6, 8, 8],
        depths=[4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2],
        indep_reconsts=[11, 12],
        embed_dims=[84, 84, 84, 84, 84, 84, 84, 108, 108, 108, 108],
        num_heads=[6] * 11,
        spynet_path=None,
        pa_frames=2,
        deformable_groups=12,
    )


def load_official_state(checkpoint: Path) -> dict[str, Tensor]:
    """Read the serialized official module and return its tensor state."""

    if not checkpoint.is_file():
        if (
            checkpoint.name == _RECOVERED_OFFICIAL_CHECKPOINT.name
            and _RECOVERED_OFFICIAL_CHECKPOINT.is_file()
        ):
            checkpoint = _RECOVERED_OFFICIAL_CHECKPOINT
        else:
            raise FileNotFoundError(checkpoint)
    if str(_REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_REFERENCE_ROOT))
    try:
        import models.RViDeformer_arch as source_arch
    except ModuleNotFoundError as error:
        if error.name not in {"models", "models.RViDeformer_arch"}:
            raise
        package = sys.modules.get("models")
        if package is None:
            package = types.ModuleType("models")
            package.__path__ = []
            sys.modules["models"] = package
        module_name = "models.RViDeformer_arch"
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).with_name("rvideformer_arch.py"),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load recovered module {module_name}")
        source_arch = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = source_arch
        spec.loader.exec_module(source_arch)
        package.RViDeformer_arch = source_arch
    from einops.layers.torch import Rearrange

    allowed_globals: list[object] = [
        set,
        nn.GELU,
        nn.Conv3d,
        nn.Dropout,
        nn.ModuleList,
        nn.Linear,
        nn.AvgPool2d,
        nn.Softmax,
        nn.Sequential,
        nn.Identity,
        nn.LayerNorm,
        Rearrange,
        source_arch.TMSA,
        source_arch.DropPath,
        source_arch.TMSAG,
        source_arch.Stage,
        source_arch.WindowAttention,
        source_arch.Mlp_GEGLU,
        source_arch.RViDeformer,
        source_arch.RTMSA,
    ]
    with torch.serialization.safe_globals(allowed_globals):
        source_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(source_model, nn.Module):
        raise TypeError(f"checkpoint did not contain a module: {type(source_model)!r}")
    state = source_model.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def build_loaded_model(checkpoint: Path, device: torch.device) -> RViDeformer:
    model = build_rvideformer()
    state = load_official_state(checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _gradient_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    dx = torch.abs(prediction[..., :, 1:] - prediction[..., :, :-1]) - torch.abs(target[..., :, 1:] - target[..., :, :-1])
    dy = torch.abs(prediction[..., 1:, :] - prediction[..., :-1, :]) - torch.abs(target[..., 1:, :] - target[..., :-1, :])
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    return 0.5 * (_masked_mean(torch.abs(dx), mask_x) + _masked_mean(torch.abs(dy), mask_y))


def _highpass_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    pred_hp = prediction - F.avg_pool3d(prediction, (1, 3, 3), stride=1, padding=(0, 1, 1))
    target_hp = target - F.avg_pool3d(target, (1, 3, 3), stride=1, padding=(0, 1, 1))
    return _masked_mean(torch.abs(pred_hp - target_hp), mask)


def compute_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    gradient_weight: float,
    highpass_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    mask = mask.expand_as(prediction)
    charbonnier = _masked_mean(torch.sqrt((prediction - target).square() + 1e-6), mask)
    gradient = _gradient_loss(prediction, target, mask)
    highpass = _highpass_loss(prediction, target, mask)
    total = charbonnier + gradient_weight * gradient + highpass_weight * highpass
    return total, {
        "loss": float(total.detach().cpu()),
        "charbonnier": float(charbonnier.detach().cpu()),
        "gradient": float(gradient.detach().cpu()),
        "highpass": float(highpass.detach().cpu()),
    }


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: nn.Module, ema: nn.Module, optimizer: torch.optim.Optimizer, step: int, config: A3Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
    }, path)


def restore_checkpoint(
    path: Path,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    step = int(payload["step"])
    if step < 0:
        raise ValueError(f"checkpoint step must be non-negative, got {step}")
    return step


def train(config: A3Config) -> dict:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A3 requires CUDA; torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a3_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    dataset = PhaseADataset(config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = build_loaded_model(Path(config.checkpoint), device)
    ema = build_loaded_model(Path(config.checkpoint), device)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    log_path = output_dir / "train.jsonl"
    step = 0
    if config.resume_checkpoint is not None:
        step = restore_checkpoint(Path(config.resume_checkpoint), model, ema, optimizer, device)
        if step >= config.steps:
            raise ValueError(f"resume checkpoint is already at step {step}, target is {config.steps}")
    iterator = iter(loader)
    log_mode = "a" if config.resume_checkpoint is not None else "w"
    with log_path.open(log_mode, encoding="utf-8") as log:
        while step < config.steps:
            try:
                inputs, targets, masks = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets, masks = next(iterator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device).unsqueeze(2)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            prediction = prediction.clamp(0.0, 1.0)
            loss, metrics = compute_loss(prediction, targets, masks, config.gradient_weight, config.highpass_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}: {metrics}")
            loss.backward()
            optimizer.step()
            update_ema(ema, model)
            step += 1
            record = {"step": step, **metrics}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(f"[a3] step={step} loss={metrics['loss']:.6f} raw={metrics['charbonnier']:.6f}", flush=True)
    save_checkpoint(output_dir / f"checkpoint_step_{step:06d}.pth", model, ema, optimizer, step, config)
    save_checkpoint(output_dir / "latest.pth", model, ema, optimizer, step, config)
    return {"steps": step, "checkpoint": str(output_dir / f"checkpoint_step_{step:06d}.pth"), "log": str(log_path)}

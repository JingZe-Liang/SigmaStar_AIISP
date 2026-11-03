"""Prepare or explicitly run the A12 sampling-only continuation of A3 EMA."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from raw_fusion.finetune.a3 import A3Config, build_rvideformer, compute_loss, set_seed, update_ema
from raw_fusion.finetune.a12_sampling import A12SamplingDataset, A12StepSampler


REPO_ROOT = Path(__file__).resolve().parents[3]
A3_STEP5000_SHA256 = "97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b"
CHECKPOINT_SCHEMA = "a12_sampling_only_v1"


@dataclass(frozen=True, slots=True)
class A12TrainConfig(A3Config):
    """Sampling-only pilot; the loss and static targets remain those of A3."""

    steps: int = 200
    learning_rate: float = 1e-5
    samples_per_epoch: int = 512
    checkpoint_interval: int = 50
    expected_checkpoint_sha256: str = A3_STEP5000_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict:
    """Store NumPy state as primitive integers for weights_only checkpoint loading."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state: dict, require_cuda: bool = False) -> None:
    cuda_states = state["torch_cuda"]
    if require_cuda and not cuda_states:
        raise ValueError("CUDA training resume requires saved CUDA RNG states")
    if cuda_states and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("CUDA device count changed; exact RNG restoration is unavailable")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["generator"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def validate_a12_config(config: A12TrainConfig) -> Path:
    output = Path(config.output_dir).resolve()
    if output.parent != (REPO_ROOT / "artifacts").resolve() or not output.name.startswith("a12_"):
        raise ValueError("A12 output_dir must be a new artifacts/a12_* directory in this repository")
    if config.batch_size != 1 or config.patch_size != 64 or config.temporal_frames != 8:
        raise ValueError("the sampling-only pilot requires batch 1, eight frames, and 64x64 packed patches")
    if config.steps <= 0 or config.checkpoint_interval <= 0 or config.learning_rate <= 0:
        raise ValueError("steps, checkpoint_interval, and learning_rate must be positive")
    if config.gradient_weight != 0.1 or config.highpass_weight != 0.1:
        raise ValueError("the sampling-only pilot preserves both A3 structure weights at 0.1")
    if config.holdout_gain != "1000x" or config.real_fraction != 0.5:
        raise ValueError("the sampling-only pilot preserves the 1000x holdout and 1:1 real/synthetic ratio")
    if config.expected_checkpoint_sha256 != A3_STEP5000_SHA256:
        raise ValueError("this pilot must initialize from the audited A3 step-5000 checkpoint")
    if config.resume_checkpoint is not None and Path(config.resume_checkpoint).resolve().parent == output:
        raise ValueError("resume must write to a new A12 directory, preserving the source run")
    if output.exists():
        unexpected = [path.name for path in output.iterdir() if path.name != "prepare.json"]
        if unexpected:
            raise FileExistsError(f"A12 output contains existing run artifacts: {unexpected}")
    return output


def validate_resume_payload(payload: dict, config: A12TrainConfig) -> int:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("resume requires an A12 checkpoint with RNG and sampler state")
    if payload.get("source_checkpoint_sha256") != config.expected_checkpoint_sha256:
        raise ValueError("resume checkpoint has a different A3 source identity")
    saved = payload["config"]
    current = asdict(config)
    allowed_changes = {"steps", "output_dir", "resume_checkpoint"}
    mismatches = [key for key, value in current.items() if key not in allowed_changes and saved.get(key) != value]
    if mismatches:
        raise ValueError(f"resume changes training/sampling configuration: {mismatches}")
    step = int(payload["step"])
    if not 0 <= step < config.steps or payload["next_sample_index"] != step * config.batch_size:
        raise ValueError("resume step/sample position is inconsistent or already reaches the requested stop")
    for key in ("model", "ema", "optimizer", "rng"):
        if key not in payload:
            raise ValueError(f"resume checkpoint lacks {key}")
    return step


def prepare_a12(config: A12TrainConfig) -> dict:
    """Check paths, source identity, architecture, and sample plans on CPU only."""

    output = validate_a12_config(config)
    checkpoint_path = Path(config.checkpoint).resolve()
    source_sha = sha256_file(checkpoint_path)
    if source_sha != config.expected_checkpoint_sha256:
        raise ValueError(f"A3 checkpoint SHA-256 mismatch: {source_sha}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("step") != 5000 or "ema" not in payload:
        raise ValueError("initialization requires the A3 step-5000 EMA payload")
    model = build_rvideformer()
    model.load_state_dict(payload["ema"], strict=True)
    dataset = A12SamplingDataset(config, samples_per_epoch=config.samples_per_epoch)
    start_step = 0
    if config.resume_checkpoint is not None:
        resume = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=True)
        start_step = validate_resume_payload(resume, config)
        model.load_state_dict(resume["model"], strict=True)
        model.load_state_dict(resume["ema"], strict=True)
    report = {
        "schema": CHECKPOINT_SCHEMA, "status": "prepared", "training_started": False,
        "config": asdict(config), "source_checkpoint_sha256": source_sha,
        "initial_weights": "A3 step-5000 EMA for both model and EMA" if config.resume_checkpoint is None else f"A12 model/EMA/optimizer from {config.resume_checkpoint}",
        "start_step": start_step, "next_sample_index": start_step * config.batch_size,
        "remaining_updates": config.steps - start_step,
        "first_sample_plans": [asdict(dataset.sample_plan(i)) for i in range(start_step, min(start_step + 8, len(dataset)))],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "FP32", "checkpoint_interval": config.checkpoint_interval,
        "validation": "Existing A3 evaluation is unchanged; no image-quality acceptance is claimed by prepare.",
        "resume_contract": "Preserve config and runtime/device topology; restore Python, NumPy, torch CPU/CUDA RNG, model, EMA, optimizer, and absolute sample position.",
        "limitations": [
            "No training has been executed by prepare; GPU memory and training duration are unmeasured.",
            "Targets remain static and the A3 noise model is unchanged; this is a short pipeline pilot, not proof of a sampling benefit.",
            "Two hundred updates do not span a 512-sample epoch; multi-epoch coverage requires a longer controlled experiment.",
            "Causal attribution requires an equal-budget original-sampler continuation with the same EMA initialization, optimizer, and learning rate.",
            "Resume assumes unchanged RAW, reference, noise-model, manifest and mask contents; this entry point does not hash the complete training dataset.",
            "Bitwise GPU determinism is not guaranteed across hardware, software, or nondeterministic CUDA kernels.",
            "Samples retain A3 validity masks; some randomly selected patches may have very few valid pixels.",
            "Training explicitly stops on an empty validity mask; prepare only inspects plans and does not certify full pilot mask coverage.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "prepare.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous["config"] != report["config"]:
            raise FileExistsError("existing prepare.json uses a different config; use a new A12 directory")
    else:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def save_a12_checkpoint(
    path: Path, model: nn.Module, ema: nn.Module, optimizer: torch.optim.Optimizer,
    step: int, config: A12TrainConfig,
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA, "step": step, "next_sample_index": step * config.batch_size,
        "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(), "config": asdict(config),
        "source_checkpoint_sha256": config.expected_checkpoint_sha256,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)


def train_a12(config: A12TrainConfig) -> dict:
    """Run only after explicit CLI --train; prepare itself never calls this."""

    prepare_a12(config)
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the full A12 pilot requires an available CUDA device")
    torch.cuda.set_device(device.index or 0)
    set_seed(config.seed)
    dataset = A12SamplingDataset(config, samples_per_epoch=config.samples_per_epoch)
    model = build_rvideformer()
    initial = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(initial["ema"], strict=True)
    model.to(device)
    ema = deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    step = 0
    resumed_rng = None
    if config.resume_checkpoint is not None:
        payload = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=True)
        step = validate_resume_payload(payload, config)
        model.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        resumed_rng = payload["rng"]
    loader_generator = torch.Generator().manual_seed(config.seed + 12)
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=A12StepSampler(dataset, start_step=step),
                        num_workers=0, drop_last=False, generator=loader_generator)
    iterator = iter(loader)
    if resumed_rng is not None:
        restore_rng_state(resumed_rng, require_cuda=True)
    model.train()
    output = Path(config.output_dir).resolve()
    with (output / "train.jsonl").open("x", encoding="utf-8") as log:
        for inputs, targets, masks in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            masks = masks.to(device).unsqueeze(2)
            valid_fraction = float(masks.mean())
            if valid_fraction == 0.0:
                raise RuntimeError(f"sample {step} has no valid supervision; no optimizer update was performed")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs).clamp(0.0, 1.0)
            loss, metrics = compute_loss(prediction, targets, masks, config.gradient_weight, config.highpass_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite A12 loss at step {step + 1}")
            loss.backward()
            optimizer.step()
            update_ema(ema, model)
            step += 1
            record = {"step": step, "sample_index": step - 1, "valid_fraction": valid_fraction, **metrics}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % config.checkpoint_interval == 0 or step == config.steps:
                save_a12_checkpoint(output / f"checkpoint_step_{step:06d}.pth", model, ema, optimizer, step, config)
            if step == 1 or step % 10 == 0:
                print(f"[a12] step={step} loss={metrics['loss']:.6f} valid_fraction={valid_fraction:.4f}", flush=True)
    result = {"schema": CHECKPOINT_SCHEMA, "status": "training_completed", "step": step,
              "checkpoint": str(output / f"checkpoint_step_{step:06d}.pth")}
    with (output / "training_result.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result

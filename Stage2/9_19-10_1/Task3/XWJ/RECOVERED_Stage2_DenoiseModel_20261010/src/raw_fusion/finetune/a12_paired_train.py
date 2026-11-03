"""Explicit shared-prefix and fixed/refresh branches for the A12 ablation."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from raw_fusion.finetune.a3 import A3Config, build_rvideformer, compute_loss, set_seed, update_ema
from raw_fusion.finetune.a12_paired_sampling import A12PairedSamplingDataset
from raw_fusion.finetune.a12_sampling import A12StepSampler
from raw_fusion.finetune.a12_train import (
    A12TrainConfig, A3_STEP5000_SHA256, capture_rng_state, restore_rng_state, sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_SCHEMA = "a12_paired_refresh_v1"
AUDIT_SCHEMA = "a12_paired_sampling_audit_v1"


@dataclass(frozen=True, slots=True)
class A12PairedTrainConfig(A12TrainConfig):
    """Pin data and shared ancestry independently of a branch's output path."""

    steps: int = 512
    checkpoint_interval: int = 128
    phase: Literal["shared", "fixed", "refresh"] = "shared"
    purpose: Literal["experiment", "resume_verification"] = "experiment"
    parent_checkpoint_sha256: str | None = None
    audit_path: str | None = None
    expected_audit_sha256: str | None = None
    sample_overrides_path: str | None = None
    expected_sample_overrides_sha256: str | None = None


def state_digest(value: object) -> str:
    """Hash nested optimizer and RNG state without serialization storage IDs."""
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        digest.update(type(item).__name__.encode() + b"\0")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode() + b"\0")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr(item).encode() + b"\0")
        else:
            raise TypeError(f"unsupported state value for hashing: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def runtime_identity(device: torch.device) -> dict:
    return {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def validate_paired_config(config: A12PairedTrainConfig) -> Path:
    output = Path(config.output_dir).resolve()
    if output.parent != (REPO_ROOT / "artifacts").resolve() or not output.name.startswith("a12_"):
        raise ValueError("paired output must be a new artifacts/a12_* directory")
    if config.phase not in {"shared", "fixed", "refresh"}:
        raise ValueError("unknown paired phase")
    if config.purpose not in {"experiment", "resume_verification"}:
        raise ValueError("unknown paired purpose")
    if (config.batch_size, config.temporal_frames, config.patch_size, config.samples_per_epoch) != (1, 8, 64, 512):
        raise ValueError("paired training requires batch 1, 8 frames, 64 pixels, and a 512-sample cycle")
    if (config.learning_rate, config.real_fraction, config.gradient_weight, config.highpass_weight) != (1e-5, 0.5, 0.1, 0.1):
        raise ValueError("paired optimization and source ratio must preserve the approved configuration")
    if config.holdout_gain != "1000x" or config.expected_checkpoint_sha256 != A3_STEP5000_SHA256:
        raise ValueError("paired source checkpoint and holdout must remain the audited A3 baseline")
    if config.checkpoint_interval != 128:
        raise ValueError("paired checkpoints must use the approved 128-step interval")
    if config.purpose == "experiment":
        expected_steps = 512 if config.phase == "shared" else 1024
        if config.steps != expected_steps:
            raise ValueError(f"phase {config.phase} must stop at cumulative step {expected_steps}")
    else:
        if config.phase != "shared" or not 1 <= config.steps <= 4:
            raise ValueError("resume verification is restricted to at most four shared-phase steps")
    if config.phase != "shared" and config.resume_checkpoint is None:
        raise ValueError("both experimental branches must fork from a shared checkpoint")
    if (config.resume_checkpoint is None) != (config.parent_checkpoint_sha256 is None):
        raise ValueError("resume checkpoint and its expected SHA-256 must be supplied together")
    if config.resume_checkpoint is not None and Path(config.resume_checkpoint).resolve().parent == output:
        raise ValueError("resume must use a new output directory")
    if not config.audit_path or not config.expected_audit_sha256:
        raise ValueError("paired training requires a hash-pinned complete CPU sampling audit")
    if (config.sample_overrides_path is None) != (config.expected_sample_overrides_sha256 is None):
        raise ValueError("sample overrides and their expected SHA-256 must be supplied together")
    if output.exists() and any(path.name != "prepare.json" for path in output.iterdir()):
        raise FileExistsError("paired output already contains run artifacts")
    return output


def validate_paired_resume_payload(payload: dict, config: A12PairedTrainConfig) -> int:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("resume requires the paired checkpoint schema")
    if payload.get("source_checkpoint_sha256") != config.expected_checkpoint_sha256:
        raise ValueError("resume source checkpoint identity differs")
    for key in ("model", "ema", "optimizer", "rng", "runtime", "fork_parent_checkpoint_sha256"):
        if key not in payload:
            raise ValueError(f"resume checkpoint lacks {key}")
    saved = payload["config"]
    current = asdict(config)
    if set(saved) != set(current):
        raise ValueError("resume configuration schema differs")
    step = int(payload["step"])
    if not 0 < step < config.steps or payload["next_sample_index"] != step:
        raise ValueError("resume step and sample position are inconsistent")
    allowed = {"steps", "output_dir", "resume_checkpoint", "parent_checkpoint_sha256"}
    if saved["phase"] != config.phase:
        if saved["phase"] != "shared" or config.phase not in {"fixed", "refresh"}:
            raise ValueError("only a shared checkpoint can fork into an experimental branch")
        if saved["purpose"] != "experiment" or config.purpose != "experiment" or step != 512 or saved["steps"] != 512:
            raise ValueError("a fork requires the completed experimental shared step-512 checkpoint")
        if payload["fork_parent_checkpoint_sha256"] is not None:
            raise ValueError("the shared checkpoint cannot already have branch ancestry")
        allowed.add("phase")
    else:
        if config.phase != "shared" and not payload["fork_parent_checkpoint_sha256"]:
            raise ValueError("branch continuation lacks its original shared checkpoint identity")
    mismatches = [key for key in current if key not in allowed and current[key] != saved[key]]
    if mismatches:
        raise ValueError(f"resume changes paired training configuration: {mismatches}")
    return step


def load_paired_audit(config: A12PairedTrainConfig) -> tuple[dict, dict]:
    if config.audit_path is None or config.expected_audit_sha256 is None:
        raise ValueError("sampling audit identity is missing")
    path = Path(config.audit_path)
    if sha256_file(path) != config.expected_audit_sha256:
        raise ValueError("sampling audit SHA-256 mismatch")
    audit = json.loads(path.read_text(encoding="utf-8"))
    required_checks = {
        "first_cycle_matches_a12_plans", "identical_repeating_source_record_gain",
        "exact_source_ratio_every_cycle", "first_cycle_tensor_identity", "fixed_cycle_tensor_replay",
        "refresh_changes_real_inputs", "refresh_changes_synthetic_inputs", "valid_tensor_contents", "no_empty_supervision",
    }
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != "passed":
        raise ValueError("the complete paired sampling audit has not passed")
    for key in ("sample_overrides_path", "expected_sample_overrides_sha256"):
        if audit[key] != getattr(config, key):
            raise ValueError(f"sampling audit override identity mismatch: {key}")
    if config.sample_overrides_path is not None:
        if sha256_file(Path(config.sample_overrides_path)) != config.expected_sample_overrides_sha256:
            raise ValueError("sample overrides SHA-256 mismatch")
    if not required_checks <= audit["checks"].keys() or not all(audit["checks"].values()):
        raise ValueError("sampling audit checks are missing or failed")
    if audit["samples_per_epoch"] != 512 or audit["cycles"] != 2:
        raise ValueError("sampling audit does not cover two complete cycles")
    ignored = {"steps", "device", "output_dir", "resume_checkpoint"}
    current = asdict(config)
    mismatches = [field.name for field in fields(A3Config)
                  if field.name not in ignored and audit["config"].get(field.name) != current[field.name]]
    if mismatches:
        raise ValueError(f"sampling audit configuration mismatch: {mismatches}")
    for mode in ("fixed", "refresh"):
        coverage = audit["coverage"][mode]
        if coverage["samples_read"] != 1024 or coverage["zero_valid_sample_indices"]:
            raise ValueError(f"sampling audit lacks full valid coverage for {mode}")
    for source, expected in audit["metadata_sha256"].items():
        if sha256_file(Path(source)) != expected:
            raise ValueError(f"audited metadata changed: {source}")
    for record in audit["raw_files"]:
        stat = Path(record["path"]).stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"audited RAW identity changed: {record['path']}")
    manifest_path = Path(audit["sample_manifest"])
    if sha256_file(manifest_path) != audit["sample_manifest_sha256"]:
        raise ValueError("audited sample manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for mode in ("fixed", "refresh"):
        if len(manifest["modes"][mode]) != 1024:
            raise ValueError("audited sample manifest has incomplete coverage")
    return audit, manifest


def prepare_paired(config: A12PairedTrainConfig) -> dict:
    """Validate a run on CPU, including its full data audit and ancestry."""
    output = validate_paired_config(config)
    audit, manifest = load_paired_audit(config)
    if sha256_file(Path(config.checkpoint)) != config.expected_checkpoint_sha256:
        raise ValueError("A3 source checkpoint SHA-256 mismatch")
    initial = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    if initial.get("step") != 5000 or "ema" not in initial:
        raise ValueError("initialization requires the A3 step-5000 EMA")
    model = build_rvideformer()
    model.load_state_dict(initial["ema"], strict=True)
    sampling_mode = "fixed" if config.phase == "fixed" else "refresh"
    overrides_path = Path(config.sample_overrides_path) if config.sample_overrides_path is not None else None
    dataset = A12PairedSamplingDataset(replace(config, steps=1024), sampling_mode, config.samples_per_epoch,
                                       plan_overrides_path=overrides_path)
    source_schedule = []
    for index in range(512):
        plan = dataset.sample_plan(index)
        source_schedule.append({"position": plan.epoch_index, "source": plan.source,
                                "record_index": plan.record_index, "gain": plan.gain})
    schedule_sha = hashlib.sha256(json.dumps(source_schedule, sort_keys=True).encode()).hexdigest()
    if schedule_sha != audit["source_schedule_sha256"]:
        raise ValueError("training source schedule differs from the CPU audit")
    for index in range(1024):
        if asdict(dataset.sample_plan(index)) != manifest["modes"][sampling_mode][index]["plan"]:
            raise ValueError(f"training sample plan differs from the audit at {index}")
    start_step = 0
    fork_parent_sha = None
    if config.resume_checkpoint is not None:
        if sha256_file(Path(config.resume_checkpoint)) != config.parent_checkpoint_sha256:
            raise ValueError("parent checkpoint SHA-256 mismatch")
        resume = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=True)
        start_step = validate_paired_resume_payload(resume, config)
        model.load_state_dict(resume["model"], strict=True)
        model.load_state_dict(resume["ema"], strict=True)
        if resume["config"]["phase"] == "shared" and config.phase != "shared":
            fork_parent_sha = config.parent_checkpoint_sha256
        else:
            fork_parent_sha = resume["fork_parent_checkpoint_sha256"]
    report = {
        "schema": CHECKPOINT_SCHEMA, "status": "prepared", "training_started": False,
        "config": asdict(config), "sampling_mode": sampling_mode,
        "start_step": start_step, "remaining_updates": config.steps - start_step,
        "source_checkpoint_sha256": config.expected_checkpoint_sha256,
        "fork_parent_checkpoint_sha256": fork_parent_sha,
        "audit_sha256": config.expected_audit_sha256, "source_schedule_sha256": schedule_sha,
        "sample_manifest_sha256": audit["sample_manifest_sha256"],
        "sample_overrides_path": config.sample_overrides_path,
        "sample_overrides_sha256": config.expected_sample_overrides_sha256,
        "precision": "FP32", "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "data_contract": "Recheck audited input/target/mask hashes before every optimizer update.",
        "oom_policy": "stop_without_retry",
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "prepare.json"
    if report_path.exists():
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise FileExistsError("prepare report differs; choose a new output directory")
    else:
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    return report


def save_paired_checkpoint(
    path: Path, model: nn.Module, ema: nn.Module, optimizer: torch.optim.Optimizer,
    step: int, config: A12PairedTrainConfig, runtime: dict, fork_parent_sha: str | None,
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA, "step": step, "next_sample_index": step,
        "config": asdict(config), "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "rng": capture_rng_state(), "runtime": runtime,
        "source_checkpoint_sha256": config.expected_checkpoint_sha256,
        "fork_parent_checkpoint_sha256": fork_parent_sha,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)


def _execute_training(config: A12PairedTrainConfig, device: torch.device, preparation: dict) -> dict:
    """Execute the same loop for CUDA runs and CPU fixture-based recovery tests."""
    set_seed(config.seed)
    _, manifest = load_paired_audit(config)
    mode = "fixed" if config.phase == "fixed" else "refresh"
    overrides_path = Path(config.sample_overrides_path) if config.sample_overrides_path is not None else None
    dataset = A12PairedSamplingDataset(config, mode, config.samples_per_epoch, plan_overrides_path=overrides_path)
    model = build_rvideformer()
    initial = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(initial["ema"], strict=True)
    model.to(device)
    ema = deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    runtime = runtime_identity(device)
    step = 0
    resumed_rng = None
    if config.resume_checkpoint is not None:
        if sha256_file(Path(config.resume_checkpoint)) != config.parent_checkpoint_sha256:
            raise ValueError("parent checkpoint changed after preparation")
        payload = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=True)
        step = validate_paired_resume_payload(payload, config)
        if payload["runtime"] != runtime:
            raise ValueError("runtime/device identity changed across checkpoint restoration")
        model.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        resumed_rng = payload["rng"]
    loader = DataLoader(dataset, batch_size=1, sampler=A12StepSampler(dataset, start_step=step),
                        num_workers=0, drop_last=False, generator=torch.Generator().manual_seed(config.seed + 12))
    iterator = iter(loader)
    if resumed_rng is not None:
        restore_rng_state(resumed_rng, require_cuda=device.type == "cuda")
    model.train()
    output = Path(config.output_dir)
    start_step = step
    initial_state = {
        "step": step, "phase": config.phase, "runtime": runtime,
        "fork_parent_checkpoint_sha256": preparation["fork_parent_checkpoint_sha256"],
        "state_sha256": {"model": state_digest(model.state_dict()), "ema": state_digest(ema.state_dict()),
                         "optimizer": state_digest(optimizer.state_dict()), "rng": state_digest(capture_rng_state())},
    }
    with (output / "initial_state.json").open("x", encoding="utf-8") as stream:
        json.dump(initial_state, stream, indent=2)
        stream.write("\n")
    with (output / "train.jsonl").open("x", encoding="utf-8") as log:
        for inputs, targets, masks in iterator:
            actual_hashes = [hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for tensor in (inputs, targets, masks)]
            expected_sample = manifest["modes"][mode][step]
            if actual_hashes != expected_sample["tensor_sha256"]:
                raise ValueError(f"training data differs from the CPU audit at sample {step}")
            valid_fraction = float(masks.mean())
            if valid_fraction == 0.0:
                raise RuntimeError(f"sample {step} has no valid supervision")
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device).unsqueeze(2)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs).clamp(0.0, 1.0)
            loss, metrics = compute_loss(prediction, targets, masks, config.gradient_weight, config.highpass_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite paired loss at step {step + 1}")
            loss.backward()
            optimizer.step()
            update_ema(ema, model)
            step += 1
            record = {"step": step, "sample_index": step - 1, "phase": config.phase,
                      "source": expected_sample["plan"]["source"], "gain": expected_sample["plan"]["gain"],
                      "valid_fraction": valid_fraction, **metrics}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if step % config.checkpoint_interval == 0 or step == config.steps:
                save_paired_checkpoint(output / f"checkpoint_step_{step:06d}.pth", model, ema, optimizer,
                                       step, config, runtime, preparation["fork_parent_checkpoint_sha256"])
            if step == start_step + 1 or step % 16 == 0:
                print(f"[a12 paired] phase={config.phase} step={step} loss={metrics['loss']:.6f} valid={valid_fraction:.4f}", flush=True)
    if step != config.steps:
        raise RuntimeError(f"paired data loader stopped at {step}, expected {config.steps}")
    result = {
        "schema": CHECKPOINT_SCHEMA, "status": "training_completed", "phase": config.phase,
        "purpose": config.purpose, "start_step": start_step, "step": step, "completed_updates": step - start_step,
        "checkpoint": str(output / f"checkpoint_step_{step:06d}.pth"),
        "fork_parent_checkpoint_sha256": preparation["fork_parent_checkpoint_sha256"], "runtime": runtime,
    }
    with (output / "training_result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def train_paired(config: A12PairedTrainConfig) -> dict:
    preparation = prepare_paired(config)
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paired training requires an explicitly selected available CUDA device")
    if device.index is None:
        raise ValueError("paired CUDA device must include its verified device index")
    torch.cuda.set_device(device)
    return _execute_training(config, device, preparation)

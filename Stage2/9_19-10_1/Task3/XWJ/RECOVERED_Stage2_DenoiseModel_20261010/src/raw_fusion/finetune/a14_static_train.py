"""Frozen, balanced static supervision from MIS20S1, SC465SL, and sc635hai."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor

from raw_fusion.finetune.a3 import A3Config, build_rvideformer, compute_loss, set_seed, update_ema
from raw_fusion.finetune.a8_adaptation import A8AdaptationConfig, StaticSplitHalfDataset, sha256_file
from raw_fusion.finetune.a12_sampling import A12SamplingDataset
from raw_fusion.finetune.a12_train import A3_STEP5000_SHA256, capture_rng_state
from raw_fusion.finetune.cross_sensor import NativeSequenceSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "a14_static_training_v1"
EXPECTED_GPU_UUID = "GPU-07a5046e-9b56-8557-3845-d6cca9ea0497"
TENSOR_NAMES = ("inputs", "targets", "masks")


def canonical_sha256(payload: dict | list) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_training_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_training_config(config)
    return config


def validate_training_config(config: dict) -> None:
    if config["schema"] != SCHEMA:
        raise ValueError("A14 requires its explicit static-training schema")
    if config["a3_checkpoint_sha256"] != A3_STEP5000_SHA256:
        raise ValueError("A14 must start from the original A3 step-5000 EMA")
    if config["expected_gpu_uuid"] != EXPECTED_GPU_UUID:
        raise ValueError("A14 is authorized on the specified RTX 3090 only")
    if config["precision"] != "bfloat16":
        raise ValueError("the frozen A14 protocol requires BF16 autocast and FP32 loss")
    if config["steps"] <= 0 or config["steps"] % 6:
        raise ValueError("steps must be a positive multiple of six for exact source/subtype balance")
    if config["checkpoint_interval"] <= 0 or config["steps"] % config["checkpoint_interval"]:
        raise ValueError("checkpoint_interval must be positive and divide steps")
    if config["seed"] < 0 or config["max_candidates_per_slot"] <= 0:
        raise ValueError("seed must be nonnegative and candidate budget positive")
    for name in ("learning_rate", "gradient_clip_norm", "min_valid_fraction", "ema_decay"):
        if not math.isfinite(config[name]) or config[name] <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config["min_valid_fraction"] > 1.0 or config["ema_decay"] >= 1.0:
        raise ValueError("invalid coverage threshold or EMA decay")
    if config["gradient_weight"] != 0.1 or config["highpass_weight"] != 0.1:
        raise ValueError("A14 preserves both original A3 structural loss weights at 0.1")
    for name in ("batch_size", "patch_size", "temporal_frames"):
        if name in config and config[name] != {"batch_size": 1, "patch_size": 64, "temporal_frames": 8}[name]:
            raise ValueError(f"A14 has a fixed {name}")
    sources = config["static_sources"]
    if len(sources) != 2 or len({source["name"] for source in sources}) != 2:
        raise ValueError("A14 requires exactly two distinct static split-half sources")
    expected = {"SC465SL": (300, "256x"), "sc635hai": (150, "131x")}
    observed: set[str] = set()
    for source in sources:
        payload = dict(source["source"])
        spec = NativeSequenceSpec.from_mapping(str(payload.pop("name")), payload)
        if spec.sensor not in expected:
            raise ValueError("only the approved SC465SL/sc635hai static sources are trainable")
        frames, gain = expected[spec.sensor]
        if spec.frames_used != frames or gain not in spec.name:
            raise ValueError("static training gain/frame identity differs from the approved source")
        if not spec.path.resolve().is_relative_to(Path(config["dataset_root"]).resolve()):
            raise ValueError("static source lies outside the authoritative dataset root")
        observed.add(spec.sensor)
    if observed != set(expected):
        raise ValueError("A14 requires one source from each of SC465SL and sc635hai")
    prepared = Path(config["prepared_dir"]).resolve()
    output = Path(config["output_dir"]).resolve()
    for path in (prepared, output):
        artifacts = REPO_ROOT / "artifacts"
        if not path.is_relative_to(artifacts) or not path.relative_to(artifacts).parts or not path.relative_to(artifacts).parts[0].startswith("a14_"):
            raise ValueError("A14 writes only to dedicated artifacts/a14_* directories")
    if prepared.is_relative_to(output) or output.is_relative_to(prepared):
        raise ValueError("prepared samples and training outputs must use distinct non-nested directories")


def build_training_datasets(config: dict) -> tuple[list, list[str]]:
    capacity = config["steps"] * config["max_candidates_per_slot"] * 2
    mis_config = A3Config(
        dataset_root=config["dataset_root"], supervision_dir=config["mis_supervision_dir"],
        noise_dir=config["mis_noise_dir"], output_dir=config["output_dir"],
        checkpoint=config["a3_checkpoint"], seed=config["seed"], steps=capacity,
        learning_rate=config["learning_rate"], real_fraction=0.5, holdout_gain="1000x",
    )
    datasets: list = [A12SamplingDataset(mis_config, samples_per_epoch=512)]
    names = ["mis20s1_static_replay"]
    for position, source in enumerate(config["static_sources"], start=1):
        payload = dict(source["source"])
        spec = NativeSequenceSpec.from_mapping(str(payload.pop("name")), payload)
        adaptation = A8AdaptationConfig(
            dataset_root=config["dataset_root"], source=spec, reference_dir=source["reference_dir"],
            a3_checkpoint=config["a3_checkpoint"], a3_checkpoint_sha256=config["a3_checkpoint_sha256"],
            source_sha256=source["source_sha256"], replay_supervision_dir=config["mis_supervision_dir"],
            replay_noise_dir=config["mis_noise_dir"], output_dir=config["output_dir"],
            seed=config["seed"] + position * 1_000_003, steps=config["steps"],
            min_valid_fraction=config["min_valid_fraction"],
        )
        adaptation.validate()
        datasets.append(StaticSplitHalfDataset(adaptation, samples=capacity))
        names.append(source["name"])
    return datasets, names


def source_inventory(config: dict, datasets: list) -> list[dict]:
    paths: set[Path] = {Path(config["a3_checkpoint"]).resolve()}
    expected_hashes: dict[Path, str] = {}
    for source in config["static_sources"]:
        raw = Path(source["source"]["path"]).resolve()
        expected_hashes[raw] = source["source_sha256"]
        paths.add(raw)
        reference = Path(source["reference_dir"])
        for name in ("report.json", "full_mean_dn.npy", "even_mean_dn.npy", "odd_mean_dn.npy",
                     "even_confidence.npy", "odd_confidence.npy", "valid_region.npy",
                     "motion_threshold_per_frame.npy"):
            paths.add((reference / name).resolve())
    mis = datasets[0]
    supervision = Path(config["mis_supervision_dir"])
    reference_manifest_path = supervision / "references/a1_reference_manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    paths.add(reference_manifest_path.resolve())
    paths.add((supervision / "references/a1_reference_report.json").resolve())
    for key in ("clean_reference_path", "valid_region_path"):
        paths.add(Path(reference_manifest[key]).resolve())
    paths.add(Path(reference_manifest["fpn_paths"]["offset_dn"]).resolve())
    for name in ("noise_model.json", "synthetic_pairs_manifest.json"):
        paths.add((Path(config["mis_noise_dir"]) / name).resolve())
    for record in mis.real:
        manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
        raw = Path(manifest["path"]).resolve()
        if not raw.is_relative_to(Path(config["dataset_root"]).resolve()):
            raise ValueError("MIS RAW path lies outside the authoritative dataset root")
        paths.update((record.manifest_path.resolve(), record.motion_mask_path.resolve(), raw))
    low_gain_manifest = supervision / "stream_manifests/04_shdarkroom_10x.json"
    paths.add(low_gain_manifest.resolve())
    paths.add(Path(json.loads(low_gain_manifest.read_text(encoding="utf-8"))["path"]).resolve())
    inventory: list[dict] = []
    for path in sorted(paths):
        before = path.stat()
        digest = sha256_file(path)
        if path in expected_hashes and digest != expected_hashes[path]:
            raise ValueError(f"static source SHA mismatch: {path}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed during hashing: {path}")
        inventory.append({"path": str(path), "sha256": digest, "bytes": after.st_size,
                          "mtime_ns": after.st_mtime_ns})
    return inventory


def validate_sample(sample: tuple[Tensor, Tensor, Tensor]) -> float:
    for tensor, shape, name in zip(sample, ((8, 4, 64, 64), (8, 4, 64, 64), (8, 64, 64)),
                                   TENSOR_NAMES, strict=True):
        if tensor.shape != shape or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 with shape {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"non-finite {name}")
        if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
            raise ValueError(f"{name} must lie in [0, 1]")
    if not bool(((sample[2] == 0.0) | (sample[2] == 1.0)).all()):
        raise ValueError("masks must be binary")
    return float(sample[2].mean())


def freeze_balanced_samples(
    datasets: list, source_names: list[str], steps: int, min_valid_fraction: float,
    max_candidates_per_slot: int, samples_dir: Path,
) -> tuple[list[dict], list[dict]]:
    if len(datasets) != 3 or len(source_names) != 3 or steps <= 0 or steps % 6:
        raise ValueError("freeze requires three sources and a positive multiple of six steps")
    next_candidates = [[0, 1] for _ in datasets]
    plan: list[dict] = []
    rejected: list[dict] = []
    samples_dir.mkdir(parents=True, exist_ok=False)
    for step in range(steps):
        source_index = step % 3
        local_slot = step // 3
        parity = local_slot % 2
        dataset = datasets[source_index]
        for attempt in range(max_candidates_per_slot):
            candidate = next_candidates[source_index][parity]
            next_candidates[source_index][parity] += 2
            sample = dataset[candidate]
            valid_fraction = validate_sample(sample)
            if valid_fraction < min_valid_fraction:
                rejection = {"step": step + 1, "source": source_names[source_index],
                             "candidate_index": candidate, "valid_fraction": valid_fraction}
                if source_index == 0:
                    rejected_plan = dataset.sample_plan(candidate)
                    rejection.update({"gain": rejected_plan.gain, "kind": rejected_plan.source})
                else:
                    rejection.update({"kind": "static_split_half", "input_parity": parity})
                rejected.append(rejection)
                continue
            else:
                break
        else:
            raise RuntimeError(f"no sufficiently valid sample within budget at step {step + 1}")
        filename = f"sample_{step + 1:06d}.npz"
        arrays = {name: tensor.contiguous().numpy() for name, tensor in zip(TENSOR_NAMES, sample, strict=True)}
        with (samples_dir / filename).open("xb") as stream:
            np.savez(stream, **arrays)
        record = {
            "step": step + 1, "source_index": source_index, "source": source_names[source_index],
            "source_local_slot": local_slot, "candidate_index": candidate,
            "valid_fraction": valid_fraction, "attempts": attempt + 1,
            "sample_file": f"samples/{filename}", "sample_file_sha256": sha256_file(samples_dir / filename),
            "tensor_sha256": {name: tensor_sha256(tensor) for name, tensor in zip(TENSOR_NAMES, sample, strict=True)},
        }
        if source_index == 0:
            sample_plan = asdict(dataset.sample_plan(candidate))
            expected_kind = "real" if parity == 0 else "synthetic"
            if sample_plan["source"] != expected_kind:
                raise ValueError("MIS replacement changed the frozen real/synthetic balance")
            record.update({"kind": expected_kind, "sample_plan": sample_plan})
        else:
            record.update({"kind": "static_split_half", "input_parity": parity,
                           "target_reference_parity": 1 - parity,
                           "sampling_seed": dataset.config.seed + candidate * 104729})
        plan.append(record)
        if (step + 1) % 30 == 0 or step == 0:
            print(f"[a14 prepare] frozen={step + 1}/{steps} rejected={len(rejected)}", flush=True)
    return plan, rejected


def prepare_static_training(config: dict) -> dict:
    validate_training_config(config)
    prepared = Path(config["prepared_dir"]).resolve()
    if prepared.exists():
        raise FileExistsError(f"preparation preserves existing directories: {prepared}")
    if sha256_file(Path(config["a3_checkpoint"])) != config["a3_checkpoint_sha256"]:
        raise ValueError("A3 checkpoint SHA mismatch")
    payload = torch.load(config["a3_checkpoint"], map_location="cpu", weights_only=True)
    if payload.get("step") != 5000 or "ema" not in payload:
        raise ValueError("A14 initialization requires the A3 step-5000 EMA payload")
    model = build_rvideformer()
    model.load_state_dict(payload["ema"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, payload
    datasets, names = build_training_datasets(config)
    inventory = source_inventory(config, datasets)
    prepared.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan, rejected = freeze_balanced_samples(
        datasets, names, config["steps"], config["min_valid_fraction"],
        config["max_candidates_per_slot"], prepared / "samples",
    )
    for entry in inventory:
        stat = Path(entry["path"]).stat()
        if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise RuntimeError(f"source changed during CPU preparation: {entry['path']}")
    summary = []
    for name in names:
        rows = [row for row in plan if row["source"] == name]
        summary.append({"source": name, "samples": len(rows),
                        "kind_counts": dict(Counter(row["kind"] for row in rows)),
                        "input_parity_counts": dict(Counter(str(row["input_parity"]) for row in rows if "input_parity" in row)),
                        "valid_fraction_min": min(row["valid_fraction"] for row in rows),
                        "valid_fraction_mean": float(np.mean([row["valid_fraction"] for row in rows]))})
    mis_rows = [row for row in plan if row["source_index"] == 0]
    mis_gain_coverage = {
        kind: {"expected_gains": sorted({record.gain for record in records}),
               "accepted_counts": dict(Counter(row["sample_plan"]["gain"] for row in mis_rows if row["kind"] == kind)),
               "rejected_counts": dict(Counter(row["gain"] for row in rejected if row["source"] == names[0] and row["kind"] == kind)),
               "missing_gains": sorted({record.gain for record in records} - {row["sample_plan"]["gain"] for row in mis_rows if row["kind"] == kind})}
        for kind, records in (("real", datasets[0].real), ("synthetic", datasets[0].synthetic))
    }
    audit = {"schema": SCHEMA, "status": "prepared", "training_started": False, "config": config,
             "config_sha256": canonical_sha256(config), "plan": plan, "plan_sha256": canonical_sha256(plan),
             "source_inventory": inventory, "source_inventory_sha256": canonical_sha256(inventory),
             "source_summary": summary, "rejected_candidates": rejected,
             "mis_gain_coverage": mis_gain_coverage,
             "trainable_parameters": parameter_count, "elapsed_seconds": time.monotonic() - started,
             "loss": "Exact A3 masked Charbonnier + 0.1 gradient + 0.1 highpass; highpass uses center mask.",
             "limitations": ["Static reference supervision only; no claim of dynamic or unseen-scene generalization.",
                             "Split halves use disjoint frames; noise independence is unverified and fixed-pattern or temporally correlated residuals can remain.",
                             "MIS real cross-gain registration/exposure fits and references retain A3 limitations.",
                             "The original highpass loss can include invalid neighbors at mask boundaries.",
                             "BF16 autocast training is evaluated separately with FP32 inference."]}
    with (prepared / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return audit


def load_frozen_sample(prepared: Path, row: dict, min_valid_fraction: float) -> tuple[Tensor, Tensor, Tensor]:
    path = (prepared / row["sample_file"]).resolve()
    if not path.is_relative_to(prepared.resolve() / "samples"):
        raise ValueError("sample path escapes prepared samples")
    if sha256_file(path) != row["sample_file_sha256"]:
        raise ValueError(f"frozen sample file SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        sample = tuple(torch.from_numpy(payload[name].copy()) for name in TENSOR_NAMES)
    valid_fraction = validate_sample(sample)
    if valid_fraction < min_valid_fraction or valid_fraction != row["valid_fraction"]:
        raise ValueError("frozen sample coverage changed")
    for name, tensor in zip(TENSOR_NAMES, sample, strict=True):
        if tensor_sha256(tensor) != row["tensor_sha256"][name]:
            raise ValueError(f"frozen {name} SHA mismatch")
    return sample


def validate_prepared_audit(audit: dict, config: dict) -> None:
    if audit["schema"] != SCHEMA or audit["status"] != "prepared":
        raise ValueError("training requires a completed A14 CPU audit")
    if audit["config"] != config or audit["config_sha256"] != canonical_sha256(config):
        raise ValueError("training config differs from the frozen CPU audit")
    if audit["plan_sha256"] != canonical_sha256(audit["plan"]):
        raise ValueError("frozen sample plan SHA mismatch")
    if audit["source_inventory_sha256"] != canonical_sha256(audit["source_inventory"]):
        raise ValueError("frozen source inventory SHA mismatch")
    plan = audit["plan"]
    if len(plan) != config["steps"]:
        raise ValueError("frozen plan does not cover the configured update budget")
    seen: set[tuple[int, int]] = set()
    for position, row in enumerate(plan):
        source_index, local_slot = position % 3, position // 3
        if row["step"] != position + 1 or row["source_index"] != source_index or row["source_local_slot"] != local_slot:
            raise ValueError("frozen plan is not the prescribed balanced interleave")
        candidate = row["candidate_index"]
        if candidate < 0 or candidate % 2 != local_slot % 2 or (source_index, candidate) in seen:
            raise ValueError("frozen candidate parity/uniqueness is invalid")
        seen.add((source_index, candidate))
        if source_index == 0:
            if row["kind"] != ("real" if local_slot % 2 == 0 else "synthetic"):
                raise ValueError("MIS real/synthetic balance changed")
        else:
            if row["input_parity"] != local_slot % 2 or row["target_reference_parity"] != 1 - local_slot % 2:
                raise ValueError("static input and target parities overlap")


def train_static_model(config: dict, *, output_dir: Path | None = None) -> dict:
    validate_training_config(config)
    output = Path(config["output_dir"]).resolve() if output_dir is None else output_dir.resolve()
    validate_training_config({**config, "output_dir": str(output)})
    prepared = Path(config["prepared_dir"]).resolve()
    audit_path = prepared / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_prepared_audit(audit, config)
    if sha256_file(Path(config["a3_checkpoint"])) != config["a3_checkpoint_sha256"]:
        raise ValueError("A3 initialization checkpoint changed after preparation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["expected_gpu_uuid"]:
        raise RuntimeError("isolate the explicitly authorized GPU UUID with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training requires exactly one isolated CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    gpu_uuid = f"GPU-{properties.uuid}"
    if gpu_uuid != config["expected_gpu_uuid"]:
        raise RuntimeError("visible CUDA device UUID differs from the authorized GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the authorized GPU does not support the frozen BF16 training protocol")
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    status = {"schema": SCHEMA, "status": "running", "completed_steps": 0,
              "config": config, "audit_sha256": sha256_file(audit_path),
              "runtime_output_dir": str(output),
              "gpu_uuid": gpu_uuid, "gpu_name": properties.name,
              "precision": "BF16 autocast forward; FP32 parameters, loss, optimizer and EMA",
              "torch_version": str(torch.__version__), "oom_policy": "abort immediately; no retry or device switch"}
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    step = 0
    try:
        set_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        model = build_rvideformer()
        initial = torch.load(config["a3_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(initial["ema"], strict=True)
        del initial
        model.to(device).train().requires_grad_(True)
        ema = deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        with (output / "train.jsonl").open("x", encoding="utf-8") as log:
            for row in audit["plan"]:
                inputs, targets, masks = load_frozen_sample(prepared, row, config["min_valid_fraction"])
                inputs, targets = inputs.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
                masks = masks.unsqueeze(0).unsqueeze(2).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(inputs).clamp(0.0, 1.0)
                loss, metrics = compute_loss(prediction.float(), targets, masks,
                                             config["gradient_weight"], config["highpass_weight"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss before update {step + 1}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True,
                )
                optimizer.step()
                update_ema(ema, model, decay=config["ema_decay"])
                step += 1
                record = {"step": step, "source": row["source"], "kind": row["kind"],
                          "candidate_index": row["candidate_index"], "valid_fraction": row["valid_fraction"],
                          "gradient_norm_before_clip": float(gradient_norm), **metrics,
                          "elapsed_seconds": time.monotonic() - started,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if step % config["checkpoint_interval"] == 0 or step == config["steps"]:
                    checkpoint = {"schema": SCHEMA, "step": step, "next_plan_position": step,
                                  "model": model.state_dict(), "ema": ema.state_dict(),
                                  "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
                                  "config": config, "source_checkpoint_sha256": config["a3_checkpoint_sha256"],
                                  "audit_sha256": status["audit_sha256"], "plan_sha256": audit["plan_sha256"],
                                  "runtime": {"gpu_uuid": gpu_uuid, "precision": status["precision"],
                                              "runtime_output_dir": str(output)}}
                    with (output / f"checkpoint_step_{step:06d}.pth").open("xb") as stream:
                        torch.save(checkpoint, stream)
                if step == 1 or step % 10 == 0:
                    print(f"[a14 train] step={step}/{config['steps']} source={row['source']} loss={metrics['loss']:.6f}", flush=True)
                    status.update({"completed_steps": step, "elapsed_seconds": time.monotonic() - started})
                    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        status.update({"status": "completed", "completed_steps": step,
                       "elapsed_seconds": time.monotonic() - started,
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "final_checkpoint": str(output / f"checkpoint_step_{step:06d}.pth")})
    except torch.cuda.OutOfMemoryError as error:
        status.update({"status": "stopped_oom", "completed_steps": step,
                       "error": str(error), "elapsed_seconds": time.monotonic() - started})
        raise
    except BaseException as error:
        status.update({"status": "failed", "completed_steps": step,
                       "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status

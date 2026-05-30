from __future__ import annotations

"""Command-line orchestration for Stage 0+ AI fusion training."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW

from .data import (
    build_hard_crop_map,
    build_hard_sample_weights,
    build_loader,
    build_split_dataset,
    close_dataset,
    load_hard_sampling_records,
    summarize_hard_crop_map,
)
from .engine import run_dry_pass, train_one_epoch, validate
from .losses import LossConfig
from .optim import build_model, build_parameter_groups, build_scheduler
from .persistence import append_jsonl, format_log, load_checkpoint, load_model_weights, save_checkpoint, vars_for_json, write_json
from .tensor_utils import resolve_amp_dtype, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MambaFusionWeightNet-Lite on H5 RAW fusion data.")

    parser.add_argument("--data-root", type=Path, default=Path("H5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ai_fusion_stage0"))
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene ids or names, e.g. 1 2 scene_3.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--val-split", choices=("val", "test"), default="val")

    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--val-crop-size", type=int, default=None, help="Defaults to crop-size. Use 0 for full frames.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--hard-sampling-index", type=Path, default=None, help="JSON produced by build_hard_sampling_index.py.")
    parser.add_argument(
        "--hard-sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sample_weight from --hard-sampling-index with WeightedRandomSampler.",
    )
    parser.add_argument("--hard-sample-weight-key", default="sample_weight")
    parser.add_argument("--hard-sample-default-weight", type=float, default=1.0)
    parser.add_argument("--hard-sampler-num-samples", type=int, default=None)
    parser.add_argument(
        "--hard-sampler-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample with replacement when hard sampler is enabled.",
    )
    parser.add_argument("--hard-crop-prob", type=float, default=0.0)
    parser.add_argument("--hard-crop-min-sample-weight", type=float, default=2.0)
    parser.add_argument("--hard-crop-min-gap-weight", type=float, default=0.4)
    parser.add_argument("--hard-crop-top-k", type=int, default=8, help="Use <=0 to keep all crop candidates per frame.")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum optimizer steps. Use 0 to train full epochs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", default="auto", choices=("auto", "bf16", "fp16", "none"))

    parser.add_argument("--weight-mode", default="w4", choices=("w1", "w4"))
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--mamba-state-dim", type=int, default=8)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-scan-backend", default="mamba_ssm", choices=("auto", "reference", "mamba_ssm"))
    parser.add_argument("--weight-bias-init", type=float, default=0.0)
    parser.add_argument("--cfa-pattern", default="GBRG")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--lambda-tv", type=float, default=0.005)
    parser.add_argument("--lambda-plane", type=float, default=0.0)
    parser.add_argument("--lambda-oracle", type=float, default=0.0)
    parser.add_argument("--oracle-loss-mode", default="analytic", choices=("analytic", "soft_winner"))
    parser.add_argument("--soft-oracle-tau", type=float, default=3e-5)
    parser.add_argument("--soft-oracle-margin", type=float, default=1e-4)
    parser.add_argument("--soft-oracle-min-delta", type=float, default=1e-6)
    parser.add_argument("--soft-oracle-2dnr-weight", type=float, default=2.0)
    parser.add_argument("--soft-oracle-3dnr-weight", type=float, default=0.5)
    parser.add_argument("--lambda-diversity", type=float, default=0.0)
    parser.add_argument("--motion-weight-alpha", type=float, default=0.0)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None, help="Load model weights only; reset optimizer/scheduler.")
    parser.add_argument("--dry-run", action="store_true", help="Run one forward/backward pass and exit.")
    return parser.parse_args()


def validate_hard_sampling_args(args: argparse.Namespace) -> None:
    if args.hard_sampling_index is None:
        if args.hard_crop_prob > 0.0:
            raise ValueError("--hard-crop-prob requires --hard-sampling-index.")
        return

    if not args.hard_sampling_index.exists():
        raise FileNotFoundError(f"Hard-sampling index not found: {args.hard_sampling_index}")
    if not 0.0 <= args.hard_crop_prob <= 1.0:
        raise ValueError("--hard-crop-prob must be in [0, 1].")
    if args.hard_sample_default_weight <= 0.0:
        raise ValueError("--hard-sample-default-weight must be > 0.")
    if args.hard_sampler_num_samples is not None and args.hard_sampler_num_samples <= 0:
        raise ValueError("--hard-sampler-num-samples must be > 0 when provided.")
    if args.hard_crop_min_sample_weight < 0.0:
        raise ValueError("--hard-crop-min-sample-weight must be >= 0.")
    if args.hard_crop_min_gap_weight < 0.0:
        raise ValueError("--hard-crop-min-gap-weight must be >= 0.")


def validate_oracle_args(args: argparse.Namespace) -> None:
    if args.lambda_oracle < 0.0:
        raise ValueError("--lambda-oracle must be >= 0.")
    if args.soft_oracle_tau <= 0.0:
        raise ValueError("--soft-oracle-tau must be > 0.")
    if args.soft_oracle_margin <= 0.0:
        raise ValueError("--soft-oracle-margin must be > 0.")
    if args.soft_oracle_min_delta < 0.0:
        raise ValueError("--soft-oracle-min-delta must be >= 0.")
    if args.soft_oracle_2dnr_weight < 0.0:
        raise ValueError("--soft-oracle-2dnr-weight must be >= 0.")
    if args.soft_oracle_3dnr_weight < 0.0:
        raise ValueError("--soft-oracle-3dnr-weight must be >= 0.")
    if args.soft_oracle_2dnr_weight == 0.0 and args.soft_oracle_3dnr_weight == 0.0:
        raise ValueError("At least one soft-oracle direction weight must be > 0.")


def main() -> None:
    args = parse_args()
    validate_hard_sampling_args(args)
    validate_oracle_args(args)
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from are mutually exclusive.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", vars_for_json(args))

    hard_records = None
    hard_crop_map = None
    if args.hard_sampling_index is not None:
        hard_records = load_hard_sampling_records(args.hard_sampling_index)
        if args.hard_crop_prob > 0.0:
            hard_crop_map = build_hard_crop_map(
                hard_records,
                min_sample_weight=args.hard_crop_min_sample_weight,
                min_gap_weight=args.hard_crop_min_gap_weight,
                top_k=args.hard_crop_top_k,
            )

    train_dataset, train_stats = build_split_dataset(
        root_dir=args.data_root,
        split=args.split,
        scenes=args.scenes,
        crop_size=args.crop_size,
        random_crop=True,
        cfa_pattern=args.cfa_pattern,
        seed=args.seed,
        hard_crop_map=hard_crop_map,
        hard_crop_prob=args.hard_crop_prob,
    )
    val_crop_size = args.crop_size if args.val_crop_size is None else args.val_crop_size
    if val_crop_size <= 0:
        val_crop_size = None
    val_dataset, val_stats = build_split_dataset(
        root_dir=args.data_root,
        split=args.val_split,
        scenes=args.scenes,
        crop_size=val_crop_size,
        random_crop=False,
        cfa_pattern=args.cfa_pattern,
        seed=args.seed,
    )

    train_sample_weights = None
    hard_sampling_stats = None
    if hard_records is not None and args.hard_sampler:
        train_sample_weights, hard_sampling_stats = build_hard_sample_weights(
            train_dataset,
            hard_records,
            weight_key=args.hard_sample_weight_key,
            default_weight=args.hard_sample_default_weight,
        )

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_sample_weights is None,
        seed=args.seed,
        sample_weights=train_sample_weights,
        sampler_num_samples=args.hard_sampler_num_samples,
        sampler_replacement=args.hard_sampler_replacement,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed + 1000,
    )

    model = build_model(args).to(device)
    optimizer = AdamW(
        build_parameter_groups(model, weight_decay=args.weight_decay, lr=args.lr),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    requested_steps = args.epochs * steps_per_epoch
    total_steps = min(args.max_steps, requested_steps) if args.max_steps > 0 else requested_steps
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        base_lr=args.lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    start_epoch = 0
    global_step = 0
    best_score = -float("inf")
    if args.resume is not None:
        start_epoch, global_step, best_score = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    elif args.init_from is not None:
        load_model_weights(args.init_from, model=model, device=device)

    run_summary = {
        "device": str(device),
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else "none",
        "train": asdict(train_stats),
        "val": asdict(val_stats),
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
    }
    if hard_records is not None:
        run_summary["hard_sampling"] = {
            "index": str(args.hard_sampling_index),
            "sampler_enabled": train_sample_weights is not None,
            "sampler": hard_sampling_stats,
            "hard_crop_prob": args.hard_crop_prob,
            "hard_crop": summarize_hard_crop_map(hard_crop_map),
        }
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))

    loss_config = LossConfig(
        lambda_tv=args.lambda_tv,
        lambda_plane=args.lambda_plane,
        lambda_oracle=args.lambda_oracle,
        lambda_diversity=args.lambda_diversity,
        motion_weight_alpha=args.motion_weight_alpha,
        oracle_loss_mode=args.oracle_loss_mode,
        soft_oracle_tau=args.soft_oracle_tau,
        soft_oracle_margin=args.soft_oracle_margin,
        soft_oracle_min_delta=args.soft_oracle_min_delta,
        soft_oracle_2dnr_weight=args.soft_oracle_2dnr_weight,
        soft_oracle_3dnr_weight=args.soft_oracle_3dnr_weight,
    )

    try:
        if args.dry_run:
            run_dry_pass(model, train_loader, optimizer, scaler, loss_config, device, amp_dtype, args.grad_clip_norm)
            return

        for epoch in range(start_epoch, args.epochs):
            train_metrics, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loss_config=loss_config,
                device=device,
                amp_dtype=amp_dtype,
                epoch=epoch,
                global_step=global_step,
                total_steps=total_steps,
                grad_accum_steps=max(1, args.grad_accum_steps),
                grad_clip_norm=args.grad_clip_norm,
                log_every=args.log_every,
            )
            append_jsonl(args.output_dir / "train_log.jsonl", {"epoch": epoch + 1, "step": global_step, **train_metrics})

            should_validate = (epoch + 1) % args.val_every_epochs == 0 or global_step >= total_steps
            if should_validate:
                val_metrics = validate(model, val_loader, device=device, amp_dtype=amp_dtype)
                score = val_metrics["psnr"] + 0.5 * val_metrics["motion_psnr_gain_dnr3"]
                val_record = {"epoch": epoch + 1, "step": global_step, "score": score, **val_metrics}
                append_jsonl(args.output_dir / "val_log.jsonl", val_record)
                print(format_log("val", val_record))
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_score=best_score,
                        args=args,
                        train_stats=train_stats,
                        val_stats=val_stats,
                    )

            if (epoch + 1) % args.save_every_epochs == 0 or global_step >= total_steps:
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_score=best_score,
                    args=args,
                    train_stats=train_stats,
                    val_stats=val_stats,
                )

            if global_step >= total_steps:
                break
    finally:
        close_dataset(train_dataset)
        close_dataset(val_dataset)

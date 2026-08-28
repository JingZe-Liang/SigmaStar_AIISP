from __future__ import annotations

"""Evaluate AI fusion checkpoints on val/test splits without training."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .training.data import build_loader, build_split_dataset, close_dataset
from .training.engine import validate
from .training.optim import build_model
from .training.persistence import append_jsonl, format_log, load_model_weights, vars_for_json, write_json
from .training.tensor_utils import resolve_amp_dtype, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MambaFusionWeightNet-Lite on H5 RAW fusion data.")

    parser.add_argument("--data-root", type=Path, default=Path("H5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ai_fusion_eval"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene ids or names, e.g. 1 2 scene_3.")

    parser.add_argument("--crop-size", type=int, default=384, help="Use 0 for full frames.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", vars_for_json(args))

    crop_size = None if args.crop_size <= 0 else args.crop_size
    dataset, stats = build_split_dataset(
        root_dir=args.data_root,
        split=args.split,
        scenes=args.scenes,
        crop_size=crop_size,
        random_crop=False,
        cfa_pattern=args.cfa_pattern,
        seed=args.seed,
    )
    loader = build_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    model = build_model(args).to(device)
    load_model_weights(args.checkpoint, model=model, device=device)

    print(
        json.dumps(
            {
                "device": str(device),
                "amp_dtype": str(amp_dtype) if amp_dtype is not None else "none",
                "split": asdict(stats),
                "checkpoint": str(args.checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    try:
        metrics = validate(model, loader, device=device, amp_dtype=amp_dtype)
        score = metrics["psnr"] + 0.5 * metrics["motion_psnr_gain_dnr3"]
        record = {"split": args.split, "score": score, **metrics}
        append_jsonl(args.output_dir / "eval_log.jsonl", record)
        print(format_log("eval", record))
    finally:
        close_dataset(dataset)


if __name__ == "__main__":
    main()

"""Primary command-line entry points for the data-first V2 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .schemas.common import ContractError


_DEFAULT_ROOT = Path("/data1/wangzepu/Jaime")


def _parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=prog, description=description, allow_abbrev=False)


def _run(callable_, argv: list[str] | None) -> int:
    try:
        callable_(argv)
        return 0
    except SystemExit as error:
        return int(error.code)
    except (ContractError, OSError, RuntimeError, FloatingPointError, TypeError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


def dataset_validate(argv: list[str] | None = None) -> int:
    def run(values) -> None:
        parser = _parser("raw-fusion-v2-dataset-validate", "Validate the data-first dataset")
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--allowed-root", type=Path, default=_DEFAULT_ROOT)
        args = parser.parse_args(values)
        from .dataset import load_dataset_v2
        load_dataset_v2(args.config, allowed_root=args.allowed_root, validate_assets=True)
    return _run(run, argv)


def split_validate(argv: list[str] | None = None) -> int:
    def run(values) -> None:
        parser = _parser("raw-fusion-v2-split-validate", "Validate the data-first split")
        parser.add_argument("--config", required=True, type=Path)
        args = parser.parse_args(values)
        from .dataset import load_split_v2
        load_split_v2(args.config)
    return _run(run, argv)


def _dataset(args):
    from .data_first_dataset import DataFirstDataset
    from .data_first_supervision import MOG2SupervisionConfig
    from .md import Mog2ConfigV2
    return DataFirstDataset.from_paths(args.dataset, args.split, MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)), allowed_root=args.allowed_root, mog2_cache=getattr(args, "mog2_cache", None))


def mog2_cache_generate(argv: list[str] | None = None) -> int:
    def run(values) -> None:
        parser = _parser("raw-fusion-v2-mog2-cache-generate", "Generate parallel MOG2 supervision cache")
        parser.add_argument("--dataset", required=True, type=Path)
        parser.add_argument("--split", required=True, type=Path)
        parser.add_argument("--output-dir", required=True, type=Path)
        parser.add_argument("--allowed-root", type=Path, default=_DEFAULT_ROOT)
        parser.add_argument("--workers", type=int)
        args = parser.parse_args(values)
        from .data_first_mog2_cache import generate_mog2_cache
        result = generate_mog2_cache(args.dataset, args.split, args.output_dir, workers=args.workers, allowed_root=args.allowed_root)
        print(json.dumps({"protocol": "raw_fusion_v2_data_first", "cache": str(result.root), "completed": result.completed, "total": result.total, "elapsed_seconds": result.elapsed_seconds}, sort_keys=True))
    return _run(run, argv)


def train(argv: list[str] | None = None) -> int:
    def run(values) -> None:
        parser = _parser("raw-fusion-v2-train", "Train the data-first fusion network")
        parser.add_argument("--data-first", action="store_true")
        parser.add_argument("--dataset", required=True, type=Path)
        parser.add_argument("--split", required=True, type=Path)
        parser.add_argument("--output-dir", required=True, type=Path)
        parser.add_argument("--allowed-root", type=Path, default=_DEFAULT_ROOT)
        parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        parser.add_argument("--seed", type=int, default=20260826)
        parser.add_argument("--batch-size", type=int, default=2)
        parser.add_argument("--max-steps", type=int, default=1000)
        parser.add_argument("--log-interval", type=int, default=1)
        parser.add_argument("--checkpoint-interval", type=int, default=25)
        parser.add_argument("--resume", type=Path)
        parser.add_argument("--mog2-cache", type=Path)
        args = parser.parse_args(values)
        if not args.data_first:
            raise ContractError("the data-first protocol requires --data-first")
        from .data_first_train import run_data_first_training
        result = run_data_first_training(
            _dataset(args),
            args.output_dir,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
        )
        print(json.dumps({"protocol": "raw_fusion_v2_data_first", "checkpoint": str(result.checkpoint), "global_step": result.global_step}, sort_keys=True))
    return _run(run, argv)


def infer(argv: list[str] | None = None) -> int:
    def run(values) -> None:
        parser = _parser("raw-fusion-v2-infer", "Run data-first inference without MD")
        parser.add_argument("--data-first", action="store_true")
        parser.add_argument("--checkpoint", required=True, type=Path)
        parser.add_argument("--dataset", required=True, type=Path)
        parser.add_argument("--split", required=True, type=Path)
        parser.add_argument("--output-dir", required=True, type=Path)
        parser.add_argument("--allowed-root", type=Path, default=_DEFAULT_ROOT)
        parser.add_argument("--max-frames", type=int)
        args = parser.parse_args(values)
        if not args.data_first:
            raise ContractError("the data-first protocol requires --data-first")
        from .data_first_infer import run_data_first_inference
        result = run_data_first_inference(args.checkpoint, _dataset(args), args.output_dir, max_frames=args.max_frames)
        print(json.dumps({"protocol": "raw_fusion_v2_data_first", "frames": result.frames, "fallback_fraction": result.fallback_fraction}, sort_keys=True))
    return _run(run, argv)


def compare(argv: list[str] | None = None) -> int:
    def run(values) -> None:
        parser = _parser("raw-fusion-v2-compare", "Compare output with denoised")
        parser.add_argument("--prediction", required=True, type=Path)
        parser.add_argument("--denoised", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        args = parser.parse_args(values)
        import numpy as np
        from .data_first_compare import compare_against_denoised
        baselines = {condition: np.load(args.denoised / f"{condition}.npy") for condition in ("128x", "645x") if (args.denoised / f"{condition}.npy").is_file()}
        compare_against_denoised(args.prediction, baselines, args.output)
    return _run(run, argv)


__all__ = ["compare", "dataset_validate", "infer", "mog2_cache_generate", "split_validate", "train"]

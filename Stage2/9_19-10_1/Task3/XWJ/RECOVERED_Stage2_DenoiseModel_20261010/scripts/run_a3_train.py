"""Run the Phase A RViDeformer fine-tuning experiment."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a3 import A3Config, train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset"))
    parser.add_argument("--supervision-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/supervision")
    parser.add_argument("--noise-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/noise_model")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/finetune_mis20s1/training")
    parser.add_argument("--checkpoint", type=Path, default=Path("/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/checkpoints/RViDeformer-L_CRVD.pth"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--holdout-gain", default="1000x")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-temporal-variants", type=int, default=8)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    config = A3Config(
        dataset_root=str(args.dataset_root),
        supervision_dir=str(args.supervision_dir),
        noise_dir=str(args.noise_dir),
        output_dir=str(args.output_dir),
        checkpoint=str(args.checkpoint),
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        holdout_gain=args.holdout_gain,
        device=args.device,
        synthetic_temporal_variants=args.synthetic_temporal_variants,
        resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
    )
    print(f"[a3] config={config}", flush=True)
    result = train(config)
    print(f"[a3] completed={result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

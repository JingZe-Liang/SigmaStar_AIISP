"""Build the A2 calibration noise model, scene PTC, and synthetic pairs."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.noise import build_a2_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "finetune_mis20s1")
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    report = build_a2_artifacts(args.dataset_root, args.out, args.chunk_frames, args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

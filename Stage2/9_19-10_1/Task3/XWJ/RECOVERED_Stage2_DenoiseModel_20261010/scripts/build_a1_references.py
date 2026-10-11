"""Build A1 clean and auxiliary temporal references.

Usage:
  python scripts/build_a1_references.py [--out artifacts/finetune_mis20s1]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.reference import build_a1_references  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "finetune_mis20s1",
    )
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--fpn-gain", default="800")
    args = parser.parse_args()
    report = build_a1_references(
        dataset_root=args.dataset_root,
        out_dir=args.out,
        chunk_frames=args.chunk_frames,
        fpn_gain=args.fpn_gain,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

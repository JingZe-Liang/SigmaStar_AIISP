"""Build all A0 artifacts (stream manifests + motion masks) for groups 04/03.

Usage:
  python scripts/build_a0_manifests.py [--groups 04 03] [--out artifacts/finetune_mis20s1]
                                       [--max-frames N] [--chunk-frames N]
Long output redirects well to a tmp log; progress also lands in
<out>/progress.json (finetune_a0_progress_v1 contract).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.motion import MotionMaskParams  # noqa: E402
from raw_fusion.finetune.pipeline import build_a0_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "finetune_mis20s1")
    parser.add_argument("--groups", nargs="+", default=["04", "03"], choices=["04", "03"])
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--k-mad", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=None, help="cap frames per stream (smoke runs only)")
    args = parser.parse_args()

    summary = build_a0_artifacts(
        dataset_root=args.dataset_root,
        out_dir=args.out,
        groups=tuple(args.groups),
        chunk_frames=args.chunk_frames,
        mask_params=MotionMaskParams(k_mad=args.k_mad),
        max_frames=args.max_frames,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "labels"}, indent=2))
    print(f"[a0] {len(summary['labels'])} manifests written under {summary['manifest_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

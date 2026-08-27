from __future__ import annotations

import sys
from pathlib import Path


STAGE3_ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = STAGE3_ROOT.parent / "Stage2"
sys.path.insert(0, str(STAGE2_ROOT))

import make_four_view_stage2  # noqa: E402


def add_default(flag: str, value: str | Path) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def main() -> int:
    add_default("--inference-root", STAGE3_ROOT / "outputs" / "fold_128_to_645_temporal")
    add_default("--output-root", STAGE3_ROOT / "outputs" / "four_view_128_to_645_temporal")
    add_default("--stage-label", "Stage3")
    add_default("--output-filename", "stage3_motion_2dnr_3dnr_fusion.mp4")
    return make_four_view_stage2.main()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path


STAGE3_ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = STAGE3_ROOT.parent / "Stage2"
sys.path.insert(0, str(STAGE2_ROOT))

import infer_stage2  # noqa: E402
infer_stage2.STAGE_NAME = "Stage3"


def add_default(flag: str, value: Path) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def main() -> int:
    add_default("--checkpoint", STAGE3_ROOT / "runs" / "fold_128_to_645_temporal" / "phase1_best.pt")
    add_default("--output", STAGE3_ROOT / "outputs" / "fold_128_to_645_temporal")
    return infer_stage2.main()


if __name__ == "__main__":
    raise SystemExit(main())

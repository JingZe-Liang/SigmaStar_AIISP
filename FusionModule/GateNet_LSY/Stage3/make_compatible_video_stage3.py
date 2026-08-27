from __future__ import annotations

import sys
from pathlib import Path


STAGE3_ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = STAGE3_ROOT.parent / "Stage2"
sys.path.insert(0, str(STAGE2_ROOT))

import make_compatible_video  # noqa: E402


def main() -> int:
    if "--root" not in sys.argv:
        sys.argv.extend(["--root", str(STAGE3_ROOT / "outputs" / "fold_128_to_645_temporal")])
    return make_compatible_video.main()


if __name__ == "__main__":
    raise SystemExit(main())

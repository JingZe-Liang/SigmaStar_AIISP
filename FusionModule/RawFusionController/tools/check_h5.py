from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rawfusion.utils.h5_utils import inspect_h5


def main() -> None:
    p = argparse.ArgumentParser(description="Check RAW fusion H5 file format.")
    p.add_argument("--h5", required=True, help="Path to one h5 file, e.g. D:/Data/scene_1/shard_0.h5")
    p.add_argument("--out_json", default="", help="Optional output json path.")
    args = p.parse_args()
    info = inspect_h5(args.h5)
    text = json.dumps(info, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

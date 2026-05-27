from __future__ import annotations

import argparse
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Iterable, List

DEFAULT_TRAIN = ["scene_1", "scene_2", "scene_4", "scene_6", "scene_7"]
DEFAULT_VAL = ["scene_3", "scene_8"]
DEFAULT_TEST = ["scene_5", "scene_9"]


def collect(root: Path, scenes: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for s in scenes:
        scene_dir = root / s
        if not scene_dir.exists():
            print(f"[WARN] missing scene: {scene_dir}")
            continue
        files.extend(sorted(scene_dir.glob("shard_*.h5")))
    return files


def write_list(path: Path, files: List[Path], relative_to: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in files:
        if relative_to is not None:
            try:
                lines.append(str(f.resolve().relative_to(relative_to.resolve())).replace("\\", "/"))
            except ValueError:
                lines.append(str(f.resolve()))
        else:
            lines.append(str(f.resolve()))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {path} ({len(files)} h5 files)")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate train/val/test h5 lists from scene folders.")
    p.add_argument("--data_root", required=True, help="Folder containing scene_1 ... scene_9, e.g. D:/Data")
    p.add_argument("--out_dir", default="data_catalog")
    p.add_argument("--relative", action="store_true", help="Write paths relative to project root. Absolute is recommended on Windows.")
    p.add_argument("--train_scenes", default=",".join(DEFAULT_TRAIN))
    p.add_argument("--val_scenes", default=",".join(DEFAULT_VAL))
    p.add_argument("--test_scenes", default=",".join(DEFAULT_TEST))
    args = p.parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    rel_to = Path.cwd() if args.relative else None
    train = collect(root, [x.strip() for x in args.train_scenes.split(",") if x.strip()])
    val = collect(root, [x.strip() for x in args.val_scenes.split(",") if x.strip()])
    test = collect(root, [x.strip() for x in args.test_scenes.split(",") if x.strip()])
    write_list(out_dir / "train.txt", train, rel_to)
    write_list(out_dir / "val.txt", val, rel_to)
    write_list(out_dir / "test.txt", test, rel_to)
    write_list(out_dir / "train_tiny.txt", train[:2], rel_to)
    write_list(out_dir / "val_tiny.txt", val[:2], rel_to)
    write_list(out_dir / "test_tiny.txt", test[:2], rel_to)
    overlap = set(map(str, train)).intersection(map(str, val)) or set(map(str, train)).intersection(map(str, test)) or set(map(str, val)).intersection(map(str, test))
    if overlap:
        raise RuntimeError(f"Split overlap detected: {sorted(overlap)[:5]}")


if __name__ == "__main__":
    main()

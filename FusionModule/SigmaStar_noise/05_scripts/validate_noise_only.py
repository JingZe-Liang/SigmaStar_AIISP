from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = (
    HERE.parents[1]
    if HERE.parent.name == "05_scripts"
    else HERE.parent / "SigmaStar_noise_components_only"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_npz(path: Path) -> None:
    forbidden = {"source_frames_raw12", "fixed_mean_raw12", "temporal_residual_raw12"}
    with np.load(path, allow_pickle=False) as archive:
        overlap = forbidden.intersection(archive.files)
        if overlap:
            raise ValueError(f"preview fields found in {path}: {sorted(overlap)}")
        for key in archive.files:
            value = archive[key]
            if value.dtype == object:
                raise TypeError(f"object array is not allowed: {path}:{key}")
            _ = value.shape


def main() -> None:
    raw_files = list(ROOT.rglob("*.raw"))
    preview_paths = [path for path in ROOT.rglob("*") if "preview" in path.name.lower()]
    if raw_files or preview_paths:
        raise ValueError(f"unexpected raw/preview content: {raw_files + preview_paths}")

    black_static = sorted((ROOT / "02_black_noise").glob("black_*_noise.npz"))
    black_dynamic = sorted((ROOT / "02_black_noise").glob("black_*_dynamic_temporal.npz"))
    flat = sorted((ROOT / "03_flat_noise").glob("flat_*_noise.npz"))
    if (len(black_static), len(black_dynamic), len(flat)) != (9, 9, 9):
        raise ValueError(
            f"unexpected component counts: static={len(black_static)}, "
            f"dynamic={len(black_dynamic)}, flat={len(flat)}"
        )

    files = sorted(path for path in ROOT.rglob("*") if path.is_file())
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            validate_npz(path)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                list(csv.reader(stream))
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".md", ".txt", ".py"}:
            path.read_text(encoding="utf-8")

    validation = ROOT / "06_validation"
    result = {
        "black_static_component_files": len(black_static),
        "black_dynamic_component_files": len(black_dynamic),
        "flat_component_files": len(flat),
        "raw_files": 0,
        "preview_files": 0,
        "validated_files_before_manifest_refresh": len(files),
    }
    (validation / "VALIDATION_RESULT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    contents_path = validation / "PACKAGE_CONTENTS.txt"
    manifest_path = validation / "MANIFEST.sha256.txt"
    contents_path.touch(exist_ok=True)
    manifest_path.touch(exist_ok=True)
    files = sorted(path for path in ROOT.rglob("*") if path.is_file())
    contents_path.write_text(
        "\n".join(path.relative_to(ROOT).as_posix() for path in files) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
            for path in files
            if path != manifest_path
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Replace noisy/2dnr/3dnr keys in the complete H5 dataset.

Each file is copied to a temporary sibling, transformed there, checked, and
atomically replaced. The clean key and all non-target HDF5 content are kept.
Run with --apply to perform the replacement; without it the script only
prints the discovered dataset manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_company_style_from_clean import (  # noqa: E402
    CompanyStyle3DNR,
    NoiseSynthesizer,
    StrongBayer2DNR,
)


EXPECTED_KEYS = {"clean", "noisy", "2dnr", "3dnr"}


def scene_paths(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.glob("scene_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )


def shard_paths(scene: Path) -> list[Path]:
    return sorted(scene.glob("shard_*.h5"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def sha256_dataset(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    if dataset.shape == ():
        digest.update(np.asarray(dataset[()]).tobytes())
        return digest.hexdigest()
    for index in range(dataset.shape[0]):
        digest.update(np.ascontiguousarray(dataset[index]).tobytes())
    return digest.hexdigest()


def inspect(root: Path) -> dict[str, Any]:
    scenes = scene_paths(root)
    manifest: dict[str, Any] = {"root": str(root), "scenes": [], "total_files": 0, "total_frames": 0}
    for scene in scenes:
        shards = shard_paths(scene)
        scene_frames = 0
        for shard in shards:
            with h5py.File(shard, "r") as handle:
                keys = set(handle.keys())
                required = {
                    "clean": ((30, 1080, 1920), "uint16"),
                    "noisy": ((30, 2, 1080, 1920), "uint16"),
                    "2dnr": ((30, 1080, 1920), "uint16"),
                    "3dnr": ((30, 1080, 1920), "uint16"),
                }
                if keys != EXPECTED_KEYS:
                    raise ValueError(f"{shard}: expected keys {EXPECTED_KEYS}, got {keys}")
                for key, (shape, dtype) in required.items():
                    if tuple(handle[key].shape) != shape or str(handle[key].dtype) != dtype:
                        raise ValueError(f"{shard}:{key}: expected {shape}/{dtype}, got {handle[key].shape}/{handle[key].dtype}")
                scene_frames += int(handle["clean"].shape[0])
        manifest["scenes"].append({"scene": scene.name, "shards": len(shards), "frames": scene_frames})
        manifest["total_files"] += len(shards)
        manifest["total_frames"] += scene_frames
    return manifest


def process_shard(
    source: Path,
    noise: NoiseSynthesizer,
    two_dnr: StrongBayer2DNR,
    three_dnr: CompanyStyle3DNR,
    previous_noisy: np.ndarray | None,
    dry_run: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if dry_run:
        return {"path": str(source), "status": "planned"}, previous_noisy

    temp_name: str | None = None
    before_clean_hash: str
    try:
        with h5py.File(source, "r") as original:
            before_clean_hash = sha256_dataset(original["clean"])
        fd, temp_name = tempfile.mkstemp(prefix=f".{source.stem}.", suffix=".tmp.h5", dir=str(source.parent))
        os.close(fd)
        shutil.copy2(source, temp_name)

        with h5py.File(temp_name, "r+") as handle:
            clean_dataset = handle["clean"]
            noisy_dataset = handle["noisy"]
            two_dataset = handle["2dnr"]
            three_dataset = handle["3dnr"]
            frame_count = int(clean_dataset.shape[0])
            for index in range(frame_count):
                clean = clean_dataset[index].astype(np.float32)
                current_noisy = noise(clean)
                current_2dnr = two_dnr(current_noisy)
                current_3dnr, _ = three_dnr(current_noisy, current_2dnr)
                slot0 = current_noisy if previous_noisy is None else previous_noisy
                noisy_dataset[index, 0] = np.clip(np.rint(slot0), 0, 4095).astype(np.uint16)
                noisy_dataset[index, 1] = np.clip(np.rint(current_noisy), 0, 4095).astype(np.uint16)
                two_dataset[index] = np.clip(np.rint(current_2dnr), 0, 4095).astype(np.uint16)
                three_dataset[index] = np.clip(np.rint(current_3dnr), 0, 4095).astype(np.uint16)
                previous_noisy = current_noisy.copy()

            handle.attrs["3dnr_generated_from"] = "clean -> synthesized noisy -> 2dnr"
            handle.attrs["3dnr_hqdn3d_filter"] = "company_style_coarse_motion_gate"
            handle.attrs["company_style_pipeline"] = "clean_based_company_style_3dnr"
            handle.attrs["company_style_exposure_compensation"] = 5.0
            handle.attrs["company_style_noise_scale"] = 0.60
            handle.attrs["company_style_static_2dnr_weight"] = 0.80
            handle.attrs["company_style_motion_noise_gain"] = 1.35
            noisy_dataset.attrs["generated_from"] = "clean"
            two_dataset.attrs["generated_from"] = "synthesized noisy"
            three_dataset.attrs["generated_from"] = "synthesized noisy + 2dnr + frame-difference gate"
            three_dataset.attrs["hqdn3d_filter"] = "company_style_coarse_motion_gate"
            handle.flush()

        with h5py.File(source, "r") as original, h5py.File(temp_name, "r") as updated:
            after_clean_hash = sha256_dataset(updated["clean"])
            if before_clean_hash != after_clean_hash:
                raise RuntimeError(f"clean dataset changed in {source}")
            for key in ("noisy", "2dnr", "3dnr"):
                if tuple(updated[key].shape) != tuple(original[key].shape):
                    raise RuntimeError(f"shape changed for {source}:{key}")
        os.replace(temp_name, source)
        temp_name = None
        return {"path": str(source), "status": "replaced", "clean_sha256": before_clean_hash}, previous_noisy
    finally:
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", type=Path, default=Path("data/H5"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="perform atomic H5 replacement")
    parser.add_argument("--repair-boundary-slots", action="store_true", help="repair noisy[0,0] at shard boundaries from the previous shard current frame")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    manifest = inspect(args.h5_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.repair_boundary_slots:
        repaired = 0
        for scene in scene_paths(args.h5_root):
            shards = shard_paths(scene)
            for previous_path, current_path in zip(shards, shards[1:]):
                with h5py.File(previous_path, "r") as previous, h5py.File(current_path, "r+") as current:
                    current["noisy"][0, 0] = previous["noisy"][-1, 1]
                    current.flush()
                    repaired += 1
        print(json.dumps({"repaired_boundary_slots": repaired}, ensure_ascii=False, indent=2))
        return
    if not args.apply:
        print("未执行覆盖操作；如确认替换，请增加 --apply。")
        return

    records: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scene_paths(args.h5_root)):
        # One stream per scene preserves causal state across shard boundaries.
        scene_seed = args.seed + scene_index * 100003
        noise = NoiseSynthesizer(16.0, 4095.0, 4.0, 120.0, 0.60, scene_seed)
        two_dnr = StrongBayer2DNR(9, 7.0, 110.0)
        three_dnr = CompanyStyle3DNR(45.0, 8, 3, 0.80, 1.35)
        previous_noisy: np.ndarray | None = None
        for shard in shard_paths(scene):
            record, previous_noisy = process_shard(shard, noise, two_dnr, three_dnr, previous_noisy, dry_run=False)
            records.append(record)

    report = {
        "pipeline": "clean_based_company_style_3dnr",
        "h5_root": str(args.h5_root),
        "manifest": manifest,
        "parameters": {
            "seed_base": args.seed,
            "black_level": 16.0,
            "white_level": 4095.0,
            "shot_k": 4.0,
            "read_sigma": 120.0,
            "noise_scale": 0.60,
            "2dnr": {"diameter": 9, "sigma_space": 7.0, "sigma_range": 110.0},
            "3dnr": {"motion_threshold": 45.0, "motion_block": 8, "motion_dilation": 3, "static_2dnr_weight": 0.80, "motion_noise_gain": 1.35},
            "isp_exposure_compensation": 5.0,
        },
        "records": records,
    }
    report_path = args.manifest or (args.h5_root / "clean_based_company_style_3dnr_replacement_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"replaced_files": len(records), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


H, W = 1080, 1920
PLANES = ((0, 0), (0, 1), (1, 0), (1, 1))


def default_workspace() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] if here.parent.name == "05_scripts" else here.parent


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only noise components for one black RAW frame."
    )
    parser.add_argument("--workspace", type=Path, default=default_workspace())
    parser.add_argument("--gain", required=True, help="gain label, e.g. 100 or 25600")
    parser.add_argument("--frame", type=int, required=True, help="zero-based frame index")
    parser.add_argument("--out", type=Path, required=True, help="output NPZ path")
    return parser.parse_args()


def find_source(workspace: Path, gain: str) -> tuple[Path, int]:
    inventory = workspace / "sigmastar_noise_results" / "file_inventory.csv"
    with inventory.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] == "black" and row["gain_label"] == gain:
                return Path(row["path"]), int(row["frames"])
    raise FileNotFoundError(f"black gain {gain!r} not found in {inventory}")


def put_phase(target: np.ndarray, phase_values: np.ndarray) -> None:
    for index, (r, c) in enumerate(PLANES):
        target[r::2, c::2] = phase_values[index]


def main() -> None:
    options = args()
    workspace = options.workspace.resolve()
    source, frame_count = find_source(workspace, options.gain)
    if options.frame < 0 or options.frame >= frame_count:
        raise IndexError(f"frame must be in [0, {frame_count - 1}]")

    component_dir = workspace / "SigmaStar_noise_components_only" / "02_black_noise"
    static_path = component_dir / f"black_{options.gain}_noise.npz"
    dynamic_path = component_dir / f"black_{options.gain}_dynamic_temporal.npz"
    with np.load(static_path, allow_pickle=False) as static, np.load(
        dynamic_path, allow_pickle=False
    ) as dynamic:
        fixed = np.zeros((H, W), dtype=np.float32)
        put_phase(
            fixed,
            np.asarray(
                [
                    static["black_level_raw12"][i]
                    + static["row_fpn_raw12"][i][:, None]
                    + static["col_fpn_raw12"][i][None, :]
                    + static["pixel_fpn_raw12"][i]
                    for i in range(4)
                ],
                dtype=np.float32,
            ),
        )
        common = dynamic["frame_common_offset_raw12"][options.frame]
        row = dynamic["row_dynamic_raw12"][options.frame]
        col = dynamic["col_dynamic_raw12"][options.frame]
        common_map = np.zeros((H, W), dtype=np.float32)
        row_map = np.zeros((H, W), dtype=np.float32)
        col_map = np.zeros((H, W), dtype=np.float32)
        put_phase(common_map, np.stack([np.full((H // 2, W // 2), common[i]) for i in range(4)]))
        put_phase(row_map, np.stack([np.broadcast_to(row[i][:, None], (H // 2, W // 2)) for i in range(4)]))
        put_phase(col_map, np.stack([np.broadcast_to(col[i][None, :], (H // 2, W // 2)) for i in range(4)]))

    raw = np.memmap(source, dtype="<u2", mode="r", shape=(frame_count, H, W))
    frame = np.right_shift(raw[options.frame], 4).astype(np.float32)
    total_temporal = frame - fixed
    unstructured = total_temporal - common_map - row_map - col_map
    options.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        options.out,
        fixed_baseline_raw12=fixed,
        frame_common_raw12=common_map,
        dynamic_row_raw12=row_map,
        dynamic_col_raw12=col_map,
        unstructured_temporal_noise_raw12=unstructured,
    )
    print(options.out)


if __name__ == "__main__":
    main()

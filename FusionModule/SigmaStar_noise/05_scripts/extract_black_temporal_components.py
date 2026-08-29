from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "sigmastar_noise_results"
OUT = BASE / "SigmaStar_noise_components_only"
PLANES = ((0, 0), (0, 1), (1, 0), (1, 1))
H, W = 1080, 1920


def inventory() -> list[dict[str, str]]:
    with (RESULTS / "file_inventory.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = [row for row in csv.DictReader(stream) if row["kind"] == "black"]
    return sorted(rows, key=lambda row: float(row["gain_label"]))


def decompose(row: dict[str, str]) -> list[dict[str, float | int | str]]:
    gain = row["gain_label"]
    frames = int(row["frames"])
    source = Path(row["path"])
    with np.load(
        RESULTS / "maps" / f"black_{gain}_components.npz", allow_pickle=False
    ) as archive:
        mean = archive["mean_raw12"].astype(np.float32)
        hot = np.stack(
            [archive[f"hot_mask_p{plane}"].astype(bool) for plane in range(4)]
        )

    raw = np.memmap(source, dtype="<u2", mode="r", shape=(frames, H, W))
    common = np.empty((frames, 4), dtype=np.float32)
    rows = np.empty((frames, 4, H // 2), dtype=np.float32)
    cols = np.empty((frames, 4, W // 2), dtype=np.float32)
    residual_sum = np.zeros((4, H // 2, W // 2), dtype=np.float64)
    residual_sumsq = np.zeros_like(residual_sum)
    residual_count = np.zeros_like(residual_sum, dtype=np.uint16)

    for frame_index in range(frames):
        frame = np.right_shift(raw[frame_index], 4).astype(np.float32)
        residual = frame - mean
        for plane, (r, c) in enumerate(PLANES):
            values = residual[r::2, c::2]
            global_offset = float(np.median(values))
            row_offset = np.median(values - global_offset, axis=1)
            row_offset -= np.median(row_offset)
            after_row = values - global_offset - row_offset[:, None]
            col_offset = np.median(after_row, axis=0)
            col_offset -= np.median(col_offset)
            white = values - global_offset - row_offset[:, None] - col_offset[None, :]

            common[frame_index, plane] = global_offset
            rows[frame_index, plane] = row_offset
            cols[frame_index, plane] = col_offset
            valid = ~hot[plane] & np.isfinite(white)
            safe = np.where(valid, white, 0.0).astype(np.float64)
            residual_sum[plane] += safe
            residual_sumsq[plane] += safe * safe
            residual_count[plane] += valid.astype(np.uint16)

        if (frame_index + 1) % 10 == 0 or frame_index + 1 == frames:
            print(f"black {gain}: {frame_index + 1}/{frames}", flush=True)

    count = residual_count.astype(np.float64)
    numerator = residual_sumsq - np.divide(
        residual_sum * residual_sum,
        count,
        out=np.zeros_like(residual_sum),
        where=count > 0,
    )
    variance = np.divide(
        numerator,
        count - 1.0,
        out=np.full_like(numerator, np.nan),
        where=count > 1,
    )
    white_std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)

    target = OUT / "02_black_noise" / f"black_{gain}_dynamic_temporal.npz"
    np.savez_compressed(
        target,
        frame_common_offset_raw12=common,
        row_dynamic_raw12=rows,
        col_dynamic_raw12=cols,
        unstructured_temporal_std_raw12=white_std,
        valid_sample_count=residual_count,
    )

    summary: list[dict[str, float | int | str]] = []
    for plane in range(4):
        valid_std = white_std[plane][~hot[plane]]
        summary.append(
            {
                "gain_label": gain,
                "gain_x": float(gain) / 100.0,
                "plane": plane,
                "frames": frames,
                "common_dynamic_rms_raw12": float(np.sqrt(np.mean(common[:, plane] ** 2))),
                "row_dynamic_rms_raw12": float(np.sqrt(np.mean(rows[:, plane] ** 2))),
                "col_dynamic_rms_raw12": float(np.sqrt(np.mean(cols[:, plane] ** 2))),
                "unstructured_temporal_rms_raw12": float(np.nanmedian(valid_std)),
            }
        )
    return summary


def main() -> None:
    all_rows: list[dict[str, float | int | str]] = []
    for row in inventory():
        all_rows.extend(decompose(row))
    target = OUT / "04_models" / "black_dynamic_noise_summary.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()

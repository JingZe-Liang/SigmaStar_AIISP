from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import h5py
import torch
from tqdm import tqdm

from rawfusion.data.dataset import read_path_list
from rawfusion.losses.metrics import SSEMetric, summarize_metrics


def iter_frames(files, data_max_value: float):
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = int(f["clean"].shape[0])
            for i in range(n):
                d2 = torch.from_numpy(f["2dnr"][i].astype("float32") / float(data_max_value))[None, None]
                d3 = torch.from_numpy(f["3dnr"][i].astype("float32") / float(data_max_value))[None, None]
                c = torch.from_numpy(f["clean"][i].astype("float32") / float(data_max_value))[None, None]
                yield d2, d3, c


def main():
    p = argparse.ArgumentParser(description="Evaluate raw 2DNR/3DNR/avg/oracle baselines on a list.")
    p.add_argument("--list", required=True, help="val/test txt list")
    p.add_argument("--data_max_value", type=float, default=4095.0)
    p.add_argument("--psnr_max_value", type=float, default=4095.0)
    p.add_argument("--out_json", default="outputs/metrics/baseline_eval.json")
    args = p.parse_args()
    files = read_path_list(args.list)
    metrics = {"2dnr": SSEMetric(), "3dnr": SSEMetric(), "avg_50_50": SSEMetric(), "oracle_pixelwise": SSEMetric()}
    total = 0
    for d2, d3, c in tqdm(iter_frames(files, args.data_max_value), desc="baseline"):
        metrics["2dnr"].update(d2, c)
        metrics["3dnr"].update(d3, c)
        metrics["avg_50_50"].update(0.5 * d2 + 0.5 * d3, c)
        oracle = torch.where((d3 - c).abs() <= (d2 - c).abs(), d3, d2)
        metrics["oracle_pixelwise"].update(oracle, c)
        total += 1
    out = {"list": str(Path(args.list).resolve()), "files": len(files), "frames": total}
    out.update(summarize_metrics(metrics, max_i_code=args.psnr_max_value, storage_max=args.data_max_value))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

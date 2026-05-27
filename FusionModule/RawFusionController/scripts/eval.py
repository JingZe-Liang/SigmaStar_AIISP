from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import yaml
from torch.utils.data import DataLoader

from rawfusion.data.dataset import H5FusionDataset, read_path_list
from rawfusion.engine import evaluate
from rawfusion.models.fusion_net import MotionAwareFusionUNet, count_parameters


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained fusion controller.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--out_json", default="outputs/metrics/eval.json")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patch_size", type=int, default=0, help="0 means full-frame eval")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", default="")
    args = p.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    data_max = float(cfg.get("data_max_value", 4095.0))
    psnr_max = float(cfg.get("psnr_max_value", data_max))
    feature_mode = cfg.get("feature_mode", "strong")
    files = read_path_list(args.list)
    ds = H5FusionDataset(files, data_max_value=data_max, patch_size=args.patch_size, strict_range_check=not bool(cfg.get("allow_value_clip", False)), feature_mode=feature_mode, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    in_ch = int(ckpt.get("in_ch", ds[0]["x"].shape[0]))
    model = MotionAwareFusionUNet(in_ch=in_ch, base=int(cfg.get("model_base", 24)), groups=int(cfg.get("gn_groups", 8)), init_alpha3d=float(cfg.get("init_alpha3d", 0.80))).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    loss_cfg = {
        "lam_grad": float(cfg.get("lam_grad", 0.15)),
        "lam_tv": float(cfg.get("lam_tv", 0.002)),
        "lam_motion": float(cfg.get("lam_motion", 0.01)),
        "lam_oracle": float(cfg.get("lam_oracle", 0.03)),
        "oracle_tau": float(cfg.get("oracle_tau", 0.015)),
    }
    print(f"device={device}, samples={len(ds)}, params={count_parameters(model):,}")
    out = evaluate(model, loader, device, loss_cfg, psnr_max, data_max, compute_ssim=True, eval_baselines=True)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

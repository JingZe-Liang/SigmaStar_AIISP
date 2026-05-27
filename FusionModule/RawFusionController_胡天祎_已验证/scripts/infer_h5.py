from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from rawfusion.data.dataset import _normalize_uint16, build_fusion_features
from rawfusion.models.fusion_net import MotionAwareFusionUNet, fuse_alpha3d
from rawfusion.utils.visualize import make_compare_grid, save_alpha_png, save_pgm_u16, save_png_gray, to_u16_code


def infer_tile(model, x, dnr2, dnr3, tile: int, overlap: int):
    if tile <= 0:
        alpha = model(x)
        return alpha, fuse_alpha3d(alpha, dnr2, dnr3)
    _, _, h, w = x.shape
    tile = min(tile, h, w)
    stride = max(tile - overlap, 1)
    alpha_sum = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)
    fused_sum = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    for y in ys:
        for xx in xs:
            sl = (slice(y, y + tile), slice(xx, xx + tile))
            a = model(x[:, :, sl[0], sl[1]])
            f = fuse_alpha3d(a, dnr2[:, :, sl[0], sl[1]], dnr3[:, :, sl[0], sl[1]])
            alpha_sum[:, :, sl[0], sl[1]] += a.float()
            fused_sum[:, :, sl[0], sl[1]] += f.float()
            weight_sum[:, :, sl[0], sl[1]] += 1.0
    return alpha_sum / weight_sum.clamp_min(1e-6), fused_sum / weight_sum.clamp_min(1e-6)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer one H5 shard and export fused RAW / alpha map / comparison PNGs.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out_dir", default="outputs/infer")
    p.add_argument("--frames", default="0,1", help="Comma separated frame indices, or 'all'.")
    p.add_argument("--tile", type=int, default=768, help="Tile size. 0 means full-frame.")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--save_npy", action="store_true")
    p.add_argument("--save_pgm", action="store_true")
    p.add_argument("--save_png", action="store_true", default=True)
    p.add_argument("--auto_stretch", action="store_true", default=True)
    p.add_argument("--device", default="")
    args = p.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    data_max = float(cfg.get("data_max_value", 4095.0))
    feature_mode = cfg.get("feature_mode", "strong")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    in_ch = int(ckpt.get("in_ch", 7))
    model = MotionAwareFusionUNet(in_ch=in_ch, base=int(cfg.get("model_base", 24)), groups=int(cfg.get("gn_groups", 8)), init_alpha3d=float(cfg.get("init_alpha3d", 0.80))).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as f:
        n = int(f["clean"].shape[0]) if "clean" in f else int(f["noisy"].shape[0])
        if args.frames.strip().lower() == "all":
            frame_ids = list(range(n))
        else:
            frame_ids = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        manifest = []
        for i in tqdm(frame_ids, desc="infer"):
            noisy = _normalize_uint16(f["noisy"][i], data_max, strict_range_check=False)
            dnr2 = _normalize_uint16(f["2dnr"][i], data_max, strict_range_check=False)[None, ...]
            dnr3 = _normalize_uint16(f["3dnr"][i], data_max, strict_range_check=False)[None, ...]
            clean = _normalize_uint16(f["clean"][i], data_max, strict_range_check=False)[None, ...] if "clean" in f else None
            x_np = build_fusion_features(noisy, dnr2, dnr3, feature_mode=feature_mode)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            d2 = torch.from_numpy(dnr2).unsqueeze(0).to(device)
            d3 = torch.from_numpy(dnr3).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                alpha, fused = infer_tile(model, x, d2, d3, tile=args.tile, overlap=args.overlap)
            alpha_np = alpha[0, 0].detach().cpu().numpy().astype(np.float32)
            fused_np = fused[0, 0].detach().cpu().numpy().astype(np.float32)
            stem = f"frame_{i:04d}"
            fused_code = to_u16_code(fused_np, data_max)
            if args.save_npy:
                np.save(out_dir / f"{stem}_fused.npy", fused_code)
                np.save(out_dir / f"{stem}_alpha3d.npy", alpha_np)
            if args.save_pgm:
                save_pgm_u16(out_dir / f"{stem}_fused.pgm", fused_code)
            save_png_gray(out_dir / f"{stem}_fused.png", fused_code, code_max=data_max, auto_stretch=args.auto_stretch)
            save_alpha_png(out_dir / f"{stem}_alpha3d.png", alpha_np)
            grid_imgs = {
                "noisy_curr": noisy[1] * data_max,
                "2DNR": dnr2[0] * data_max,
                "3DNR": dnr3[0] * data_max,
                "AI_fused": fused_np * data_max,
                "alpha3d": alpha_np,
            }
            if clean is not None:
                grid_imgs["clean"] = clean[0] * data_max
            make_compare_grid(grid_imgs, out_dir / f"{stem}_compare.png", code_max=data_max, auto_stretch=args.auto_stretch)
            manifest.append({"frame": i, "fused_png": f"{stem}_fused.png", "alpha_png": f"{stem}_alpha3d.png", "compare_png": f"{stem}_compare.png", "alpha_mean": float(alpha_np.mean()), "alpha_var": float(alpha_np.var())})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved outputs -> {out_dir}")


if __name__ == "__main__":
    main()

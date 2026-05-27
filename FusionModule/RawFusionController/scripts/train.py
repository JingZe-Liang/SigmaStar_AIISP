from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from rawfusion.data.dataset import H5FusionDataset, read_path_list
from rawfusion.engine import evaluate, train_one_epoch
from rawfusion.models.fusion_net import MotionAwareFusionUNet, count_parameters
from rawfusion.utils.seed import set_seed


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="Train RAW 2DNR/3DNR AI fusion controller.")
    p.add_argument("--config", default="configs/train_tiny.yaml")
    p.add_argument("--train_list", default=None)
    p.add_argument("--val_list", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    for k in ["train_list", "val_list", "save_dir", "device"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    for k in ["epochs", "batch_size", "patch_size"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    set_seed(int(cfg.get("seed", 42)))
    train_files = read_path_list(cfg["train_list"])
    val_files = read_path_list(cfg["val_list"])
    data_max = float(cfg.get("data_max_value", 4095.0))
    psnr_max = float(cfg.get("psnr_max_value", data_max))
    feature_mode = cfg.get("feature_mode", "strong")
    train_ds = H5FusionDataset(
        train_files,
        data_max_value=data_max,
        patch_size=int(cfg.get("patch_size", 384)),
        strict_range_check=not bool(cfg.get("allow_value_clip", False)),
        feature_mode=feature_mode,
        max_samples=int(cfg.get("max_train_samples", 0)),
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = H5FusionDataset(
        val_files,
        data_max_value=data_max,
        patch_size=int(cfg.get("val_patch_size", 0)),
        strict_range_check=not bool(cfg.get("allow_value_clip", False)),
        feature_mode=feature_mode,
        max_samples=int(cfg.get("max_val_samples", 0)),
        seed=int(cfg.get("seed", 42)) + 1,
    )
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 1)), shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("val_batch_size", 1)), shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    in_ch = int(train_ds[0]["x"].shape[0])
    model = MotionAwareFusionUNet(in_ch=in_ch, base=int(cfg.get("model_base", 24)), groups=int(cfg.get("gn_groups", 8)), init_alpha3d=float(cfg.get("init_alpha3d", 0.80))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg.get("epochs", 10)), 1), eta_min=float(cfg.get("min_lr", 2e-6)))
    amp = bool(cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    loss_cfg = {
        "lam_grad": float(cfg.get("lam_grad", 0.15)),
        "lam_tv": float(cfg.get("lam_tv", 0.002)),
        "lam_motion": float(cfg.get("lam_motion", 0.01)),
        "lam_oracle": float(cfg.get("lam_oracle", 0.03)),
        "oracle_tau": float(cfg.get("oracle_tau", 0.015)),
    }
    save_dir = Path(cfg.get("save_dir", "checkpoints/exp"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "log.csv"
    best_path = save_dir / "best.pt"
    (save_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"device={device}, in_ch={in_ch}, params={count_parameters(model):,}, train_samples={len(train_ds)}, val_samples={len(val_ds)}")
    print(f"save_dir={save_dir}")
    best_psnr = -1.0
    fields = None
    for epoch in range(1, int(cfg.get("epochs", 10)) + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler, amp, loss_cfg, psnr_max, data_max, grad_clip=float(cfg.get("grad_clip", 1.0)))
        va = evaluate(model, val_loader, device, loss_cfg, psnr_max, data_max, compute_ssim=bool(cfg.get("compute_ssim", True)), eval_baselines=True)
        scheduler.step()
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        if fields is None:
            fields = list(row.keys())
            with log_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(
            f"epoch={epoch:03d} train_psnr={tr['psnr']:.3f} val_ai={va['ai_psnr']:.3f} "
            f"val_2dnr={va['2dnr_psnr']:.3f} val_3dnr={va['3dnr_psnr']:.3f} "
            f"val_avg={va['avg_50_50_psnr']:.3f} val_oracle={va['oracle_pixelwise_psnr']:.3f} "
            f"alpha_mean={va['alpha_mean']:.3f} alpha_var={va['alpha_var']:.5f}"
        )
        torch.save({"model": model.state_dict(), "config": cfg, "in_ch": in_ch, "epoch": epoch, "val_ai_psnr": va["ai_psnr"]}, save_dir / "last.pt")
        if va["ai_psnr"] > best_psnr:
            best_psnr = va["ai_psnr"]
            torch.save({"model": model.state_dict(), "config": cfg, "in_ch": in_ch, "epoch": epoch, "val_ai_psnr": va["ai_psnr"]}, best_path)
            print(f"saved best -> {best_path} ({best_psnr:.3f} dB)")
    (save_dir / "summary.json").write_text(json.dumps({"best_val_ai_psnr": best_psnr, "best_path": str(best_path)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

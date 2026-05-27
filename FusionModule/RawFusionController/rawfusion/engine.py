from __future__ import annotations

from typing import Dict, Optional

import torch
from tqdm import tqdm

from rawfusion.losses.fusion_losses import fusion_loss
from rawfusion.losses.metrics import SSEMetric, ssim, summarize_metrics
from rawfusion.models.fusion_net import fuse_alpha3d


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp: bool,
    loss_cfg: Dict[str, float],
    max_i_code: float,
    storage_max: float,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {"loss": 0, "rec": 0, "grad": 0, "tv": 0, "motion": 0, "oracle": 0, "alpha_mean": 0, "alpha_var": 0}
    count = 0
    sse = SSEMetric()
    iterator = tqdm(loader, desc="train", leave=False)
    for batch in iterator:
        batch = to_device(batch, device)
        x = batch["x"]
        dnr2 = batch["dnr2"]
        dnr3 = batch["dnr3"]
        clean = batch["clean"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            alpha = model(x)
            fused = fuse_alpha3d(alpha, dnr2, dnr3)
            loss, m = fusion_loss(fused, clean, alpha, dnr2, dnr3, x, **loss_cfg)
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bsz = int(x.shape[0])
        count += bsz
        for k in ["loss", "rec", "grad", "tv", "motion", "oracle"]:
            sums[k] += m[k] * bsz
        sums["alpha_mean"] += float(alpha.detach().mean().cpu().item()) * bsz
        sums["alpha_var"] += float(alpha.detach().var(unbiased=False).cpu().item()) * bsz
        sse.update(fused.detach(), clean.detach())
        iterator.set_postfix(loss=f"{m['loss']:.4f}", psnr=f"{sse.psnr(max_i_code, storage_max):.2f}")
    out = {k: v / max(count, 1) for k, v in sums.items()}
    out["psnr"] = sse.psnr(max_i_code, storage_max)
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    loss_cfg: Dict[str, float],
    max_i_code: float,
    storage_max: float,
    compute_ssim: bool = True,
    eval_baselines: bool = True,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {"loss": 0, "rec": 0, "grad": 0, "tv": 0, "motion": 0, "oracle": 0, "alpha_mean": 0, "alpha_var": 0, "ssim": 0}
    count = 0
    metrics = {
        "ai": SSEMetric(),
        "2dnr": SSEMetric(),
        "3dnr": SSEMetric(),
        "avg_50_50": SSEMetric(),
        "oracle_pixelwise": SSEMetric(),
    }
    for batch in tqdm(loader, desc="val", leave=False):
        batch = to_device(batch, device)
        x = batch["x"]
        dnr2 = batch["dnr2"]
        dnr3 = batch["dnr3"]
        clean = batch["clean"]
        alpha = model(x)
        fused = fuse_alpha3d(alpha, dnr2, dnr3)
        _, m = fusion_loss(fused, clean, alpha, dnr2, dnr3, x, **loss_cfg)
        bsz = int(x.shape[0])
        count += bsz
        for k in ["loss", "rec", "grad", "tv", "motion", "oracle"]:
            sums[k] += m[k] * bsz
        sums["alpha_mean"] += float(alpha.mean().cpu().item()) * bsz
        sums["alpha_var"] += float(alpha.var(unbiased=False).cpu().item()) * bsz
        if compute_ssim:
            sums["ssim"] += ssim(fused, clean, data_range=max_i_code / storage_max) * bsz
        metrics["ai"].update(fused, clean)
        if eval_baselines:
            metrics["2dnr"].update(dnr2, clean)
            metrics["3dnr"].update(dnr3, clean)
            metrics["avg_50_50"].update(0.5 * dnr2 + 0.5 * dnr3, clean)
            e2 = (dnr2 - clean).abs()
            e3 = (dnr3 - clean).abs()
            oracle = torch.where(e3 <= e2, dnr3, dnr2)
            metrics["oracle_pixelwise"].update(oracle, clean)
    out = {k: v / max(count, 1) for k, v in sums.items()}
    out.update(summarize_metrics(metrics, max_i_code=max_i_code, storage_max=storage_max))
    out["psnr"] = out["ai_psnr"]
    return out

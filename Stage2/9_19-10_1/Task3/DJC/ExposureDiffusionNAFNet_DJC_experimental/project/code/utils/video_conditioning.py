"""Measured camera conditions and clean-target supervision for video diffusion."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

CAMERA_FEATURES = 6


def load_video_model(checkpoint, device, weights="auto"):
    """One inference loader: EMA when saved, strict architecture/weights match."""
    from models.video_diffusion import CleanVideoDiffusion
    cfg = checkpoint["config"]["model"]
    model = CleanVideoDiffusion(widths=tuple(cfg["widths"]), blocks=cfg["blocks_per_level"],
        time_dim=cfg["time_dim"], heads=cfg["attention_heads"],
        camera_dim=CAMERA_FEATURES if cfg.get("camera_conditioning", False) else 0,
        reconstruction_guidance=bool(cfg.get("reconstruction_guidance", False))).to(device).eval()
    if weights not in ("auto", "model", "ema_model"):
        raise ValueError("weights must be auto/model/ema_model")
    key = ("ema_model" if "ema_model" in checkpoint else "model") if weights == "auto" else weights
    model.load_state_dict(checkpoint[key])
    return model, key


@torch.no_grad()
def camera_features(mapping, bank, names, device):
    """Analog/ISP gain, exposure, black DN, and normalized std at 0/64 DN.

    Statistics use accepted observation models; this does not fit new noise or
    equate gain with exposure. Features and DN normalizations are fixed at inference.
    """
    cache = getattr(bank, "_video_condition_cache", {})
    for name in dict.fromkeys(names):
        if name in cache:
            continue
        p = mapping.profiles[name]
        physical = (p.analog_gain_factor, p.isp_gain_factor, p.exposure_us)
        if any(value is None or not math.isfinite(value) or value <= 0 for value in physical):
            raise ValueError(f"{name}: camera conditioning needs measured gain and exposure")
        scalar = torch.tensor([
            math.log2(physical[0]) / 11., math.log2(physical[1]) / 11.,
            math.log2(physical[2]) / 20., p.black_level_dn / bank.white_level,
        ], device=device)
        probes = [torch.full((1, 4, 1, 1), level, device=device) for level in (0., 64.)]
        # Existing noise_std API returns a normalized scalar, already /4095.
        stds = [bank.noise_std(probe, (name,)).reshape(1) for probe in probes]
        cache[name] = torch.cat([scalar, *stds])
    bank._video_condition_cache = cache
    return torch.stack([cache[name] for name in names]).to(device)


def clean_reconstruction_losses(prediction, target, valid):
    """Supervise actual clean colors; temporal loss compares ERRORS, not frames."""
    denominator = valid.sum().clamp_min(1.)
    reconstruction = ((prediction - target).abs() * valid).sum() / denominator
    # Block means preserve low-frequency color/brightness; masked pixels do not
    # dilute the reference, and motion is present in the target on both sides.
    error = (prediction - target) * valid
    flat_error, flat_valid = error.flatten(0, 1), valid.flatten(0, 1)
    size = min(16, *target.shape[-2:])
    means = F.avg_pool2d(flat_error, size)
    coverage = F.avg_pool2d(flat_valid, size)
    color = means.abs().sum() / coverage.sum().clamp_min(1.)
    pair_valid = valid[:, 1:] * valid[:, :-1]
    temporal = (((prediction - target)[:, 1:] - (prediction - target)[:, :-1]).square() * pair_valid).sum() / pair_valid.sum().clamp_min(1.)
    return reconstruction, color, temporal

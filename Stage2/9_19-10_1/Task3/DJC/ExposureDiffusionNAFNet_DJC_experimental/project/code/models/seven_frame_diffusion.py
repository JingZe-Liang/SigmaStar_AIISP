"""Joint seven-frame epsilon prediction: trainable NAFNet + existing pyramid."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .naf_diffusion import NAFDiffusion
from .schedule import linear_schedule, predict_x0_from_epsilon
from .temporal_refiner import PyramidTemporalRefiner


class SevenFrameNAFDiffusion(nn.Module):
    def __init__(self, spatial_widths=(64, 128, 256, 384, 512),
                 temporal_widths=(64, 128, 256, 384), temporal_blocks=(2, 3, 4, 6),
                 blocks_per_level=2, time_dim=64, bottleneck_attention_blocks=2,
                 attention_heads=8, max_flow_px=32., gradient_checkpointing=True):
        super().__init__()
        self.spatial = NAFDiffusion(widths=tuple(spatial_widths),
            blocks_per_level=blocks_per_level, time_dim=time_dim,
            bottleneck_attention_blocks=bottleneck_attention_blocks,
            attention_heads=attention_heads)
        self.temporal = PyramidTemporalRefiner(widths=tuple(temporal_widths),
            blocks=tuple(temporal_blocks), max_flow_px=max_flow_px,
            residual_parameterization="raw", time_dim=time_dim)
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("alpha_bars", linear_schedule(1000, torch.device("cpu"))["alpha_bars"])

    def _run(self, function, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing and (self.training or
                any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args)):
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def forward(self, state, timestep):
        if state.ndim != 5 or state.shape[1:3] != (7, 4):
            raise ValueError("expected [B,7,4,H,W] diffusion state")
        if timestep.shape != (state.shape[0],) or timestep.dtype != torch.long:
            raise ValueError("timestep must be long [B]")
        if (timestep < 0).any() or (timestep >= 1000).any():
            raise ValueError("expected a timestep in [0,999]")
        # Frame-wise microbatches bound activation memory, without changing capacity.
        base = torch.stack([self._run(self.spatial, state[:, i], timestep)
                            for i in range(7)], dim=1)
        sigma = (1 - self.alpha_bars[timestep]).sqrt()[:, None].expand(-1, 4)
        time = self.spatial.time(timestep)
        features = self._run(self.temporal.encode_features, state, sigma, time)
        outputs = []
        for i in range(7):
            def refine(x, initial, scale, *levels, index=i):
                return self.temporal.refine_features(levels, x, initial, index, scale)
            outputs.append(self._run(refine, state, base[:, i], sigma, *features))
        return torch.stack(outputs, dim=1)


def ddpm_step(state, epsilon, timestep, schedule, noise=None, clip_x0=True):
    """Adjacent stochastic DDPM step; all seven frames follow the same schedule."""
    beta, alpha = schedule["betas"][timestep], schedule["alphas"][timestep]
    bar = schedule["alpha_bars"][timestep]
    previous_bar = schedule["alpha_bars"][timestep - 1] if timestep else bar.new_tensor(1.)
    t = torch.full((state.shape[0],), timestep, device=state.device, dtype=torch.long)
    clean = predict_x0_from_epsilon(state, epsilon, t, schedule["alpha_bars"])
    if clip_x0:
        clean = clean.clamp(-1, 1)  # Model support, never a change to the clean target.
    mean = beta * previous_bar.sqrt() / (1 - bar) * clean
    mean = mean + (1 - previous_bar) * alpha.sqrt() / (1 - bar) * state
    if not timestep:
        return clean
    variance = beta * (1 - previous_bar) / (1 - bar)
    return mean + variance.sqrt() * (torch.randn_like(state) if noise is None else noise)


@torch.no_grad()
def gaussian_restore(model, noisy, start_t, schedule, deterministic=False):
    """For y=x0+sigma*epsilon, start at sqrt(alpha_bar[t])*y, without extra noise."""
    state = schedule["alpha_bars"][start_t].sqrt() * noisy
    for t in range(start_t, -1, -1):
        times = torch.full((state.shape[0],), t, device=state.device, dtype=torch.long)
        epsilon = model(state, times)
        state = ddpm_step(state, epsilon, t, schedule,
                          noise=torch.zeros_like(state) if deterministic else None)
    return state

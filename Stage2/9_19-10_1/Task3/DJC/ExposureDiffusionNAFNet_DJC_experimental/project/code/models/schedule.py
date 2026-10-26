from __future__ import annotations

import torch


def _coefficients_for(x: torch.Tensor, t: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Gather per-sample diffusion coefficients for 4D or 5D tensors."""
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must have shape [batch] matching x")
    return values[t].reshape(t.shape[0], *([1] * (x.ndim - 1)))


def linear_schedule(steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 2e-2, steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


def q_sample(
    x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor
) -> torch.Tensor:
    a = _coefficients_for(x0, t, alpha_bars)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise


def predict_x0_from_epsilon(
    x_t: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = _coefficients_for(x_t, timestep, alpha_bars)
    return (x_t - torch.sqrt(1.0 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar)


def v_target(
    x0: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Stable Imagen/SD-style v target for VP diffusion."""
    alpha_bar = _coefficients_for(x0, timestep, alpha_bars)
    return alpha_bar.sqrt() * noise - (1.0 - alpha_bar).sqrt() * x0


def predict_x0_from_v(
    x_t: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = _coefficients_for(x_t, timestep, alpha_bars)
    return alpha_bar.sqrt() * x_t - (1.0 - alpha_bar).sqrt() * v


def predict_epsilon_from_v(
    x_t: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = _coefficients_for(x_t, timestep, alpha_bars)
    return (1.0 - alpha_bar).sqrt() * x_t + alpha_bar.sqrt() * v


def ddim_step(
    x_t: torch.Tensor,
    x0_prediction: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    timestep: int,
    previous_timestep: int,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Deterministic DDIM update (eta=0) used for reproducible R1 sampling."""
    if previous_timestep < -1 or previous_timestep >= timestep:
        raise ValueError("previous_timestep must be in [-1, timestep)")
    previous_alpha = (
        torch.ones((), device=x_t.device, dtype=x_t.dtype)
        if previous_timestep < 0
        else alpha_bars[previous_timestep].to(dtype=x_t.dtype)
    )
    return torch.sqrt(previous_alpha) * x0_prediction + torch.sqrt(
        1.0 - previous_alpha
    ) * epsilon_prediction


def sample_timesteps(training_steps: int, sampling_steps: int) -> list[int]:
    """Strictly decreasing evenly-spaced DDIM schedule including endpoints."""
    if training_steps <= 1 or sampling_steps <= 0:
        raise ValueError("training_steps must exceed 1 and sampling_steps must be positive")
    if sampling_steps > training_steps:
        raise ValueError("sampling_steps cannot exceed training_steps")
    points = torch.linspace(training_steps - 1, 0, sampling_steps).round().long().tolist()
    result: list[int] = []
    for point in points:
        if not result or point < result[-1]:
            result.append(int(point))
    if result[-1] != 0:
        result.append(0)
    return result

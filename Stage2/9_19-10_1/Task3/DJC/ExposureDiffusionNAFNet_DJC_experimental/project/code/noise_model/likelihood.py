"""Differentiable empirical RAW residual likelihoods.

The Poisson-Gaussian function is retained only as an explicit ablation baseline.
The default prototype uses empirical residual samples from dark frames instead.
"""

from __future__ import annotations

import torch


def empirical_residual_log_prob(
    y: torch.Tensor,
    x: torch.Tensor,
    residual_samples: torch.Tensor,
    bandwidth_dn: float,
    offset_dn: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Estimate log p(y|x) from real residual samples with a Laplace kernel.

    The kernel is only a differentiable interpolation of an empirical residual
    distribution. It is not a physical Poisson-Gaussian sensor assumption.
    ``residual_samples`` should come from dark-frame decomposition and be held
    out by frame when validating the model.
    """
    if bandwidth_dn <= 0:
        raise ValueError("bandwidth_dn must be positive")
    samples = residual_samples.to(dtype=torch.float32, device=y.device).reshape(-1)
    if samples.numel() == 0:
        raise ValueError("residual_samples must not be empty")
    residual = y.to(torch.float32) - x.to(torch.float32) - torch.as_tensor(
        offset_dn, dtype=torch.float32, device=y.device
    )
    delta = residual[..., None] - samples
    log_kernel = -torch.abs(delta) / bandwidth_dn - torch.log(
        torch.as_tensor(2.0 * bandwidth_dn, dtype=residual.dtype, device=residual.device)
    )
    return torch.logsumexp(log_kernel, dim=-1) - torch.log(
        torch.as_tensor(float(samples.numel()), dtype=residual.dtype, device=residual.device)
    )


def poisson_gaussian_log_prob(
    y: torch.Tensor,
    x: torch.Tensor,
    gain_dn_per_electron: float,
    sigma_dn: float,
    offset_dn: torch.Tensor | float = 0.0,
    tail_sigma: float = 8.0,
) -> torch.Tensor:
    """Return per-element log p(y|x) using a finite Poisson sum.

    x and y are DN-equivalent tensors. The summation range is selected from the
    current lambda range and is detached from autograd so bounds do not affect
    gradients.
    """
    if gain_dn_per_electron <= 0 or sigma_dn <= 0:
        raise ValueError("gain_dn_per_electron and sigma_dn must be positive")
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    offset = torch.as_tensor(offset_dn, dtype=x.dtype, device=x.device)
    lam = torch.clamp(x / gain_dn_per_electron, min=1e-6)
    max_lam = float(lam.detach().max().cpu())
    min_lam = float(lam.detach().min().cpu())
    low = max(0, int(min_lam - tail_sigma * (min_lam + 1.0) ** 0.5))
    high = int(max_lam + tail_sigma * (max_lam + 1.0) ** 0.5) + 1
    high = max(high, low + 1)
    n = torch.arange(low, high, dtype=x.dtype, device=x.device)
    log_pois = n.reshape((-1,) + (1,) * x.ndim) * torch.log(lam) - lam
    log_pois = log_pois - torch.lgamma(n.reshape((-1,) + (1,) * x.ndim) + 1.0)
    means = gain_dn_per_electron * n.reshape((-1,) + (1,) * x.ndim) + offset
    normal = -0.5 * ((y - means) / sigma_dn) ** 2 - torch.log(
        torch.as_tensor(sigma_dn * (2.0 * torch.pi) ** 0.5, device=x.device)
    )
    return torch.logsumexp(log_pois + normal, dim=0)


def poisson_gaussian_sample(
    x_dn: torch.Tensor,
    gain_dn_per_electron: float,
    sigma_dn: float,
    offset_dn: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    lam = torch.clamp(x_dn / gain_dn_per_electron, min=0.0)
    counts = torch.poisson(lam)
    return counts * gain_dn_per_electron + torch.randn_like(x_dn) * sigma_dn + offset_dn

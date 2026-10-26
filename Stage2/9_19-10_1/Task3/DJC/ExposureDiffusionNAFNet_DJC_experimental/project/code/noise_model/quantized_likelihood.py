"""Differentiable quantized RAW likelihood.

The sensor model follows the photon-transfer formulation used by EMVA 1288
and the Poisson signal-dependent term used by CVPR 2026 2-Shots.  The dark
residual is represented by a small Gaussian mixture fitted from dark frames.
The probability of an observed DN is an interval probability over the ADC
quantization bin, rather than a continuous density evaluated at the code.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def fit_gaussian_mixture_1d(
    samples: np.ndarray,
    components: int,
    max_iter: int = 100,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a small one-dimensional Gaussian mixture with deterministic EM."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if components not in (1, 3, 5):
        raise ValueError("components must be one of 1, 3, or 5")
    if values.size < components * 20:
        raise ValueError("too few finite samples for Gaussian mixture")
    rng = np.random.default_rng(seed)
    global_std = max(float(np.std(values)), 1e-3)
    if components == 1:
        means = np.array([np.mean(values)], dtype=np.float64)
        sigmas = np.array([global_std], dtype=np.float64)
        weights = np.ones(1, dtype=np.float64)
        return weights, means, sigmas
    quantiles = np.linspace(0.05, 0.95, components)
    means = np.quantile(values, quantiles).astype(np.float64)
    means += rng.normal(0.0, global_std * 1e-3, size=components)
    sigmas = np.full(components, max(global_std / components, 1e-3), dtype=np.float64)
    weights = np.full(components, 1.0 / components, dtype=np.float64)
    log2pi = math.log(2.0 * math.pi)
    for _ in range(max_iter):
        log_prob = (
            np.log(np.maximum(weights, 1e-12))[None]
            - 0.5 * ((values[:, None] - means[None]) / sigmas[None]) ** 2
            - np.log(np.maximum(sigmas, 1e-12))[None]
            - 0.5 * log2pi
        )
        max_log = np.max(log_prob, axis=1, keepdims=True)
        responsibility = np.exp(log_prob - max_log)
        responsibility /= np.maximum(responsibility.sum(axis=1, keepdims=True), 1e-20)
        counts = responsibility.sum(axis=0)
        new_weights = counts / values.size
        new_means = (responsibility * values[:, None]).sum(axis=0) / np.maximum(counts, 1e-12)
        new_vars = (
            responsibility * (values[:, None] - new_means[None]) ** 2
        ).sum(axis=0) / np.maximum(counts, 1e-12)
        new_sigmas = np.sqrt(np.maximum(new_vars, 1e-6))
        if np.max(np.abs(new_means - means)) < 1e-5 and np.max(
            np.abs(new_sigmas - sigmas)
        ) < 1e-5:
            weights, means, sigmas = new_weights, new_means, new_sigmas
            break
        weights, means, sigmas = new_weights, new_means, new_sigmas
    order = np.argsort(means)
    return weights[order], means[order], sigmas[order]


def _log_interval_probability(
    lower: torch.Tensor,
    upper: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Log normal probability in [lower, upper], stable enough for RAW bins."""
    # torch.special.ndtr is the standard-normal CDF Φ, so its argument is
    # (value - mean) / sigma.  The sqrt(2) factor belongs only to an erf-based
    # implementation of Φ and must not be applied here.
    scale = sigma
    # Avoid evaluating infinities in the unused ordinary-bin branch: MPS
    # backward otherwise propagates NaNs through `where` at a clipped code.
    upper_is_inf = torch.isposinf(upper)
    lower_is_inf = torch.isneginf(lower)
    safe_upper = torch.where(upper_is_inf, mean, upper)
    safe_lower = torch.where(lower_is_inf, mean, lower)
    upper_standard = (safe_upper - mean) / scale
    lower_standard = (safe_lower - mean) / scale
    upper_cdf = torch.special.ndtr(upper_standard)
    lower_cdf = torch.special.ndtr(lower_standard)
    probability = torch.clamp(upper_cdf - lower_cdf, min=torch.finfo(mean.dtype).tiny)
    ordinary = torch.log(probability)
    # `ndtr(-z)` is a stable survival probability for an upper-clipped code;
    # using `1 - ndtr(z)` would destroy its gradient near saturation.
    lower_tail = torch.log(torch.clamp(upper_cdf, min=torch.finfo(mean.dtype).tiny))
    upper_tail = torch.log(
        torch.clamp(torch.special.ndtr(-lower_standard), min=torch.finfo(mean.dtype).tiny)
    )
    return torch.where(lower_is_inf, lower_tail, torch.where(upper_is_inf, upper_tail, ordinary))


def _observation_bounds(
    y: torch.Tensor,
    quantization_step_dn: float,
    lower_clip_dn: float | None,
    upper_clip_dn: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = y - quantization_step_dn / 2.0
    upper = y + quantization_step_dn / 2.0
    if lower_clip_dn is not None:
        lower = torch.where(y <= float(lower_clip_dn), torch.full_like(lower, -torch.inf), lower)
    if upper_clip_dn is not None:
        upper = torch.where(y >= float(upper_clip_dn), torch.full_like(upper, torch.inf), upper)
    return lower, upper


def _quantized_gaussian_mixture_log_prob(
    y: torch.Tensor,
    x: torch.Tensor,
    gain_dn_per_electron: float,
    mixture_weights: torch.Tensor,
    mixture_means_dn: torch.Tensor,
    mixture_sigmas_dn: torch.Tensor,
    dark_offset_dn: torch.Tensor,
    quantization_step_dn: float,
    lower_clip_dn: float | None,
    upper_clip_dn: float | None,
) -> torch.Tensor:
    """Large-count approximation to the Poisson/readout convolution.

    If ``lambda=x/K`` is large, the Poisson count is well approximated by a
    normal random variable.  In DN units its variance is ``K*x``.  Keeping the
    mixture components separate preserves asymmetric and heavy-tailed readout
    residuals while avoiding a Poisson sum with thousands of terms.
    """
    variance = torch.clamp(
        float(gain_dn_per_electron) * torch.clamp(x, min=0.0)
        + mixture_sigmas_dn[:, None] ** 2,
        min=torch.finfo(x.dtype).tiny,
    )
    lower, upper = _observation_bounds(
        y, quantization_step_dn, lower_clip_dn, upper_clip_dn
    )
    component_log: list[torch.Tensor] = []
    for index in range(mixture_weights.numel()):
        mean = x + dark_offset_dn + mixture_means_dn[index]
        sigma = torch.sqrt(variance[index])
        interval = _log_interval_probability(lower, upper, mean, sigma)
        component_log.append(
            torch.log(torch.clamp(mixture_weights[index], min=1e-30)) + interval
        )
    return torch.logsumexp(torch.stack(component_log, dim=0), dim=0)


def _quantized_exact_poisson_log_prob(
    y: torch.Tensor,
    x: torch.Tensor,
    gain_dn_per_electron: float,
    mixture_weights: torch.Tensor,
    mixture_means_dn: torch.Tensor,
    mixture_sigmas_dn: torch.Tensor,
    dark_offset_dn: torch.Tensor,
    quantization_step_dn: float,
    tail_sigma: float,
    lower_clip_dn: float | None,
    upper_clip_dn: float | None,
) -> torch.Tensor:
    """Exact finite Poisson sum for the low-count part of the RAW domain."""
    lam = torch.clamp(x / float(gain_dn_per_electron), min=0.0)
    max_lam = float(lam.detach().max().cpu()) if lam.numel() else 0.0
    high = max(1, int(math.ceil(max_lam + tail_sigma * math.sqrt(max_lam + 1.0))) + 1)
    counts = torch.arange(high, dtype=x.dtype, device=x.device)
    counts_view = counts.reshape((high,) + (1,) * x.ndim)
    lam_view = lam.unsqueeze(0)
    log_poisson = counts_view * torch.log(torch.clamp(lam_view, min=1e-30)) - lam_view
    log_poisson -= torch.lgamma(counts_view + 1.0)
    zero_signal = lam_view <= 1e-30
    log_poisson = torch.where(
        zero_signal & (counts_view == 0), torch.zeros_like(log_poisson), log_poisson
    )
    log_poisson = torch.where(
        zero_signal & (counts_view > 0), torch.full_like(log_poisson, -torch.inf), log_poisson
    )
    lower, upper = _observation_bounds(
        y, quantization_step_dn, lower_clip_dn, upper_clip_dn
    )
    lower = lower.unsqueeze(0)
    upper = upper.unsqueeze(0)
    component_log: list[torch.Tensor] = []
    for index in range(mixture_weights.numel()):
        mean = counts_view * float(gain_dn_per_electron) + dark_offset_dn + mixture_means_dn[index]
        interval = _log_interval_probability(lower, upper, mean, mixture_sigmas_dn[index])
        component_log.append(
            torch.log(torch.clamp(mixture_weights[index], min=1e-30)) + interval
        )
    log_readout = torch.logsumexp(torch.stack(component_log, dim=0), dim=0)
    return torch.logsumexp(log_poisson + log_readout, dim=0)


def quantized_poisson_gaussian_log_prob(
    y_dn: torch.Tensor,
    x_dn: torch.Tensor,
    gain_dn_per_electron: float,
    mixture_weights: torch.Tensor,
    mixture_means_dn: torch.Tensor,
    mixture_sigmas_dn: torch.Tensor,
    dark_offset_dn: torch.Tensor | float = 0.0,
    quantization_step_dn: float = 1.0,
    tail_sigma: float = 8.0,
    exact_lambda_threshold: float = 64.0,
    lower_clip_dn: float | None = 0.0,
    upper_clip_dn: float | None = 4095.0,
) -> torch.Tensor:
    """Return per-pixel log p(observed code | signal) in DN units.

    ``x_dn`` is black-subtracted linear signal and may require gradients.
    ``dark_offset_dn`` can be a fixed dark map.  The Gaussian mixture is the
    signal-independent residual after marginalizing frame/band components.
    ``quantization_step_dn`` is 2 for the validated dark-frame lattice and 1
    for an illuminated observation until an illuminated ADC model is fitted.
    """
    if gain_dn_per_electron <= 0 or quantization_step_dn <= 0:
        raise ValueError("gain and quantization_step_dn must be positive")
    y = y_dn.to(dtype=torch.float32)
    x = torch.clamp(x_dn.to(dtype=torch.float32), min=0.0)
    offset = torch.as_tensor(dark_offset_dn, dtype=y.dtype, device=y.device)
    weights = mixture_weights.to(dtype=y.dtype, device=y.device)
    means = mixture_means_dn.to(dtype=y.dtype, device=y.device)
    sigmas = mixture_sigmas_dn.to(dtype=y.dtype, device=y.device)
    if weights.ndim != 1 or means.shape != weights.shape or sigmas.shape != weights.shape:
        raise ValueError("mixture parameters must be one-dimensional and same-sized")
    if not torch.isfinite(torch.cat((weights, means, sigmas))).all() or torch.any(sigmas <= 0):
        raise ValueError("invalid mixture parameters")
    weights = weights / torch.clamp(weights.sum(), min=torch.finfo(y.dtype).tiny)
    if exact_lambda_threshold <= 0:
        raise ValueError("exact_lambda_threshold must be positive")
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    flat_offset = offset.expand_as(x).reshape(-1)
    lam = flat_x / float(gain_dn_per_electron)
    low_mask = lam <= float(exact_lambda_threshold)
    output = torch.empty_like(flat_x)
    if torch.any(low_mask):
        output[low_mask] = _quantized_exact_poisson_log_prob(
            flat_y[low_mask],
            flat_x[low_mask],
            gain_dn_per_electron,
            weights,
            means,
            sigmas,
            flat_offset[low_mask],
            quantization_step_dn,
            tail_sigma,
            lower_clip_dn,
            upper_clip_dn,
        )
    if torch.any(~low_mask):
        output[~low_mask] = _quantized_gaussian_mixture_log_prob(
            flat_y[~low_mask],
            flat_x[~low_mask],
            gain_dn_per_electron,
            weights,
            means,
            sigmas,
            flat_offset[~low_mask],
            quantization_step_dn,
            lower_clip_dn,
            upper_clip_dn,
        )
    return output.reshape(y.shape)


def sample_quantized_poisson_gaussian(
    x_dn: torch.Tensor,
    gain_dn_per_electron: float,
    mixture_weights: torch.Tensor,
    mixture_means_dn: torch.Tensor,
    mixture_sigmas_dn: torch.Tensor,
    dark_offset_dn: torch.Tensor | float = 0.0,
    quantization_step_dn: float = 1.0,
    white_level: float = 4095.0,
) -> torch.Tensor:
    """Sample an illuminated RAW code from the calibrated observation model."""
    if gain_dn_per_electron <= 0 or quantization_step_dn <= 0:
        raise ValueError("gain and quantization_step_dn must be positive")
    x = torch.clamp(x_dn.to(torch.float32), min=0.0)
    weights = mixture_weights.to(dtype=x.dtype, device=x.device)
    means = mixture_means_dn.to(dtype=x.dtype, device=x.device)
    sigmas = mixture_sigmas_dn.to(dtype=x.dtype, device=x.device)
    if weights.ndim != 1 or means.shape != weights.shape or sigmas.shape != weights.shape:
        raise ValueError("mixture parameters must be one-dimensional and same-sized")
    weights = weights / torch.clamp(weights.sum(), min=torch.finfo(x.dtype).tiny)
    flat = x.reshape(-1)
    component = torch.multinomial(weights.expand(flat.numel(), -1), 1).squeeze(1)
    counts = torch.poisson(flat / float(gain_dn_per_electron))
    readout = means[component] + sigmas[component] * torch.randn_like(flat)
    offset = torch.as_tensor(dark_offset_dn, dtype=x.dtype, device=x.device).expand_as(x).reshape(-1)
    observed = counts * float(gain_dn_per_electron) + offset + readout
    observed = torch.round(observed / quantization_step_dn) * quantization_step_dn
    return torch.clamp(observed.reshape_as(x), 0.0, float(white_level))

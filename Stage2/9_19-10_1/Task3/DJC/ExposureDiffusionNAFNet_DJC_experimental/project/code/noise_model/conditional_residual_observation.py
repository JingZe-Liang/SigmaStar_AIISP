"""Brightness-conditional residual observation model for static company ROIs.

This is a candidate model motivated by ``structure_exploration_20260929``.
It models a residual CDF per Bayer channel and signal node instead of forcing
the illuminated residual into a symmetric Gaussian.  The fitted CDF is used
both for synthesis and for the differentiable, quantized DPS likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ConditionalResidualObservationModel:
    profile_name: str
    black_level_dn: float
    signal_nodes_dn: torch.Tensor  # [4, N], strictly increasing
    residual_quantiles_dn: torch.Tensor  # [4, N, Q]
    probabilities: torch.Tensor  # [Q], strictly increasing in (0, 1)
    horizontal_correlation: torch.Tensor  # [4], weak same-channel packed-pixel rho
    supported_signal_max_dn: torch.Tensor  # [4]
    bandwidth_dn: float = 0.75
    white_level: float = 4095.0

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | str = "cpu"
    ) -> "ConditionalResidualObservationModel":
        with np.load(path) as archive:
            return cls(
                profile_name=str(archive["profile_name"].item()),
                black_level_dn=float(archive["black_level_dn"]),
                signal_nodes_dn=torch.from_numpy(archive["signal_nodes_dn"].astype(np.float32)).to(device),
                residual_quantiles_dn=torch.from_numpy(archive["residual_quantiles_dn"].astype(np.float32)).to(device),
                probabilities=torch.from_numpy(archive["probabilities"].astype(np.float32)).to(device),
                horizontal_correlation=torch.from_numpy(archive["horizontal_correlation"].astype(np.float32)).to(device),
                supported_signal_max_dn=torch.from_numpy(archive["supported_signal_max_dn"].astype(np.float32)).to(device),
                bandwidth_dn=float(archive["bandwidth_dn"]),
                white_level=float(archive["white_level"]),
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            model_type=np.asarray("conditional_residual_v1"),
            profile_name=np.asarray(self.profile_name),
            black_level_dn=np.float32(self.black_level_dn),
            signal_nodes_dn=self.signal_nodes_dn.detach().cpu().numpy(),
            residual_quantiles_dn=self.residual_quantiles_dn.detach().cpu().numpy(),
            probabilities=self.probabilities.detach().cpu().numpy(),
            horizontal_correlation=self.horizontal_correlation.detach().cpu().numpy(),
            supported_signal_max_dn=self.supported_signal_max_dn.detach().cpu().numpy(),
            bandwidth_dn=np.float32(self.bandwidth_dn),
            white_level=np.float32(self.white_level),
            illuminated_quantization_step_dn=np.int32(1),
            fixed_pattern_included=np.bool_(False),
        )

    def _validate(self, values: torch.Tensor) -> None:
        if values.ndim != 4 or values.shape[1] != 4:
            raise ValueError("RAW tensors must have shape [B,4,H,W]")
        if not torch.isfinite(values).all():
            raise ValueError("RAW tensors must be finite")
        if self.signal_nodes_dn.shape[:1] != (4,) or self.residual_quantiles_dn.shape[:2] != self.signal_nodes_dn.shape:
            raise ValueError("conditional residual table has inconsistent channel/node dimensions")
        if self.residual_quantiles_dn.shape[-1] != self.probabilities.numel():
            raise ValueError("conditional residual table has inconsistent quantiles")
        if self.bandwidth_dn <= 0:
            raise ValueError("bandwidth_dn must be positive")

    @staticmethod
    def _linear(values: torch.Tensor, nodes: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Interpolate a channel-specific [N,...] table over signal DN."""
        clipped = torch.clamp(values, min=float(nodes[0]), max=float(nodes[-1]))
        right = torch.bucketize(clipped.reshape(-1), nodes).reshape_as(clipped)
        right = torch.clamp(right, 1, len(nodes) - 1)
        left = right - 1
        low = nodes[left]
        high = nodes[right]
        fraction = (clipped - low) / torch.clamp(high - low, min=1e-6)
        return table[left] * (1.0 - fraction[..., None]) + table[right] * fraction[..., None]

    def _quantiles(self, signal_dn: torch.Tensor) -> torch.Tensor:
        self._validate(signal_dn)
        result = []
        for channel in range(4):
            result.append(self._linear(
                torch.clamp(signal_dn[:, channel], min=0.0),
                self.signal_nodes_dn[channel],
                self.residual_quantiles_dn[channel],
            ))
        return torch.stack(result, dim=1)

    def limit_signal(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> torch.Tensor:
        del top, left, saturation_safety_sigma
        self._validate(signal_dn)
        if saturation_margin_dn < 0:
            raise ValueError("saturation_margin_dn must be nonnegative")
        # ``supported_signal_max_dn`` is a validation boundary, never a
        # physical saturation boundary.  Clamping the latent clean target to
        # it silently changes supervision for bright pixels.  Interpolation
        # in ``_quantiles`` uses the terminal measured residual distribution
        # outside that range, while this method limits only sensor saturation.
        maximum = torch.full_like(
            signal_dn, self.white_level - self.black_level_dn - saturation_margin_dn
        )
        return torch.minimum(torch.clamp(signal_dn, min=0.0), maximum)

    def _correlated_uniform(self, shape: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        white = torch.randn(shape, device=device)
        rho = torch.clamp(self.horizontal_correlation.to(device=device), min=-0.2, max=0.2)
        output = torch.empty_like(white)
        output[..., 0] = white[..., 0]
        scale = torch.sqrt(torch.clamp(1.0 - rho.square(), min=1e-6)).reshape(1, 4, 1)
        rho = rho.reshape(1, 4, 1)
        for column in range(1, shape[-1]):
            output[..., column] = rho * output[..., column - 1] + scale * white[..., column]
        return torch.special.ndtr(output)

    def _inverse_cdf(self, quantiles: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
        probabilities = self.probabilities.to(device=uniform.device, dtype=uniform.dtype)
        right = torch.bucketize(uniform.reshape(-1), probabilities).reshape_as(uniform)
        right = torch.clamp(right, 1, probabilities.numel() - 1)
        left = right - 1
        lo_p, hi_p = probabilities[left], probabilities[right]
        lo_q = torch.gather(quantiles, -1, left[..., None]).squeeze(-1)
        hi_q = torch.gather(quantiles, -1, right[..., None]).squeeze(-1)
        return lo_q + (uniform - lo_p) * (hi_q - lo_q) / torch.clamp(hi_p - lo_p, min=1e-6)

    def sample(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        illuminated_quantization_step_dn: float = 1.0,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_strength(
            signal_dn, top, left, 1.0, illuminated_quantization_step_dn,
            saturation_margin_dn, saturation_safety_sigma,
        )

    def sample_strength(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        noise_strength: float = 1.0,
        illuminated_quantization_step_dn: float = 1.0,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training augmentation: scale centered empirical noise before ADC.

        Strength 1 exactly retains the accepted camera sampler. Other values
        are augmentation levels, not calibrated ISO/gain conditions. Keep the
        conditional mean, black level, spatial copula and quantization fixed.
        """
        if not np.isfinite(noise_strength) or noise_strength <= 0:
            raise ValueError("noise_strength must be finite and positive")
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("candidate model validates 1-DN illuminated quantization only")
        signal = self.limit_signal(signal_dn, top, left, saturation_margin_dn, saturation_safety_sigma)
        quantiles = self._quantiles(signal)
        residual = self._inverse_cdf(quantiles, self._correlated_uniform(tuple(signal.shape), signal.device))
        if noise_strength != 1.0:
            p = self.probabilities
            mean = torch.trapezoid(quantiles, p, dim=-1)
            slope_lo = (quantiles[..., 1] - quantiles[..., 0]) / (p[1] - p[0])
            slope_hi = (quantiles[..., -1] - quantiles[..., -2]) / (p[-1] - p[-2])
            mean = mean + p[0] * quantiles[..., 0] - 0.5 * p[0].square() * slope_lo
            mean = mean + (1 - p[-1]) * quantiles[..., -1] + 0.5 * (1 - p[-1]).square() * slope_hi
            residual = mean + noise_strength * (residual - mean)
        observed = torch.round(signal + self.black_level_dn + residual)
        saturated = observed >= self.white_level
        return torch.clamp(observed, 0.0, self.white_level), saturated

    def observation_to_signal(self, observation_dn: torch.Tensor, top: int | torch.Tensor, left: int | torch.Tensor,
                              clip_min: float | None = 0.0) -> torch.Tensor:
        del top, left
        self._validate(observation_dn)
        signal = observation_dn - self.black_level_dn
        return signal if clip_min is None else signal.clamp_min(clip_min)

    def log_prob(
        self,
        observation_dn: torch.Tensor,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        illuminated_quantization_step_dn: float = 1.0,
    ) -> torch.Tensor:
        del top, left
        self._validate(observation_dn)
        self._validate(signal_dn)
        if observation_dn.shape != signal_dn.shape:
            raise ValueError("observation_dn and signal_dn must have equal shapes")
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("candidate model validates 1-DN illuminated quantization only")
        residual = observation_dn - self.black_level_dn - torch.clamp(signal_dn, min=0.0)
        samples = self._quantiles(signal_dn)
        bandwidth = torch.as_tensor(self.bandwidth_dn, device=signal_dn.device, dtype=signal_dn.dtype)
        upper = torch.special.ndtr((residual[..., None] + 0.5 - samples) / bandwidth)
        lower = torch.special.ndtr((residual[..., None] - 0.5 - samples) / bandwidth)
        return torch.log(torch.clamp((upper - lower).mean(dim=-1), min=1e-12))

    def noise_std(self, signal_dn: torch.Tensor) -> torch.Tensor:
        quantiles = self._quantiles(signal_dn)
        return torch.sqrt(torch.clamp(quantiles.var(dim=-1).median(), min=1e-6)) / self.white_level

"""Differentiable observation model fitted from repeated company RAW frames.

The high-gain company clips contain repeated views of a static scene.  Their
adjacent-frame photon-transfer fit identifies the temporal, signal-dependent
component of the observation noise.  This class turns those held-out
statistics into the same quantized Poisson-Gaussian likelihood used by DPS.

It deliberately contains no fixed-pattern map: a static illuminated clip
cannot distinguish FPN from scene texture.  A future dark/flat acquisition
can add that term without changing the public observation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .quantized_likelihood import quantized_poisson_gaussian_log_prob


@dataclass
class TemporalObservationModel:
    """Per-CFA-channel quantized Poisson-Gaussian company likelihood."""

    profile_name: str
    gain_dn_per_electron: torch.Tensor  # [4], PTC slope in DN^2 / DN
    read_variance_dn2: torch.Tensor  # [4], PTC intercept in DN^2
    black_level_dn: float
    supported_signal_max_dn: torch.Tensor  # [4], largest held-out PTC bin
    white_level: float = 4095.0

    @classmethod
    def from_measurement(
        cls,
        profile_name: str,
        profile: dict[str, Any],
        measurement: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> "TemporalObservationModel":
        fits = measurement.get("fit_channels")
        if not isinstance(fits, list) or len(fits) != 4:
            raise ValueError(f"{profile_name} lacks four fitted temporal channels")
        if not bool(measurement.get("temporal_stability_pass")):
            raise ValueError(f"{profile_name} failed repeated-frame holdout stability")
        if not bool(measurement.get("positive_variance_slope")):
            raise ValueError(f"{profile_name} lacks a positive photon-transfer slope")
        slopes = torch.tensor(
            [float(item["variance_slope_dn"]) for item in fits],
            device=device,
            dtype=torch.float32,
        )
        intercepts = torch.tensor(
            [float(item["variance_intercept_dn2"]) for item in fits],
            device=device,
            dtype=torch.float32,
        )
        maxima = torch.tensor(
            [float(item["signal_max_dn"]) for item in fits],
            device=device,
            dtype=torch.float32,
        )
        if not torch.isfinite(slopes).all() or not torch.isfinite(intercepts).all():
            raise ValueError(f"{profile_name} has non-finite temporal parameters")
        if torch.any(slopes <= 0) or torch.any(intercepts < 0) or torch.any(maxima <= 0):
            raise ValueError(f"{profile_name} has invalid temporal parameters")
        return cls(
            profile_name=profile_name,
            gain_dn_per_electron=slopes,
            read_variance_dn2=intercepts,
            black_level_dn=float(profile["black_level_dn"]),
            supported_signal_max_dn=maxima,
        )

    def _channel_view(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(1, 4, 1, 1).to(device=self.gain_dn_per_electron.device)

    def _validate(self, signal_dn: torch.Tensor) -> None:
        if signal_dn.ndim != 4 or signal_dn.shape[1] != 4:
            raise ValueError("signal_dn must have shape [B,4,H,W]")
        if not torch.isfinite(signal_dn).all():
            raise ValueError("signal_dn contains non-finite values")

    def limit_signal(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> torch.Tensor:
        """Limit to the measured PTC range and the sensor saturation tail."""
        del top, left
        self._validate(signal_dn)
        if saturation_margin_dn < 0 or saturation_safety_sigma < 0:
            raise ValueError("saturation margin and safety sigma must be nonnegative")
        gain = self._channel_view(self.gain_dn_per_electron)
        read = self._channel_view(self.read_variance_dn2)
        headroom = max(self.white_level - self.black_level_dn - saturation_margin_dn, 0.0)
        z = float(saturation_safety_sigma)
        coefficient = 2.0 * headroom + z * z * gain
        constant = headroom * headroom - z * z * read
        discriminant = torch.clamp(coefficient.square() - 4.0 * constant, min=0.0)
        saturation_limit = torch.clamp((coefficient - torch.sqrt(discriminant)) / 2.0, min=0.0)
        calibrated_limit = self._channel_view(self.supported_signal_max_dn)
        maximum = torch.minimum(saturation_limit, calibrated_limit)
        return torch.minimum(torch.clamp(signal_dn, min=0.0), maximum)

    def sample(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        illuminated_quantization_step_dn: float = 1.0,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a 1-DN illuminated company RAW observation."""
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("illuminated company likelihood validates a 1-DN ADC step")
        signal = self.limit_signal(
            signal_dn, top, left, saturation_margin_dn, saturation_safety_sigma
        )
        gain = self._channel_view(self.gain_dn_per_electron)
        read_std = torch.sqrt(self._channel_view(self.read_variance_dn2))
        shot = torch.poisson(signal / gain) * gain
        observed = torch.round(shot + self.black_level_dn + read_std * torch.randn_like(signal))
        saturated = observed >= self.white_level
        return torch.clamp(observed, 0.0, self.white_level), saturated

    def observation_to_signal(
        self,
        observation_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        clip_min: float | None = 0.0,
    ) -> torch.Tensor:
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
        """Factorized per-pixel quantized PG likelihood for DPS guidance."""
        del top, left
        self._validate(observation_dn)
        self._validate(signal_dn)
        if observation_dn.shape != signal_dn.shape:
            raise ValueError("observation_dn and signal_dn must have equal shapes")
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("illuminated company likelihood validates a 1-DN ADC step")
        terms = []
        for channel in range(4):
            terms.append(
                quantized_poisson_gaussian_log_prob(
                    observation_dn[:, channel],
                    torch.clamp(signal_dn[:, channel], min=0.0),
                    float(self.gain_dn_per_electron[channel]),
                    torch.ones(1, device=signal_dn.device),
                    torch.zeros(1, device=signal_dn.device),
                    torch.sqrt(self.read_variance_dn2[channel]).reshape(1),
                    torch.full_like(observation_dn[:, channel], self.black_level_dn),
                    quantization_step_dn=illuminated_quantization_step_dn,
                )
            )
        return torch.stack(terms, dim=1)

    def noise_std(self, signal_dn: torch.Tensor) -> torch.Tensor:
        """Return a scalar normalized noise level for the prior-only baseline."""
        self._validate(signal_dn)
        gain = self._channel_view(self.gain_dn_per_electron)
        read = self._channel_view(self.read_variance_dn2)
        variance = gain * torch.clamp(signal_dn, min=0.0) + read
        return torch.sqrt(torch.median(variance)) / self.white_level

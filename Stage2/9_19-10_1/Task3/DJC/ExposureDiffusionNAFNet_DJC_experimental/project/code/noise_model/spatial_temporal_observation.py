"""Spatially correlated high-gain RAW observation model.

The model is fitted only from repeated frames of one static company scene.
Frame differences remove the scene mean, then a signal-normalized Fourier
power spectrum estimates the temporal row/column/pixel covariance.  This is
the stationary residual counterpart of PNNP's dark-residual decomposition;
it does not use the static scene mean as an FPN map.

The likelihood is a differentiable colored-Gaussian approximation in the
illuminated 1-DN domain.  At these high-gain profiles the measured temporal
standard deviation is several DN or larger, so the 1-DN quantization interval
is retained in sampling and negligible for the Fourier likelihood gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class SpatialTemporalObservationModel:
    profile_name: str
    gain_dn_per_electron: torch.Tensor  # [4]
    read_variance_dn2: torch.Tensor  # [4]
    black_level_dn: float
    supported_signal_max_dn: torch.Tensor  # [4]
    variance_scale_map: torch.Tensor  # [4,H,W], temporal variance / PTC variance
    normalized_power: torch.Tensor  # [4,H,W], E|FFT(z)|² / (H*W)
    white_level: float = 4095.0

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "SpatialTemporalObservationModel":
        with np.load(path) as archive:
            return cls(
                str(archive["profile_name"].item()),
                torch.from_numpy(archive["gain_dn_per_electron"].astype(np.float32)).to(device),
                torch.from_numpy(archive["read_variance_dn2"].astype(np.float32)).to(device),
                float(archive["black_level_dn"]),
                torch.from_numpy(archive["supported_signal_max_dn"].astype(np.float32)).to(device),
                torch.from_numpy(archive["variance_scale_map"].astype(np.float32)).to(device),
                torch.from_numpy(archive["normalized_power"].astype(np.float32)).to(device),
                float(archive["white_level"]),
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            profile_name=np.asarray(self.profile_name),
            gain_dn_per_electron=self.gain_dn_per_electron.detach().cpu().numpy(),
            read_variance_dn2=self.read_variance_dn2.detach().cpu().numpy(),
            black_level_dn=np.float32(self.black_level_dn),
            supported_signal_max_dn=self.supported_signal_max_dn.detach().cpu().numpy(),
            variance_scale_map=self.variance_scale_map.detach().cpu().numpy(),
            normalized_power=self.normalized_power.detach().cpu().numpy(),
            white_level=np.float32(self.white_level),
            illuminated_quantization_step_dn=np.int32(1),
            fixed_pattern_included=np.bool_(False),
        )

    def _validate(self, values: torch.Tensor) -> None:
        if values.ndim != 4 or values.shape[1] != 4:
            raise ValueError("RAW tensors must have shape [B,4,H,W]")
        if not torch.isfinite(values).all():
            raise ValueError("RAW tensor contains non-finite values")

    def _channel_view(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(1, 4, 1, 1).to(device=self.gain_dn_per_electron.device)

    def _crop_scale(
        self, top: int | torch.Tensor, left: int | torch.Tensor, batch: int, height: int, width: int
    ) -> torch.Tensor:
        tops = torch.as_tensor(top, device="cpu").reshape(-1).tolist()
        lefts = torch.as_tensor(left, device="cpu").reshape(-1).tolist()
        if len(tops) == 1:
            tops *= batch
        if len(lefts) == 1:
            lefts *= batch
        if len(tops) != batch or len(lefts) != batch:
            raise ValueError("top and left must be scalar or one location per batch element")
        source = self.variance_scale_map
        crops = []
        for row, column in zip(tops, lefts):
            if row < 0 or column < 0 or row + height > source.shape[1] or column + width > source.shape[2]:
                raise ValueError("observation crop is outside the fitted variance map")
            crops.append(source[:, int(row):int(row) + height, int(column):int(column) + width])
        return torch.stack(crops).to(device=self.gain_dn_per_electron.device)

    def _variance(
        self, signal_dn: torch.Tensor, top: int | torch.Tensor, left: int | torch.Tensor
    ) -> torch.Tensor:
        scale = self._crop_scale(top, left, signal_dn.shape[0], signal_dn.shape[-2], signal_dn.shape[-1])
        return torch.clamp(
            self._channel_view(self.gain_dn_per_electron) * torch.clamp(signal_dn, min=0.0)
            + self._channel_view(self.read_variance_dn2)
            + 1.0 / 12.0,
            min=1e-6,
        ) * scale

    def _power_for(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        power = self.normalized_power.to(device=device, dtype=torch.float32)
        if power.shape[-2:] != (height, width):
            shifted = torch.fft.fftshift(power, dim=(-2, -1))[None]
            shifted = functional.interpolate(shifted, size=(height, width), mode="bilinear", align_corners=False)
            power = torch.fft.ifftshift(shifted[0], dim=(-2, -1))
        # The DFT of a unit-variance white field has E|F|² = H*W.
        return torch.clamp(power * float(height * width), min=1e-6)

    def limit_signal(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> torch.Tensor:
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
        maximum = torch.minimum(saturation_limit, self._channel_view(self.supported_signal_max_dn))
        return torch.minimum(torch.clamp(signal_dn, min=0.0), maximum)

    def _sample_standardized(self, shape: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        batch, channels, height, width = shape
        if channels != 4:
            raise ValueError("MIS20S1 observations require four packed Bayer channels")
        white = torch.randn(shape, device=device)
        spectrum = torch.fft.fft2(white, dim=(-2, -1))
        power = self._power_for(height, width, device)[None]
        sample = torch.fft.ifft2(
            spectrum * torch.sqrt(power / float(height * width)), dim=(-2, -1)
        ).real
        # Frame-wide channel bias is profiled out in the likelihood.  Do not
        # invent a stationary distribution for it from a short static clip.
        return sample - sample.mean(dim=(-2, -1), keepdim=True)

    def sample(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        illuminated_quantization_step_dn: float = 1.0,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("illuminated company likelihood validates a 1-DN ADC step")
        signal = self.limit_signal(signal_dn, top, left, saturation_margin_dn, saturation_safety_sigma)
        noise = self._sample_standardized(tuple(signal.shape), signal.device) * torch.sqrt(
            self._variance(signal, top, left)
        )
        observed = torch.round(signal + self.black_level_dn + noise)
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
        """Colored residual log density, normalized per RAW pixel for DPS."""
        self._validate(observation_dn)
        self._validate(signal_dn)
        if observation_dn.shape != signal_dn.shape:
            raise ValueError("observation_dn and signal_dn must have equal shapes")
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("illuminated company likelihood validates a 1-DN ADC step")
        _, _, height, width = signal_dn.shape
        standardized = (
            observation_dn - self.black_level_dn - torch.clamp(signal_dn, min=0.0)
        ) / torch.sqrt(self._variance(signal_dn, top, left))
        standardized = standardized - standardized.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fft2(standardized, dim=(-2, -1))
        power = self._power_for(height, width, signal_dn.device)[None]
        quadratic = (spectrum.real.square() + spectrum.imag.square()) / power
        # Per-pixel normalization keeps the guidance magnitude comparable to
        # the existing per-pixel quantized likelihood.
        likelihood = -0.5 * (
            quadratic.mean(dim=(-2, -1))
            + torch.log(power).mean(dim=(-2, -1))
            + torch.log(self._variance(signal_dn, top, left)).mean(dim=(-2, -1))
        )
        return likelihood[:, :, None, None].expand_as(signal_dn)

    def noise_std(self, signal_dn: torch.Tensor) -> torch.Tensor:
        self._validate(signal_dn)
        # The profile bank calls this without a crop.  The median scale gives
        # a conservative scalar prior-only starting noise estimate.
        variance = torch.clamp(
            self._channel_view(self.gain_dn_per_electron) * torch.clamp(signal_dn, min=0.0)
            + self._channel_view(self.read_variance_dn2) + 1.0 / 12.0, min=1e-6
        ) * torch.median(self.variance_scale_map)
        power = self._power_for(signal_dn.shape[-2], signal_dn.shape[-1], signal_dn.device)
        return torch.sqrt(torch.median(variance) * torch.mean(power)) / self.white_level

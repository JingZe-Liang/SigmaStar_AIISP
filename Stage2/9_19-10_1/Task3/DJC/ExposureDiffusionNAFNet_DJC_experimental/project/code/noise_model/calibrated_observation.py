"""Torch observation model loaded from the frame-held-out MIS20S1 fit.

It has two distinct uses.  ``sample`` creates paired development examples
from clean content.  ``log_prob`` is the marginal likelihood used by R1.
Both use signal DN (black-subtracted) as their latent clean variable and add
the fitted per-pixel dark offset only at observation time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .quantized_likelihood import quantized_poisson_gaussian_log_prob


@dataclass
class CalibratedObservationModel:
    gain: int
    gain_dn_per_electron: float
    fixed_map_dn: torch.Tensor  # [4,H,W]
    mixture_weights: torch.Tensor  # [4,M]
    mixture_means_dn: torch.Tensor  # [4,M]
    mixture_sigmas_dn: torch.Tensor  # [4,M]
    row_cov: torch.Tensor  # [4,4]
    col_cov: torch.Tensor  # [4,4]
    frame_offsets_dn: torch.Tensor  # [N,4]
    white_level: float = 4095.0

    @classmethod
    def load(
        cls,
        calibration_json: str | Path,
        gain: int,
        device: torch.device | str = "cpu",
    ) -> "CalibratedObservationModel":
        report = json.loads(Path(calibration_json).read_text())
        info = report["gains"].get(str(gain))
        if info is None:
            raise KeyError(f"gain {gain} does not exist in {calibration_json}")
        calibration_path = Path(calibration_json)
        model_path = Path(info["model_file"])
        # Calibration reports are also deployed with their adjacent ``.npz``
        # files.  A report produced on another machine may retain that
        # machine's absolute path, so resolve a missing file by its basename
        # beside the report before failing.
        if not model_path.is_file():
            colocated_path = calibration_path.parent / model_path.name
            if colocated_path.is_file():
                model_path = colocated_path
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        with np.load(model_path) as archive:
            shot_gain = float(archive["gain_dn"])
            if not np.isfinite(shot_gain) or shot_gain <= 0:
                shot_gain = float(info.get("shot_gain_dn", float("nan")))
            if not np.isfinite(shot_gain) or shot_gain <= 0:
                raise ValueError(f"gain {gain} lacks a positive DN/electron estimate")
            return cls(
                gain=int(gain),
                gain_dn_per_electron=shot_gain,
                fixed_map_dn=torch.from_numpy(archive["fixed_map"].astype(np.float32)).to(device),
                mixture_weights=torch.from_numpy(archive["mixture_weights"].astype(np.float32)).to(device),
                mixture_means_dn=torch.from_numpy(archive["mixture_means_dn"].astype(np.float32)).to(device),
                mixture_sigmas_dn=torch.from_numpy(archive["mixture_sigmas_dn"].astype(np.float32)).to(device),
                row_cov=torch.from_numpy(archive["row_cov"].astype(np.float32)).to(device),
                col_cov=torch.from_numpy(archive["col_cov"].astype(np.float32)).to(device),
                frame_offsets_dn=torch.from_numpy(archive["frame_offsets"].astype(np.float32)).to(device),
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.fixed_map_dn.shape)

    @property
    def pixel_mixture_sigmas_dn(self) -> torch.Tensor:
        """Undo the shared-component marginalisation used by the likelihood.

        ``fit_quantized_likelihood`` stores mixture sigmas after adding frame,
        row and column variance so a per-pixel likelihood can marginalize those
        latent variables. The sampler draws those variables explicitly, so it
        must use only the remaining pixel variance or it counts them twice.
        """
        shared_variance = (
            torch.diagonal(self.row_cov)
            + torch.diagonal(self.col_cov)
            + self.frame_offsets_dn.var(dim=0, unbiased=True)
        )
        return torch.sqrt(
            torch.clamp(self.mixture_sigmas_dn.square() - shared_variance[:, None], min=1e-6)
        )

    def _crop_map(self, top: int, left: int, height: int, width: int) -> torch.Tensor:
        if top < 0 or left < 0 or top + height > self.shape[1] or left + width > self.shape[2]:
            raise ValueError("observation crop is outside calibrated fixed map")
        return self.fixed_map_dn[:, top : top + height, left : left + width]

    def _crop_maps(
        self,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        tops = torch.as_tensor(top, device="cpu").reshape(-1).tolist()
        lefts = torch.as_tensor(left, device="cpu").reshape(-1).tolist()
        if len(tops) == 1:
            tops *= batch
        if len(lefts) == 1:
            lefts *= batch
        if len(tops) != batch or len(lefts) != batch:
            raise ValueError("top and left must be scalar or one location per batch element")
        return torch.stack([
            self._crop_map(int(row), int(column), height, width)
            for row, column in zip(tops, lefts)
        ])

    @staticmethod
    def _sample_joint(covariance: torch.Tensor, count: int, device: torch.device) -> torch.Tensor:
        """Sample [count,4] covariance with PSD and numerical safeguards."""
        covariance = covariance.to(device=device, dtype=torch.float32)
        values, vectors = torch.linalg.eigh(covariance)
        transform = vectors @ torch.diag(torch.sqrt(torch.clamp(values, min=0.0)))
        return torch.randn(count, 4, device=device) @ transform.T

    def _sample_readout(self, shape: tuple[int, int, int, int]) -> torch.Tensor:
        batch, channels, height, width = shape
        if channels != 4:
            raise ValueError("calibrated MIS20S1 model expects four packed RAW channels")
        device = self.fixed_map_dn.device
        output = torch.empty(shape, device=device, dtype=torch.float32)
        pixel_sigmas = self.pixel_mixture_sigmas_dn
        for channel in range(channels):
            weights = self.mixture_weights[channel]
            choices = torch.multinomial(weights.expand(batch * height * width, -1), 1).reshape(
                batch, height, width
            )
            mean = self.mixture_means_dn[channel][choices]
            sigma = pixel_sigmas[channel][choices]
            output[:, channel] = mean + sigma * torch.randn_like(mean)
        frame_indices = torch.randint(len(self.frame_offsets_dn), (batch,), device=device)
        output += self.frame_offsets_dn[frame_indices, :, None, None]
        rows = self._sample_joint(self.row_cov, batch * height, device).reshape(batch, height, 4)
        cols = self._sample_joint(self.col_cov, batch * width, device).reshape(batch, width, 4)
        output += rows.permute(0, 2, 1)[:, :, :, None]
        output += cols.permute(0, 2, 1)[:, :, None, :]
        return output

    def sample(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        illuminated_quantization_step_dn: float = 1.0,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw RAW observation and return its saturation mask.

        The measured 2-DN dark lattice is intentionally *not* imposed here:
        this path models illuminated q=1 data until an illuminated ADC study
        proves a different transfer curve.
        """
        if signal_dn.ndim != 4 or signal_dn.shape[1] != 4:
            raise ValueError("signal_dn must be [B,4,H,W]")
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("R1 illuminated synthesis currently validates q=1 only")
        signal = self.limit_signal(
            signal_dn, top, left, saturation_margin_dn, saturation_safety_sigma
        )
        batch, _, height, width = signal.shape
        offset = self._crop_maps(top, left, batch, height, width)
        shot = torch.poisson(signal / self.gain_dn_per_electron) * self.gain_dn_per_electron
        observed = shot + offset + self._sample_readout(tuple(signal.shape))
        observed = torch.round(observed)
        saturated = observed >= self.white_level
        return torch.clamp(observed, 0.0, self.white_level), saturated

    def limit_signal(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> torch.Tensor:
        """Cap clean signal where a calibrated raw observation would saturate."""
        if saturation_margin_dn < 0 or saturation_safety_sigma < 0:
            raise ValueError("saturation margin and safety sigma must be nonnegative")
        batch, _, height, width = signal_dn.shape
        offset = self._crop_maps(top, left, batch, height, width)
        headroom = torch.clamp(
            self.white_level - offset - saturation_margin_dn, min=0.0
        )
        # Solve x + z*sqrt(K*x + V_read) = headroom for x.  A static DN
        # margin fails at high gain because shot-noise standard deviation
        # grows with signal; this cap makes saturation a documented tail event.
        weights = self.mixture_weights / self.mixture_weights.sum(dim=1, keepdim=True)
        mean = torch.sum(weights * self.mixture_means_dn, dim=1)
        read_variance = torch.sum(
            weights * (self.mixture_sigmas_dn.square() + self.mixture_means_dn.square()), dim=1
        ) - mean.square()
        z = float(saturation_safety_sigma)
        k = float(self.gain_dn_per_electron)
        coefficient = 2.0 * headroom + z * z * k
        constant = headroom.square() - z * z * read_variance[None, :, None, None]
        discriminant = torch.clamp(coefficient.square() - 4.0 * constant, min=0.0)
        max_signal = torch.clamp((coefficient - torch.sqrt(discriminant)) / 2.0, min=0.0)
        return torch.minimum(torch.clamp(signal_dn, min=0.0), max_signal)

    def log_prob(
        self,
        observation_dn: torch.Tensor,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        illuminated_quantization_step_dn: float = 1.0,
    ) -> torch.Tensor:
        """Per-pixel calibrated marginal log probability for R1 DPS guidance."""
        if observation_dn.shape != signal_dn.shape or observation_dn.ndim != 4:
            raise ValueError("observation_dn and signal_dn must have equal [B,4,H,W] shape")
        if illuminated_quantization_step_dn != 1.0:
            raise ValueError("R1 illuminated likelihood currently validates q=1 only")
        _, channels, height, width = signal_dn.shape
        if channels != 4:
            raise ValueError("calibrated MIS20S1 model expects four packed RAW channels")
        offset = self._crop_maps(top, left, observation_dn.shape[0], height, width)
        terms: list[torch.Tensor] = []
        for channel in range(4):
            terms.append(
                quantized_poisson_gaussian_log_prob(
                    observation_dn[:, channel],
                    signal_dn[:, channel],
                    self.gain_dn_per_electron,
                    self.mixture_weights[channel],
                    self.mixture_means_dn[channel],
                    self.mixture_sigmas_dn[channel],
                    offset[:, channel],
                    quantization_step_dn=illuminated_quantization_step_dn,
                )
            )
        return torch.stack(terms, dim=1)

    def observation_to_signal(
        self,
        observation_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        clip_min: float | None = 0.0,
    ) -> torch.Tensor:
        """Subtract only the fixed dark map for network-facing diagnostic input."""
        batch, _, height, width = observation_dn.shape
        signal = observation_dn - self._crop_maps(top, left, batch, height, width)
        return signal if clip_min is None else signal.clamp_min(clip_min)

    def noise_std(self, signal_dn: torch.Tensor) -> torch.Tensor:
        """Return a scalar normalized noise level for the prior-only baseline."""
        if signal_dn.ndim != 4 or signal_dn.shape[1] != 4:
            raise ValueError("signal_dn must have shape [B,4,H,W]")
        weights = self.mixture_weights / self.mixture_weights.sum(dim=1, keepdim=True)
        mean = torch.sum(weights * self.mixture_means_dn, dim=1)
        read_variance = torch.sum(
            weights * (self.mixture_sigmas_dn.square() + self.mixture_means_dn.square()), dim=1
        ) - mean.square()
        variance = self.gain_dn_per_electron * torch.clamp(signal_dn, min=0.0) + read_variance.reshape(1, 4, 1, 1)
        return torch.sqrt(torch.median(variance)) / self.white_level

"""CVPR 2026 2-Shots-in-the-Dark spectral dark-frame synthesis.

Lu et al., CVPR 2026, Sec. 3.1, Eqs. (1)-(13). Arrays are packed Bayer
channels in sensor DN; the implementation uses scipy's paired FFT/IFFT
normalization, so no additional inverse-transform factor is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter


@dataclass
class SpectralDarkModel:
    shading: np.ndarray
    residual_mean: np.ndarray
    reference_residual: np.ndarray
    magnitude: np.ndarray
    reference_phase: np.ndarray
    sigma: float = 50.0
    iterations: int = 10

    @classmethod
    def fit(cls, reference: np.ndarray, sigma: float = 50.0, iterations: int = 10):
        if reference.ndim != 3 or reference.shape[0] != 4:
            raise ValueError("reference must be packed Bayer [4,H,W]")
        if not np.isfinite(reference).all() or sigma <= 0 or iterations < 0:
            raise ValueError("invalid reference, sigma, or iterations")
        reference = np.asarray(reference, dtype=np.float32)
        # The paper/author demonstration applies sigma in packed-pixel units.
        shading = np.stack([
            gaussian_filter(channel, sigma=sigma, mode="reflect", truncate=3.0)
            for channel in reference
        ])
        residual = reference - shading
        residual_mean = residual.mean(axis=(-2, -1), keepdims=True)
        residual = residual - residual_mean
        spectrum = fft2(residual, axes=(-2, -1), workers=1)
        return cls(shading, residual_mean, residual, np.abs(spectrum),
                   np.angle(spectrum), float(sigma), int(iterations))

    def sample(self, seed: int | None = None, iterations: int | None = None) -> np.ndarray:
        """Sample a new dark frame in DN, preserving channel phase offsets."""
        count = self.iterations if iterations is None else iterations
        if count < 0:
            raise ValueError("iterations must be nonnegative")
        rng = np.random.default_rng(seed)
        shared_white = rng.standard_normal(self.shading.shape[-2:], dtype=np.float32)
        phase_offset = np.angle(fft2(shared_white))
        phase = self.reference_phase + phase_offset[None]
        sample = ifft2(self.magnitude * np.exp(1j * phase),
                       axes=(-2, -1)).real.astype(np.float32)
        # Each rank mapping exactly matches the reference marginal histogram.
        targets = np.sort(self.reference_residual.reshape(4, -1), axis=1)
        for _ in range(count):
            aligned = np.empty_like(sample)
            for channel in range(4):
                order = np.argsort(sample[channel].ravel(), kind="stable")
                aligned[channel].ravel()[order] = targets[channel]
            mean = aligned.mean(axis=(-2, -1), keepdims=True)
            spectrum = fft2(aligned - mean, axes=(-2, -1), workers=1)
            unit = spectrum / np.maximum(np.abs(spectrum), 1e-20)
            sample = ifft2(self.magnitude * unit, axes=(-2, -1),
                           workers=1).real.astype(np.float32) + mean
        return sample + self.shading + self.residual_mean


def load_calibrated_model(path: str | Path) -> SpectralDarkModel:
    """Load a portable per-gain reference frame and reconstruct its prior."""
    with np.load(path) as archive:
        reference = archive["reference_dark_dn"].astype(np.float32)
        sigma = float(archive["sigma"])
        iterations = int(archive["iterations"])
    return SpectralDarkModel.fit(reference, sigma=sigma, iterations=iterations)


def synthesize_observation(signal_dn: np.ndarray, model: SpectralDarkModel,
                           shot_gain_dn: float, seed: int) -> np.ndarray:
    """Eq. (15): g*Poisson(x/g) + sampled dark; x is black-subtracted DN."""
    if shot_gain_dn <= 0 or not np.isfinite(shot_gain_dn):
        raise ValueError("shot_gain_dn must be finite and positive")
    if signal_dn.shape != model.shading.shape or np.any(signal_dn < 0):
        raise ValueError("signal_dn must be nonnegative [4,H,W]")
    rng = np.random.default_rng(seed)
    shot = rng.poisson(np.asarray(signal_dn) / shot_gain_dn).astype(np.float32)
    return shot_gain_dn * shot + model.sample(seed=seed + 1)

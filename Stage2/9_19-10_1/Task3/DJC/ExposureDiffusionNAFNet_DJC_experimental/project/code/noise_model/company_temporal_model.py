"""Signal-dependent multi-frame residual model for static company RAW clips.

It combines the repeated-frame photon-transfer estimator of Lu et al.
(CVPR 2026) with the frame/row/column/pixel residual decomposition used by
PNNP (Feng et al., TPAMI 2026).  It intentionally models only *temporal*
noise: a static scene mean cannot identify sensor FPN independently of scene
texture, so that mean is never stored as a fixed noise pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pnnp_decomposition import DecomposedDarkModel


@dataclass
class CompanyTemporalNoiseModel:
    residual_model: DecomposedDarkModel
    variance_slope_dn: np.ndarray  # [4]
    variance_intercept_dn2: np.ndarray  # [4]
    black_level_dn: float
    fit_frames: int
    channel_scale: np.ndarray

    @staticmethod
    def sigma(signal_dn: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        slope = np.asarray(slope, dtype=np.float32).reshape(4, 1, 1)
        intercept = np.asarray(intercept, dtype=np.float32).reshape(4, 1, 1)
        return np.sqrt(np.maximum(intercept + slope * np.maximum(signal_dn, 0.0), 1e-6))

    @classmethod
    def fit(
        cls,
        frames_dn: np.ndarray,
        black_level_dn: float,
        variance_slope_dn: np.ndarray,
        variance_intercept_dn2: np.ndarray,
        seed: int,
    ) -> tuple["CompanyTemporalNoiseModel", np.ndarray]:
        if frames_dn.ndim != 4 or frames_dn.shape[1] != 4 or len(frames_dn) < 3:
            raise ValueError("frames_dn must be [N>=3,4,H,W]")
        slope = np.asarray(variance_slope_dn, dtype=np.float32)
        intercept = np.asarray(variance_intercept_dn2, dtype=np.float32)
        if slope.shape != (4,) or intercept.shape != (4,) or np.any(slope <= 0) or not np.isfinite(intercept).all():
            raise ValueError("four positive variance slopes and finite intercepts are required")
        mean = frames_dn.mean(axis=0, dtype=np.float64).astype(np.float32)
        signal = np.clip(mean - black_level_dn, 0.0, 4095.0)
        sigma = cls.sigma(signal, slope, intercept)
        standardized = (frames_dn - mean[None]) / sigma[None]
        residual = DecomposedDarkModel.fit(standardized, seed=seed)
        # Never use the static company scene mean as sensor fixed pattern.
        residual.fixed_map.fill(0.0)
        # PNNP's spectral-marginal projection can attenuate total variance.
        # Calibrate that deterministic sampler effect on fit frames only.
        observed_std = standardized.std(axis=(0, 2, 3), dtype=np.float64)
        sampled = np.stack([
            residual.sample(
                seed=seed + 10_000 + index,
                adc_quantize=False,
                residual_mode="spectral_marginal",
                iterations=10,
            )
            for index in range(3)
        ])
        sampled_std = sampled.std(axis=(0, 2, 3), dtype=np.float64)
        channel_scale = (observed_std / np.maximum(sampled_std, 1e-6)).astype(np.float32)
        return cls(
            residual, slope, intercept,
            black_level_dn, len(frames_dn), channel_scale,
        ), signal

    def sample(
        self,
        signal_dn: np.ndarray,
        seed: int,
        residual_mode: str = "spectral_marginal",
        iterations: int = 10,
    ) -> np.ndarray:
        signal = np.asarray(signal_dn, dtype=np.float32)
        if signal.shape != self.residual_model.fixed_map.shape:
            raise ValueError("signal_dn must match the fitted packed RAW shape")
        if not np.isfinite(signal).all() or np.any(signal < 0):
            raise ValueError("signal_dn must be finite and nonnegative")
        standardized = self.residual_model.sample(
            seed=seed, adc_quantize=False, residual_mode=residual_mode, iterations=iterations
        )
        standardized *= self.channel_scale[:, None, None]
        noisy = signal + standardized * self.sigma(
            signal, self.variance_slope_dn, self.variance_intercept_dn2
        )
        # The 2 DN code lattice is only established for dark frames. Company
        # clips are illuminated, so retain integral DN but do not force parity.
        return np.clip(np.rint(noisy), 0.0, 4095.0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.residual_model
        np.savez_compressed(
            path,
            fixed_map=model.fixed_map,
            frame_offsets=model.frame_offsets,
            row_cov=model.row_cov,
            col_cov=model.col_cov,
            pixel_pool=model.pixel_pool,
            fit_frames=np.int32(self.fit_frames),
            spectral_magnitude=model.spectral_magnitude,
            reference_phase=model.reference_phase,
            fpn_shrink=model.fpn_shrink,
            variance_slope_dn=self.variance_slope_dn,
            variance_intercept_dn2=self.variance_intercept_dn2,
            black_level_dn=np.float32(self.black_level_dn),
            channel_scale=self.channel_scale,
            illuminated_quantization_step_dn=np.int32(1),
        )

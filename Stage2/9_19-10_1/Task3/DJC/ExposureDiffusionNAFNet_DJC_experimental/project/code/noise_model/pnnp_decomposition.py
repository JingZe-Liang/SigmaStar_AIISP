"""Multi-frame dark-noise decomposition inspired by PNNP (TPAMI 2026).

Implements the paper's physics-guided separation into stable dark shading,
row/column band noise, and pixel noise.  The empirical frame offset and
pixel-residual bootstrap are documented project adaptations; this is not the
paper's unpublished neural proxy or differentiable distribution loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter


def _positive_covariance(values: np.ndarray) -> np.ndarray:
    """Estimate four-channel covariance and clip rounding-level negatives."""
    cov = np.cov(values.reshape(-1, 4), rowvar=False).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return ((eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T).astype(
        np.float32
    )


@dataclass
class DecomposedDarkModel:
    fixed_map: np.ndarray  # [4,H,W] in DN
    frame_offsets: np.ndarray  # [N,4], empirical joint offset vectors
    row_cov: np.ndarray  # [4,4]
    col_cov: np.ndarray  # [4,4]
    pixel_pool: np.ndarray  # [4,M] zero-mean residual values
    fit_frames: int
    spectral_magnitude: np.ndarray | None = None  # stochastic residual power prior
    reference_phase: np.ndarray | None = None
    fpn_shrink: np.ndarray | None = None  # empirical-Bayes correction for finite dark stacks
    adc_step_dn: int = 1
    adc_offset_dn: int = 0
    adc_offsets_per_frame: np.ndarray | None = None

    @classmethod
    def fit(cls, frames: np.ndarray, seed: int = 2026,
            pool_per_frame: int = 16384) -> "DecomposedDarkModel":
        """Fit from training-only packed Bayer frames [N,4,H,W]."""
        if frames.ndim != 4 or frames.shape[1] != 4 or len(frames) < 3:
            raise ValueError("frames must be [N>=3,4,H,W]")
        if not np.isfinite(frames).all() or pool_per_frame <= 0:
            raise ValueError("nonfinite frames or invalid pool size")
        n, _, height, width = frames.shape
        frames = np.asarray(frames, dtype=np.float32)
        adc_sample = frames[:, :, ::16, ::16]
        parity_by_frame = []
        if np.array_equal(adc_sample, np.rint(adc_sample)):
            for t in range(n):
                residues = np.unique(adc_sample[t].astype(np.int32) % 2)
                if len(residues) != 1:
                    break
                parity_by_frame.append(int(residues[0]))
        adc_step = 2 if len(parity_by_frame) == n else 1
        adc_offsets = np.array(parity_by_frame if adc_step == 2 else [0] * n,
                               dtype=np.int32)
        adc_offset = int(adc_offsets[0])
        mean_map = frames.mean(axis=0, dtype=np.float64).astype(np.float32)
        low_frequency = np.stack([
            gaussian_filter(channel, sigma=50, mode="reflect", truncate=3.0)
            for channel in mean_map
        ])
        high_frequency = mean_map - low_frequency
        observed_high_var = high_frequency.var(axis=(1, 2), dtype=np.float64)
        temporal_var = frames.var(axis=0, ddof=1, dtype=np.float64).mean(axis=(1, 2))
        random_mean_var = temporal_var / n
        # A finite N-frame mean contains stochastic noise of variance σ²/N.
        # Shrink only the high-frequency map that exceeds this noise floor.
        shrink = np.clip(1 - random_mean_var /
                         np.maximum(observed_high_var, 1e-20), 0, 1).astype(np.float32)
        fixed = low_frequency + shrink[:, None, None] * high_frequency
        offset = frames.mean(axis=(2, 3), dtype=np.float64) - fixed.mean(
            axis=(1, 2), dtype=np.float64
        )[None]
        offset = offset.astype(np.float32)
        # Use the full training mean only to center stochastic residuals.
        # The shrunken map remains the deterministic synthesis component;
        # including mean_map-fixed here would randomize a persistent pattern.
        residual = frames - mean_map[None] - offset[:, :, None, None]
        row = residual.mean(axis=3)  # [N,4,H]
        residual -= row[:, :, :, None]
        col = residual.mean(axis=2)  # [N,4,W]
        residual -= col[:, :, None, :]
        # The per-pixel training mean was estimated from these N frames.
        # Correct its (N-1)/N attenuation of random residual variance.
        correction = np.sqrt(n / (n - 1))
        residual *= correction
        row_cov = _positive_covariance(row.transpose(0, 2, 1)) * (n / (n - 1))
        col_cov = _positive_covariance(col.transpose(0, 2, 1)) * (n / (n - 1))
        rng = np.random.default_rng(seed)
        pool = np.empty((4, n * pool_per_frame), dtype=np.float32)
        for t in range(n):
            indices = rng.integers(height * width, size=pool_per_frame)
            pool[:, t * pool_per_frame:(t + 1) * pool_per_frame] = (
                residual[t].reshape(4, -1)[:, indices]
            )
        pool -= pool.mean(axis=1, keepdims=True)
        power = np.zeros((4, height, width), dtype=np.float64)
        phase = None
        for t in range(n):
            spectrum = fft2(residual[t], axes=(-2, -1), workers=1)
            power += np.abs(spectrum).astype(np.float64)**2
            if t == 0:
                phase = np.angle(spectrum).astype(np.float32)
        magnitude = np.sqrt(power / n).astype(np.float32)
        return cls(fixed, offset, row_cov, col_cov, pool, n,
                   magnitude, phase, shrink, adc_step, adc_offset, adc_offsets)

    def quantize(self, frame: np.ndarray, offset_dn: int | None = None) -> np.ndarray:
        """Project into the RAW code lattice observed in training frames."""
        offset_dn = self.adc_offset_dn if offset_dn is None else int(offset_dn)
        output = (self.adc_step_dn * np.rint(
            (frame - offset_dn) / self.adc_step_dn
        ) + offset_dn).astype(np.float32)
        np.clip(output, offset_dn,
                4095 - ((4095 - offset_dn) % self.adc_step_dn),
                out=output)
        return output

    def _spectral_pixel_residual(self, rng: np.random.Generator,
                                 iterations: int,
                                 final_marginal: bool = False) -> np.ndarray:
        """CVPR 2026 spectral/IHM sampler on PNNP-decoupled residuals.

        The averaged spectral power and pooled marginal are multi-frame
        adaptations; the alternating constraints follow Lu et al. Sec. 3.1.
        """
        if self.spectral_magnitude is None or self.reference_phase is None:
            raise ValueError("spectral prior absent from this calibration")
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        _, height, width = self.fixed_map.shape
        white_phase = np.angle(fft2(rng.standard_normal(
            (height, width), dtype=np.float32)))
        phase = self.reference_phase + white_phase[None]
        sample = ifft2(self.spectral_magnitude * np.exp(1j * phase),
                       axes=(-2, -1)).real.astype(np.float32)
        count = height * width
        sorted_pool = np.sort(self.pixel_pool, axis=1)
        target_indices = np.minimum(
            ((np.arange(count) + 0.5) * sorted_pool.shape[1] / count).astype(int),
            sorted_pool.shape[1] - 1,
        )
        targets = sorted_pool[:, target_indices]
        for _ in range(iterations):
            aligned = np.empty_like(sample)
            for channel in range(4):
                order = np.argsort(sample[channel].ravel(), kind="stable")
                aligned[channel].ravel()[order] = targets[channel]
            mean = aligned.mean(axis=(1, 2), keepdims=True)
            spectrum = fft2(aligned - mean, axes=(-2, -1), workers=1)
            unit = spectrum / np.maximum(np.abs(spectrum), 1e-20)
            sample = ifft2(self.spectral_magnitude * unit,
                           axes=(-2, -1), workers=1).real.astype(np.float32) + mean
        if final_marginal:
            # Project once more onto the pooled marginal. This is an explicit
            # project adaptation: CVPR 2026 ends with spectral correction.
            aligned = np.empty_like(sample)
            for channel in range(4):
                order = np.argsort(sample[channel].ravel(), kind="stable")
                aligned[channel].ravel()[order] = targets[channel]
            return aligned
        return sample

    def _sample_analog(self, seed: int, residual_mode: str,
                       iterations: int) -> tuple[np.ndarray, int]:
        """Draw read noise and its observed ADC state before quantization."""
        rng = np.random.default_rng(seed)
        _, height, width = self.fixed_map.shape
        frame_index = rng.integers(len(self.frame_offsets))
        offset = self.frame_offsets[frame_index]
        row = rng.multivariate_normal(np.zeros(4), self.row_cov,
                                      size=height).T.astype(np.float32)
        col = rng.multivariate_normal(np.zeros(4), self.col_cov,
                                      size=width).T.astype(np.float32)
        output = self.fixed_map.copy()
        output += offset[:, None, None]
        output += row[:, :, None]
        output += col[:, None, :]
        if residual_mode in ("spectral", "spectral_marginal"):
            output += self._spectral_pixel_residual(
                rng, iterations, final_marginal=residual_mode == "spectral_marginal"
            )
        elif residual_mode == "iid":
            for channel in range(4):
                indices = rng.integers(self.pixel_pool.shape[1],
                                       size=(height, width))
                output[channel] += self.pixel_pool[channel, indices]
        else:
            raise ValueError("residual_mode must be 'spectral', 'spectral_marginal', or 'iid'")
        parity = (self.adc_offsets_per_frame[frame_index]
                  if self.adc_offsets_per_frame is not None
                  else self.adc_offset_dn)
        return output, int(parity)

    def sample(self, seed: int, adc_quantize: bool = True,
               residual_mode: str = "spectral", iterations: int = 10) -> np.ndarray:
        """Sample a dark RAW frame, preserving joint four-channel offsets."""
        output, parity = self._sample_analog(seed, residual_mode, iterations)
        return self.quantize(output, offset_dn=parity) if adc_quantize else output

    def sample_observation(self, signal_dn: np.ndarray, shot_gain_dn: float,
                           seed: int, residual_mode: str = "spectral_marginal",
                           iterations: int = 10) -> np.ndarray:
        """Add provisional Poisson shot noise to a calibrated dark sample.

        The dark fit alone cannot identify the shot coefficient. Supply a
        separately calibrated positive g and black-subtracted linear signal.
        The 2-DN dark-frame lattice is not imposed on illuminated pixels;
        their signal-dependent ADC behavior is not calibrated here.
        """
        signal = np.asarray(signal_dn)
        if (signal.shape != self.fixed_map.shape or not np.isfinite(signal).all()
                or np.any(signal < 0)):
            raise ValueError("signal_dn must be finite, nonnegative [4,H,W]")
        if not np.isfinite(shot_gain_dn) or shot_gain_dn <= 0:
            raise ValueError("shot_gain_dn must be finite and positive")
        dark = self.sample(seed, residual_mode=residual_mode,
                           iterations=iterations)
        shot_rng = np.random.default_rng(seed + 1)
        shot = shot_gain_dn * shot_rng.poisson(signal / shot_gain_dn)
        return np.clip(np.rint(dark + shot), 0, 4095).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, fixed_map=self.fixed_map,
                            frame_offsets=self.frame_offsets, row_cov=self.row_cov,
                            col_cov=self.col_cov, pixel_pool=self.pixel_pool,
                            fit_frames=np.int32(self.fit_frames),
                            spectral_magnitude=self.spectral_magnitude,
                            reference_phase=self.reference_phase,
                            fpn_shrink=self.fpn_shrink,
                            adc_step_dn=np.int32(self.adc_step_dn),
                            adc_offset_dn=np.int32(self.adc_offset_dn),
                            adc_offsets_per_frame=self.adc_offsets_per_frame)

    @classmethod
    def load(cls, path: str | Path) -> "DecomposedDarkModel":
        with np.load(path) as archive:
            return cls(archive["fixed_map"], archive["frame_offsets"],
                       archive["row_cov"], archive["col_cov"],
                       archive["pixel_pool"], int(archive["fit_frames"]),
                       archive["spectral_magnitude"], archive["reference_phase"],
                       archive["fpn_shrink"], int(archive["adc_step_dn"]),
                       int(archive["adc_offset_dn"]),
                       archive["adc_offsets_per_frame"])

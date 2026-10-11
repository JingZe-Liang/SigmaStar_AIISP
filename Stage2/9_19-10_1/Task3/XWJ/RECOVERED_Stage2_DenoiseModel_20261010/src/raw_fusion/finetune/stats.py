"""Per-stream measurements: black level, split-half sigma, temporal means, drift.

All computations stream over packed chunks in 12-bit DN so peak memory stays
independent of stream length. Recipes follow the 2026-09-20 survey so manifest
numbers are directly comparable with the survey report.
"""

from dataclasses import dataclass
from itertools import islice
from math import ceil

import numpy as np

from raw_fusion.finetune.adapters import WHITE_CLIP_DN, SourceAdapter
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, pack_frame, pack_frames

# Median of the darkest 5% of pooled 4x4-packed-block means per channel: the
# survey-matched pedestal proxy. It sits above the true sensor black in
# flare-contaminated scenes; the manifest records it as a measurement, the
# adapter's sensor black stays nominal.
_BLACK_LOW_FRACTION = 0.05

# Pixels whose temporal mean exceeds black + this margin count as "signal".
SIGNAL_MARGIN_DN = 15.0
# Below this many signal pixels a stream is considered too dark for signal
# statistics and the whole frame area is used instead (recorded in manifests).
MIN_SIGNAL_PIXELS = 10_000

# sigma_map = |even_mean - odd_mean| * sqrt(T) / SPLIT_HALF_FACTOR, unbiased
# from the MEDIAN of the half-normal by 1/0.6745: for Gaussian noise,
# median|X| = 0.6745 * sigma (the mean would be 0.798 * sigma - do not mix
# them up; the MAD constant in motion.py is the same 1.4826).
_SPLIT_HALF_FACTOR = 2.0
_MEDIAN_TO_SIGMA = 1.4826


@dataclass(slots=True)
class StreamBasicStats:
    """One-sequential-pass measurements for a stream (12-bit DN)."""

    frames_total: int
    frames_used: int
    temporal_mean: np.ndarray  # [4, h, w] float32, packed, black NOT subtracted
    even_mean: np.ndarray  # [4, h, w] float32, mean over even-index frames
    odd_mean: np.ndarray  # [4, h, w] float32, mean over odd-index frames
    frame_means: np.ndarray  # [T] float64 per-frame packed mean
    saturation_fraction: float  # fraction of packed samples at white clip
    black_levels: dict[str, float]  # measured pedestal proxy per canonical channel


def measure_black_levels(reader, adapter: SourceAdapter, sample_frames: int = 16) -> dict[str, float]:
    """Measure the per-channel black pedestal proxy on evenly sampled frames.

    Survey-matched recipes by role: calibration black captures fill the whole
    frame with pedestal + noise, so the median 4x4-packed-block mean is the
    pedestal; scene streams only hold pedestal in their darkest blocks, so the
    median of the darkest 5% of block means is used (block means suppress the
    noise tail that drags per-pixel dark percentiles below the pedestal).
    """

    count = min(sample_frames, reader.frames)
    indices = np.linspace(0, reader.frames - 1, count).round().astype(int)
    pools: dict[str, list[np.ndarray]] = {name: [] for name in CHANNEL_NAMES}
    for index in indices:
        packed = pack_frame(reader.read_frame(int(index)), adapter)
        block_means = _packed_block_means(packed)
        for channel, name in enumerate(CHANNEL_NAMES):
            pools[name].append(block_means[channel].ravel())
    full_frame_pedestal = adapter.role == "calibration_black"
    black: dict[str, float] = {}
    for name in CHANNEL_NAMES:
        pool = np.concatenate(pools[name])
        if full_frame_pedestal:
            black[name] = float(np.median(pool))
            continue
        k = max(1, int(pool.size * _BLACK_LOW_FRACTION))
        darkest = np.partition(pool, k)[:k]
        black[name] = float(np.median(darkest))
    return black


def _packed_block_means(packed: np.ndarray, block: int = 4) -> np.ndarray:
    """4x4-packed-block (8x8 sensor) means, shape [4, h/block, w/block]."""

    channels, height, width = packed.shape
    if height % block != 0 or width % block != 0:
        raise ValueError(f"packed size {height}x{width} not divisible by block {block}")
    return packed.reshape(channels, height // block, block, width // block, block).mean(axis=(2, 4))


def compute_basic_stats(
    reader,
    adapter: SourceAdapter,
    chunk_frames: int = 25,
    max_frames: int | None = None,
    black_sample_frames: int = 16,
) -> StreamBasicStats:
    """Single sequential pass: temporal means, split-half means, frame means,
    saturation fraction. Frame parity for the split-half is the global index."""

    black_levels = measure_black_levels(reader, adapter, sample_frames=black_sample_frames)
    frames_used = reader.frames if max_frames is None else min(reader.frames, max_frames)
    first_chunk = next(reader.iter_chunks(1))
    channels, height, width = pack_frames(first_chunk, adapter).shape[1:]

    mean_acc = np.zeros((channels, height, width), dtype=np.float64)
    even_acc = np.zeros_like(mean_acc)
    odd_acc = np.zeros_like(mean_acc)
    frame_means = np.zeros(frames_used, dtype=np.float64)
    even_count = odd_count = 0
    saturated = total = 0

    consumed = 0
    for chunk in islice(reader.iter_chunks(chunk_frames), ceil(frames_used / chunk_frames)):
        packed = pack_frames(chunk, adapter).astype(np.float64)
        take = min(packed.shape[0], frames_used - consumed)
        packed = packed[:take]
        parity = (np.arange(consumed, consumed + take) % 2).astype(bool)
        mean_acc += packed.sum(axis=0)
        even_acc += packed[~parity].sum(axis=0)
        odd_acc += packed[parity].sum(axis=0)
        even_count += int((~parity).sum())
        odd_count += int(parity.sum())
        frame_means[consumed : consumed + take] = packed.mean(axis=(1, 2, 3))
        saturated += int((packed >= WHITE_CLIP_DN).sum())
        total += packed.size
        consumed += take

    return StreamBasicStats(
        frames_total=reader.frames,
        frames_used=frames_used,
        temporal_mean=(mean_acc / frames_used).astype(np.float32),
        even_mean=(even_acc / even_count).astype(np.float32),
        odd_mean=(odd_acc / odd_count).astype(np.float32),
        frame_means=frame_means,
        saturation_fraction=saturated / total,
        black_levels=black_levels,
    )


def signal_mask(mean: np.ndarray, black: np.ndarray, margin_dn: float = SIGNAL_MARGIN_DN) -> np.ndarray:
    """Boolean signal mask for one packed channel plane."""

    return mean > (black + margin_dn)


def split_half_sigma(
    basic: StreamBasicStats,
    static_region: np.ndarray | None,
) -> dict:
    """Total temporal noise per channel from the even/odd split halves.

    sigma_map = |even_mean - odd_mean| * 1.4826 * sqrt(T) / 2 per pixel
    (median-unbiased for the half-normal); the reported value is the median
    over the population: static & signal pixels when a static region is
    given, signal pixels otherwise, and the whole frame when the scene is too
    dark to define a signal population.

    Note: this is TOTAL temporal noise - it includes scene shimmer and
    illumination flicker, not just sensor noise, so it runs above the survey's
    drift-robust sensor-only sigma on scene streams. The A2 noise model fits
    on calibration captures, which carry no scene shimmer. At the lowest
    calibration gains the read noise is sub-quantization-LSB (stored values
    are dn12 << 4), so there the measured value is a quantization-limited
    floor, not the sensor's true read noise.
    """

    sigma_map = np.abs(basic.even_mean - basic.odd_mean) * (
        _MEDIAN_TO_SIGMA * np.sqrt(basic.frames_used) / _SPLIT_HALF_FACTOR
    )
    per_channel: dict[str, float] = {}
    population = "static+signal"
    for channel, name in enumerate(CHANNEL_NAMES):
        selected = signal_mask(basic.temporal_mean[channel], basic.black_levels[name])
        if static_region is not None:
            selected = static_region & selected
        else:
            population = "signal"
        if int(selected.sum()) < MIN_SIGNAL_PIXELS:
            selected = np.ones_like(selected)
            population = "all"
        per_channel[name] = float(np.median(sigma_map[channel][selected]))
    return {"per_channel": per_channel, "population": population}


def frame_drift_stats(frame_means: np.ndarray) -> dict[str, float]:
    """Exposure drift of the stream: coefficient of variation of the frame mean
    series plus the largest absolute deviation from the median frame."""

    median = float(np.median(frame_means))
    return {
        "frame_mean_cv": float(np.std(frame_means) / np.mean(frame_means)),
        "frame_mean_max_dev": float(np.max(np.abs(frame_means - median)) / median),
    }

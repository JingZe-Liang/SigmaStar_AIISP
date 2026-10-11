"""Motion mask: per-frame deviation from the temporal mean, adaptively thresholded.

A supervision frame pixel is unusable when its content disagrees with the
long-term average (people crossing, foliage, fan) - that disagreement is
exactly what ghosts the temporal mean used as the supervision target. The
mask therefore flags pixels whose deviation from the stream's temporal mean
exceeds a per-frame adaptive threshold (median + k * 1.4826 * MAD over signal
pixels). Global flicker/drift partially survives as deviation; that is
intended, since drift also contaminates the target.

Two quantities come out of one mask series:
- masks [T, 1, H, W]: per-frame motion, for input-side loss exclusion;
- long-term static region: pixels static in >= 90% of frames, for reference
  validity - comparable to the survey's "69-84% of bright area temporally
  static" numbers.
"""

from dataclasses import dataclass
from itertools import islice
from math import ceil

import numpy as np

from raw_fusion.finetune.adapters import SourceAdapter
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, pack_frames
from raw_fusion.finetune.stats import MIN_SIGNAL_PIXELS, SIGNAL_MARGIN_DN, StreamBasicStats

_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class MotionMaskParams:
    k_mad: float = 3.0  # threshold = median + k_mad * 1.4826 * MAD (per frame)
    signal_margin_dn: float = SIGNAL_MARGIN_DN  # "signal" pixels for threshold stats
    min_signal_pixels: int = MIN_SIGNAL_PIXELS  # darker scenes threshold on all pixels


@dataclass(slots=True)
class MotionMaskResult:
    masks: np.ndarray  # [T, 1, H, W] float16, 1.0 = motion (excluded from supervision)
    static_fraction_per_frame: np.ndarray  # [T] float64
    threshold_per_frame: np.ndarray  # [T] float64, the adaptive deviation threshold in DN

    @property
    def static_fraction_mean(self) -> float:
        return float(self.static_fraction_per_frame.mean())


def compute_motion_masks(
    reader,
    adapter: SourceAdapter,
    basic: StreamBasicStats,
    params: MotionMaskParams,
    chunk_frames: int = 25,
) -> MotionMaskResult:
    """Compute per-frame motion masks for one stream.

    `basic` supplies the black levels, the temporal mean and the frame count;
    the stream is re-read once chunk by chunk (cheap relative to masking).
    """

    frames_used = basic.frames_used
    temporal_mean = basic.temporal_mean.astype(np.float32).mean(axis=0)  # [h, w]
    black = float(np.mean([basic.black_levels[name] for name in CHANNEL_NAMES]))
    signal = temporal_mean > (black + params.signal_margin_dn)
    if int(signal.sum()) < params.min_signal_pixels:
        signal = np.ones_like(signal)

    masks = np.empty((frames_used, 1, basic.temporal_mean.shape[1] * 2, basic.temporal_mean.shape[2] * 2), dtype=np.float16)
    static_fractions = np.zeros(frames_used, dtype=np.float64)
    thresholds = np.zeros(frames_used, dtype=np.float64)

    consumed = 0
    for chunk in islice(reader.iter_chunks(chunk_frames), ceil(frames_used / chunk_frames)):
        packed = pack_frames(chunk, adapter).astype(np.float32)
        take = min(packed.shape[0], frames_used - consumed)
        deviation = np.abs(packed[:take].mean(axis=1) - temporal_mean[None])  # [n, h, w]
        for local_t in range(take):
            frame_dev = deviation[local_t]
            values = frame_dev[signal]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            threshold = median + params.k_mad * _MAD_TO_SIGMA * mad
            motion = frame_dev > threshold
            global_t = consumed + local_t
            static_fractions[global_t] = 1.0 - float(motion.mean())
            thresholds[global_t] = threshold
            masks[global_t, 0] = np.repeat(np.repeat(motion, 2, axis=0), 2, axis=1).astype(np.float16)
        consumed += take

    return MotionMaskResult(
        masks=masks,
        static_fraction_per_frame=static_fractions,
        threshold_per_frame=thresholds,
    )


def long_term_static_region(result: MotionMaskResult, min_static_fraction: float = 0.9) -> np.ndarray:
    """Packed-resolution boolean region that is static in enough frames to keep
    the temporal mean valid; used as the population for sigma and exposure fits."""

    packed = result.masks[:, 0, ::2, ::2]  # [T, H, W] at packed resolution
    return (1.0 - packed.mean(axis=0)) >= min_static_fraction

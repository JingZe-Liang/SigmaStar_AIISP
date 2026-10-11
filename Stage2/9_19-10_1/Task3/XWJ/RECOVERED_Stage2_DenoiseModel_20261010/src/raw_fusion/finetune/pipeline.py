"""A0 build orchestrator: discover streams, measure, register, fit exposure.

Run through scripts/build_a0_manifests.py. Every stream ends up with a
stream_manifest.json; scene streams additionally get a [T, 1, H, W] motion
mask .npy. Write boundary: only <repo>/artifacts/finetune_mis20s1/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import (
    REGISTRATION_BASELINE,
    calibration_adapter,
    parse_filename_metadata,
    resolve_adapter,
    resolve_exposure_reference,
)
from raw_fusion.finetune.exposure import ExposureFit, fit_exposure_pair
from raw_fusion.finetune.manifest import (
    MotionFacts,
    StreamManifest,
    manifest_name,
    set_exposure_fit,
    set_registration,
)
from raw_fusion.finetune.motion import (
    MotionMaskParams,
    long_term_static_region,
    compute_motion_masks,
)
from raw_fusion.finetune.progress import A0ProgressTracker
from raw_fusion.finetune.registration import (
    RegistrationResult,
    aligned_packed_pair,
    overlap_crops,
    register_streams,
)
from raw_fusion.finetune.stats import (
    StreamBasicStats,
    compute_basic_stats,
    frame_drift_stats,
    split_half_sigma,
)
from raw_fusion.finetune.stream_io import CHANNEL_NAMES, open_stream

_DATASET_ROOT = Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset")
_GROUP_DIRS = {"04": "04_Sigmastar_7_30", "03": "03_MIS20S1_2D3D"}


@dataclass(frozen=True, slots=True)
class StreamJob:
    group: str
    scene: str
    gain_label: str
    variant: str | None
    adapter: object  # SourceAdapter
    stream_path: Path  # raw file, or directory of per-frame raws

    @property
    def label(self) -> str:
        variant = f"/{self.variant}" if self.variant else ""
        return f"{self.group}/{self.scene}/{self.gain_label}{variant}"


@dataclass(slots=True)
class _StreamCache:
    mean: np.ndarray  # [4, h, w] float32 packed temporal mean
    black: dict[str, float]
    static_region: np.ndarray  # [h, w] bool packed long-term static
    registration: RegistrationResult | None


def discover_jobs(dataset_root: Path, groups: tuple[str, ...]) -> list[StreamJob]:
    """Enumerate A0-scope streams for the requested groups (04 and 03 only)."""

    jobs: list[StreamJob] = []
    for group in groups:
        group_dir = dataset_root / _GROUP_DIRS[group]
        if group == "04":
            for gain_dir in sorted((group_dir / "shdarkroom").iterdir()):
                jobs.append(_single_file_job(group_dir, "shdarkroom", gain_dir.name, gain_dir))
            for gain_dir in sorted((group_dir / "mis20s1_outdoor").iterdir()):
                jobs.append(_single_file_job(group_dir, "mis20s1_outdoor", gain_dir.name, gain_dir))
            calib = group_dir / "mis20s1_calibrationdata"
            for kind in ("black", "color"):
                for gain_dir in sorted((calib / kind).iterdir()):
                    adapter = calibration_adapter(kind)
                    jobs.append(
                        StreamJob("04", adapter.scene, gain_dir.name, kind, adapter, _single_raw(gain_dir))
                    )
        elif group == "03":
            for stream_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                gain_label = stream_dir.name.rsplit("_", 1)[1]
                adapter = resolve_adapter("03", "MIS20S1_2D3D", gain_label)
                for variant in ("denoised", "fused"):
                    jobs.append(StreamJob("03", adapter.scene, gain_label, variant, adapter, stream_dir / variant))
        else:
            raise ValueError(f"group {group!r} is Phase C scope; A0 covers 04 and 03")
    return jobs


def _single_file_job(group_dir: Path, scene: str, gain_label: str, gain_dir: Path) -> StreamJob:
    adapter = resolve_adapter("04", scene, gain_label)
    return StreamJob("04", scene, gain_label, None, adapter, _single_raw(gain_dir))


def _single_raw(directory: Path) -> Path:
    raws = sorted(directory.glob("*.raw"))
    if len(raws) != 1:
        raise ValueError(f"{directory}: expected exactly one .raw, found {len(raws)}")
    return raws[0]


def build_a0_artifacts(
    dataset_root: Path,
    out_dir: Path,
    groups: tuple[str, ...] = ("04", "03"),
    chunk_frames: int = 25,
    mask_params: MotionMaskParams | None = None,
    max_frames: int | None = None,
) -> dict:
    """Run the full A0 build. Returns a summary dict of per-stream key facts."""

    mask_params = mask_params or MotionMaskParams()
    supervision_dir = out_dir / "supervision"
    manifest_dir = supervision_dir / "stream_manifests"
    mask_dir = supervision_dir / "motion_masks"
    jobs = discover_jobs(dataset_root, groups)

    register_jobs = [j for j in jobs if j.group == "04" and j.adapter.role in ("reference", "noisy")]
    exposure_jobs = [j for j in jobs if resolve_exposure_reference(j.group, j.scene, j.gain_label)]
    total_units = len(jobs) + len(register_jobs) + len(exposure_jobs)
    tracker = A0ProgressTracker(out_dir, total_units)
    tracker.start()

    manifests: dict[str, StreamManifest] = {}
    caches: dict[str, _StreamCache] = {}
    try:
        for job in jobs:
            tracker.start_unit(job.label, "measure")
            manifests[job.label] = _measure_job(
                job, manifest_dir, mask_dir, mask_params, chunk_frames, max_frames, caches
            )
            tracker.complete_unit()
            _print_progress(tracker, job.label, "measure")

        for job in register_jobs:
            tracker.start_unit(job.label, "register")
            _register_job(job, manifests, caches, manifest_dir)
            tracker.complete_unit()
            _print_progress(tracker, job.label, "register")

        for job in exposure_jobs:
            tracker.start_unit(job.label, "exposure")
            _exposure_job(job, manifests, caches, manifest_dir)
            tracker.complete_unit()
            _print_progress(tracker, job.label, "exposure")

        tracker.complete()
    except BaseException:
        tracker.failed()
        raise

    return {
        "streams": len(jobs),
        "registrations": len(register_jobs),
        "exposure_fits": len(exposure_jobs),
        "manifest_dir": str(manifest_dir),
        "mask_dir": str(mask_dir),
        "labels": [job.label for job in jobs],
    }


def _measure_job(
    job: StreamJob,
    manifest_dir: Path,
    mask_dir: Path,
    mask_params: MotionMaskParams,
    chunk_frames: int,
    max_frames: int | None,
    caches: dict[str, _StreamCache],
) -> StreamManifest:
    reader, meta = open_stream(job.stream_path, job.adapter)
    manifest = StreamManifest.create(job.adapter, job.gain_label, job.variant, job.stream_path, meta)
    basic: StreamBasicStats = compute_basic_stats(reader, job.adapter, chunk_frames, max_frames)
    manifest.frames_total = basic.frames_total
    manifest.frames_used = basic.frames_used
    manifest.black_levels = basic.black_levels
    manifest.saturation_fraction = basic.saturation_fraction
    drift = frame_drift_stats(basic.frame_means)
    manifest.frame_mean_cv = drift["frame_mean_cv"]
    manifest.frame_mean_max_dev = drift["frame_mean_max_dev"]

    needs_mask = job.adapter.role in ("reference", "noisy", "auxiliary")
    static_region = None
    if needs_mask:
        result = compute_motion_masks(reader, job.adapter, basic, mask_params, chunk_frames)
        mask_path = mask_dir / f"{manifest_name(job.group, job.scene, job.gain_label, job.variant)}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, result.masks)
        static_region = long_term_static_region(result)
        signal = np.zeros_like(static_region)
        for channel, name in enumerate(CHANNEL_NAMES):
            signal |= basic.temporal_mean[channel] > (basic.black_levels[name] + mask_params.signal_margin_dn)
        manifest.motion = MotionFacts(
            params={
                "k_mad": mask_params.k_mad,
                "signal_margin_dn": mask_params.signal_margin_dn,
                "min_signal_pixels": mask_params.min_signal_pixels,
            },
            static_fraction_mean=result.static_fraction_mean,
            static_fraction_per_frame=[float(v) for v in result.static_fraction_per_frame],
            long_term_static_signal_fraction=float(static_region[signal].mean()) if signal.any() else 0.0,
            threshold_per_frame_mean_dn=float(result.threshold_per_frame.mean()),
            mask_path=str(mask_path),
        )
        if job.group == "04":
            caches[job.label] = _StreamCache(
                mean=basic.temporal_mean,
                black=basic.black_levels,
                static_region=static_region,
                registration=None,
            )
    manifest.sigma_split_half = split_half_sigma(basic, static_region)
    manifest.write_json(manifest_dir)
    return manifest


def _register_job(
    job: StreamJob,
    manifests: dict[str, StreamManifest],
    caches: dict[str, _StreamCache],
    manifest_dir: Path,
) -> None:
    baseline_label = REGISTRATION_BASELINE[job.scene]
    baseline_job_label = f"{job.group}/{job.scene}/{baseline_label}"
    manifest = manifests[job.label]
    if job.gain_label == baseline_label:
        result = RegistrationResult(
            baseline=baseline_label, shift_x=0.0, shift_y=0.0, integer_shift_x=0, integer_shift_y=0,
            ncc=1.0, paired=True, reason="baseline",
        )
    else:
        baseline = caches[baseline_job_label]
        moving = caches[job.label]
        result = register_streams(
            baseline.mean, baseline.black, moving.mean, moving.black, baseline_label
        )
        caches[job.label].registration = result
    set_registration(manifest, result)
    manifest.write_json(manifest_dir)


def _exposure_job(
    job: StreamJob,
    manifests: dict[str, StreamManifest],
    caches: dict[str, _StreamCache],
    manifest_dir: Path,
) -> None:
    scene, reference_gain = "shdarkroom", "10x"
    reference_label = f"{job.group}/{scene}/{reference_gain}"
    reference = caches[reference_label]
    moving = caches[job.label]

    # Registration shifts were measured against the scene baseline; compose
    # the packed integer displacements to align the stream onto the 10x frame.
    reference_shift = _shift_vs_baseline(reference)
    moving_shift = _shift_vs_baseline(moving)
    shift_x = moving_shift[0] - reference_shift[0]
    shift_y = moving_shift[1] - reference_shift[1]

    ref_crop, mov_crop, y0, x0 = aligned_packed_pair(reference.mean, moving.mean, shift_x, shift_y)
    height, width = ref_crop.shape[1:]
    ref_static = reference.static_region[y0 : y0 + height, x0 : x0 + width]
    mov_static_crop = overlap_crops(reference.static_region, moving.static_region, shift_x, shift_y)[1]
    fit: ExposureFit = fit_exposure_pair(
        mov_crop, moving.black, ref_crop, reference.black, ref_static & mov_static_crop,
        reference_label=reference_gain,
    )
    set_exposure_fit(manifests[job.label], fit)
    manifests[job.label].write_json(manifest_dir)


def _shift_vs_baseline(cache: _StreamCache) -> tuple[int, int]:
    if cache.registration is None:
        return 0, 0
    return cache.registration.integer_shift_x, cache.registration.integer_shift_y


def _print_progress(tracker: A0ProgressTracker, label: str, stage: str) -> None:
    state = tracker.state
    eta = ""
    if state.eta_seconds is not None and state.status == "running":
        eta = f" eta={int(state.eta_seconds // 60):02d}:{int(state.eta_seconds % 60):02d}"
    print(
        f"[a0 {state.completed_units}/{state.total_units}] {stage} {label}{eta}",
        flush=True,
    )

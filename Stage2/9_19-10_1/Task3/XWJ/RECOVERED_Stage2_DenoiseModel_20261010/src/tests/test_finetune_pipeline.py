"""Pipeline-level A0 tests on real files (strictly read-only): motion mask,
exposure fit, manifest end-to-end, and the progress tracker contract."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import resolve_adapter
from raw_fusion.finetune.exposure import fit_exposure_pair
from raw_fusion.finetune.manifest import StreamManifest
from raw_fusion.finetune.motion import MotionMaskParams, compute_motion_masks, long_term_static_region
from raw_fusion.finetune.pipeline import discover_jobs
from raw_fusion.finetune.progress import A0ProgressState, A0ProgressTracker
from raw_fusion.finetune.registration import overlap_crops
from raw_fusion.finetune.stats import compute_basic_stats, split_half_sigma
from raw_fusion.finetune.stream_io import open_stream

DATASET = Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset")
SHDARKROOM = DATASET / "04_Sigmastar_7_30" / "shdarkroom"
MASK_PARAMS = MotionMaskParams()


def _open(gain: str):
    adapter = resolve_adapter("04", "shdarkroom", gain)
    raw = sorted((SHDARKROOM / gain).glob("*.raw"))[0]
    return open_stream(raw, adapter)


def _basic(gain: str, max_frames: int = 48):
    reader, _ = _open(gain)
    adapter = resolve_adapter("04", "shdarkroom", gain)
    return compute_basic_stats(reader, adapter, chunk_frames=12, max_frames=max_frames, black_sample_frames=4)


def test_discover_jobs_a0_scope() -> None:
    jobs = discover_jobs(DATASET, ("04", "03"))
    labels = {job.label for job in jobs}
    assert "04/shdarkroom/10x" in labels and "04/shdarkroom/1000x" in labels
    assert "04/mis20s1_outdoor/1500x" in labels
    assert "04/mis20s1_calibrationdata/800/black" in labels
    assert "03/MIS20S1_2D3D/645x/denoised" in labels and "03/MIS20S1_2D3D/128x/fused" in labels
    assert len([j for j in jobs if j.group == "04"]) == 7 + 4 + 18
    assert len([j for j in jobs if j.group == "03"]) == 4


def test_motion_mask_real_645x(tmp_path: Path) -> None:
    adapter = resolve_adapter("04", "shdarkroom", "645x")
    reader, _ = _open("645x")
    basic = compute_basic_stats(reader, adapter, chunk_frames=12, max_frames=48, black_sample_frames=4)
    result = compute_motion_masks(reader, adapter, basic, MASK_PARAMS, chunk_frames=12)
    assert result.masks.shape == (48, 1, 1080, 1920)
    assert result.masks.dtype == np.float16
    assert set(np.unique(result.masks)).issubset({0.0, 1.0})
    # Per-frame static area at k=3 (diagnostics: ~0.91 on 48 frames).
    assert 0.80 <= result.static_fraction_mean <= 0.98
    # Long-term static area (survey: 69% of bright area over the full stream).
    static_region = long_term_static_region(result)
    assert static_region.dtype == np.bool_ and static_region.shape == (540, 960)
    signal = basic.temporal_mean.astype(np.float32).mean(axis=0) > (
        np.mean(list(basic.black_levels.values())) + 15.0
    )
    long_term_static_signal = float(static_region[signal].mean())
    assert 0.55 <= long_term_static_signal <= 0.95


def test_motion_mask_marks_moving_people_region(tmp_path: Path) -> None:
    # People cross the back corridor of the darkroom (top rows of the frame);
    # the fan spins on the right. Motion must exceed 1% of the frame somewhere.
    adapter = resolve_adapter("04", "shdarkroom", "645x")
    reader, _ = _open("645x")
    basic = compute_basic_stats(reader, adapter, chunk_frames=12, max_frames=48, black_sample_frames=4)
    result = compute_motion_masks(reader, adapter, basic, MASK_PARAMS, chunk_frames=12)
    per_frame_motion = 1.0 - result.static_fraction_per_frame
    assert float(per_frame_motion.max()) > 0.01


def test_exposure_fit_real_645x_to_10x(tmp_path: Path) -> None:
    reference = _basic("10x")
    moving = _basic("645x")
    ref_reader, _ = _open("10x")
    mov_reader, _ = _open("645x")
    ref_mask = compute_motion_masks(ref_reader, resolve_adapter("04", "shdarkroom", "10x"),
                                    reference, MASK_PARAMS, chunk_frames=12)
    mov_mask = compute_motion_masks(mov_reader, resolve_adapter("04", "shdarkroom", "645x"),
                                    moving, MASK_PARAMS, chunk_frames=12)
    static = long_term_static_region(ref_mask) & long_term_static_region(mov_mask)
    fit = fit_exposure_pair(
        moving.temporal_mean, moving.black_levels,
        reference.temporal_mean, reference.black_levels,
        static, reference_label="10x",
    )
    # Survey: 10x chart signal 1091 DN vs 645x 194 DN (~5.6x); the whole-scene
    # slope converges near 6.2 (diagnostics). Median inlier residual ~ -7 DN
    # at 10x scale; the p90 tail is relighting-limited and only loosely bound.
    assert 5.0 <= fit.slopes["Gr"] <= 7.5
    assert all(v > 0 for v in fit.slopes.values())
    assert abs(fit.residual_median_dn["Gr"]) < 15.0
    assert fit.residual_p90_dn["Gr"] < 200.0
    assert fit.inlier_fraction["Gr"] > 0.9
    assert fit.n_pixels["Gr"] >= 100_000


def test_manifest_end_to_end_real_313x(tmp_path: Path) -> None:
    adapter = resolve_adapter("04", "shdarkroom", "313x")
    reader, meta = _open("313x")
    manifest = StreamManifest.create(adapter, "313x", None, sorted((SHDARKROOM / "313x").glob("*.raw"))[0], meta)
    basic = compute_basic_stats(reader, adapter, chunk_frames=12, max_frames=48, black_sample_frames=4)
    manifest.frames_total = basic.frames_total
    manifest.frames_used = basic.frames_used
    manifest.black_levels = basic.black_levels
    mask = compute_motion_masks(reader, adapter, basic, MASK_PARAMS, chunk_frames=12)
    static_region = long_term_static_region(mask)
    manifest.sigma_split_half = split_half_sigma(basic, static_region)

    # Survey anchors for 313x (full stream): black pedestal ~259.8. The
    # split-half sigma here is TOTAL temporal noise (scene shimmer included),
    # so it runs above the survey's sensor-only 20.1 DN (measured ~44).
    assert 250.0 <= manifest.black_levels["Gr"] <= 280.0
    assert 25.0 <= manifest.sigma_split_half["per_channel"]["Gr"] <= 60.0
    assert manifest.frames_total == 300 and manifest.frames_used == 48

    written = manifest.write_json(tmp_path / "manifests")
    payload = json.loads(written.read_text())
    assert payload["gain_label"] == "313x"
    assert payload["code_version"] == "finetune_a0_v1"
    assert "R" in payload["black_levels"]


def test_progress_tracker_contract(tmp_path: Path) -> None:
    clock = {"t": 0.0}

    def monotonic() -> float:
        return clock["t"]

    def now() -> datetime:
        return datetime.fromtimestamp(1758364800 + clock["t"], tz=timezone.utc)

    tracker = A0ProgressTracker(tmp_path, total_units=3, monotonic=monotonic, now=now)
    state = tracker.start()
    assert state.status == "calibrating" and state.completed_units == 0

    clock["t"] = 10.0
    tracker.start_unit("04/shdarkroom/10x", "measure")
    clock["t"] = 20.0
    state = tracker.complete_unit()
    assert state.completed_units == 1
    assert state.seconds_per_unit_ewma == 10.0
    assert state.eta_seconds == 20.0  # 10 s/unit x 2 remaining

    tracker.start_unit("04/shdarkroom/64x", "measure")
    tracker.complete_unit()
    tracker.start_unit("04/shdarkroom/128x", "measure")
    final = tracker.complete_unit()
    assert final.completed_units == 3
    done = tracker.complete()
    assert done.status == "complete" and done.eta_seconds == 0.0

    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["protocol"] == "finetune_a0_progress_v1"
    assert payload["status"] == "complete" and payload["total_units"] == 3

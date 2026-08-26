from __future__ import annotations

import numpy as np
import torch


def _dataset():
    from raw_fusion.v2.data_first_dataset import DataFirstDataset
    from raw_fusion.v2.data_first_supervision import MOG2SupervisionConfig
    from raw_fusion.v2.md import Mog2ConfigV2

    frames = {
        (condition, kind, frame): np.full((4, 540, 960), 512, dtype=np.uint16)
        for condition in ("128x", "645x")
        for kind in ("noisy", "denoised", "fused")
        for frame in range(200)
    }
    return DataFirstDataset.from_arrays(
        frames,
        supervision_config=MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)),
        target_frames={"128x": (58, 59), "645x": (58, 59)},
    )


def test_data_first_two_step_training_writes_finite_outputs(tmp_path) -> None:
    from raw_fusion.v2.data_first_train import run_data_first_training

    result = run_data_first_training(_dataset(), tmp_path, device="cpu", seed=3, batch_size=2, max_steps=2)
    assert result.global_step == 2
    assert (tmp_path / "data_first_v2.pt").is_file()
    values = [line for line in (tmp_path / "train.jsonl").read_text().splitlines() if line]
    assert len(values) == 2
    assert all(np.isfinite(float(__import__("json").loads(line)["loss"])) for line in values)


def test_data_first_training_streams_metrics_and_saves_periodic_checkpoint(tmp_path) -> None:
    from raw_fusion.v2.data_first_train import run_data_first_training

    result = run_data_first_training(
        _dataset(),
        tmp_path,
        device="cpu",
        seed=3,
        batch_size=2,
        max_steps=1,
        log_interval=1,
        checkpoint_interval=1,
    )

    records = [__import__("json").loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines() if line]
    assert result.global_step == 1
    assert [record["global_step"] for record in records] == [1]
    assert {"loss", "loss_alpha", "elapsed_seconds", "steps_per_second", "eta_seconds"} <= set(records[0])
    assert (tmp_path / "checkpoint_step_000001.pt").is_file()


def test_data_first_training_resumes_from_checkpoint_at_next_step(tmp_path) -> None:
    from raw_fusion.v2.data_first_train import run_data_first_training

    run_data_first_training(
        _dataset(),
        tmp_path,
        device="cpu",
        seed=3,
        batch_size=2,
        max_steps=1,
        checkpoint_interval=1,
    )
    result = run_data_first_training(
        _dataset(),
        tmp_path,
        device="cpu",
        seed=3,
        batch_size=2,
        max_steps=2,
        checkpoint_interval=1,
        resume=tmp_path / "checkpoint_step_000001.pt",
    )

    records = [__import__("json").loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines() if line]
    assert result.global_step == 2
    assert [record["global_step"] for record in records] == [1, 2]


def test_data_first_training_batch_contains_no_md_model_input() -> None:
    from raw_fusion.v2.data_first_train import build_data_first_batch
    from raw_fusion.v2.data_first_dataset import DataFirstSampleRow

    batch = build_data_first_batch(_dataset(), [DataFirstSampleRow("128x", "train", 58, 4, 5), DataFirstSampleRow("645x", "train", 58, 4, 5)])
    assert set(batch.model_inputs.as_mapping()) == {"prev_noisy", "curr_noisy", "denoised", "fused", "c_tilde"}
    assert "md_mask" not in batch.model_inputs.as_mapping()

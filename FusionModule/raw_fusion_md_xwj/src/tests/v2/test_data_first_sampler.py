from __future__ import annotations


def test_deterministic_sampler_balances_conditions_and_resumes() -> None:
    from raw_fusion.v2.data_first_sampler import DeterministicCellSampler

    class Dataset:
        target_frames = {"128x": (58, 59, 60), "645x": (58, 59, 60)}
        cell_shape = (17, 30)

    first = DeterministicCellSampler.build(Dataset(), seed=7, batch_size=4, conditions=("128x", "645x"))
    second = DeterministicCellSampler.build(Dataset(), seed=7, batch_size=4, conditions=("128x", "645x"))
    assert first.digest == second.digest
    assert first.rows == second.rows
    assert first.rows[0].condition == "128x"
    assert first.rows[1].condition == "645x"
    resumed = first.resume(0, 1)
    assert resumed[0] == first.rows[1 * 4]


def test_deterministic_sampler_rejects_odd_batch_size() -> None:
    import pytest
    from raw_fusion.v2.data_first_sampler import DeterministicCellSampler

    class Dataset:
        target_frames = {"128x": (58,), "645x": (58,)}
        cell_shape = (17, 30)

    with pytest.raises(ValueError, match="even"):
        DeterministicCellSampler.build(Dataset(), seed=1, batch_size=3, conditions=("128x", "645x"))

from __future__ import annotations

import numpy as np
import torch


def test_data_first_sample_keeps_model_inputs_separate_from_supervision() -> None:
    from raw_fusion.v2.data_first_dataset import DataFirstDataset, DataFirstSampleRow
    from raw_fusion.v2.data_first_supervision import MOG2SupervisionConfig
    from raw_fusion.v2.md import Mog2ConfigV2

    frames = {
        (condition, kind, frame): np.full((4, 540, 960), 512, dtype=np.uint16)
        for condition in ("128x", "645x")
        for kind in ("noisy", "denoised", "fused")
        for frame in range(200)
    }
    dataset = DataFirstDataset.from_arrays(
        frames,
        supervision_config=MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)),
    )
    sample = dataset.sample(DataFirstSampleRow("128x", "train", 58, 4, 5))
    assert set(sample.model_inputs.as_mapping()) == {"prev_noisy", "curr_noisy", "denoised", "fused", "c_tilde"}
    assert sample.model_inputs.prev_noisy.shape == (1, 4, 320, 320)
    assert sample.supervision.policy_alpha_valid.shape[0] == 1
    assert isinstance(sample.model_inputs.prev_noisy, torch.Tensor)


def test_data_first_dataset_rejects_out_of_range_crop() -> None:
    import pytest
    from raw_fusion.v2.data_first_dataset import DataFirstDataset, DataFirstSampleRow
    from raw_fusion.v2.data_first_supervision import MOG2SupervisionConfig
    from raw_fusion.v2.md import Mog2ConfigV2
    from raw_fusion.v2.schemas.common import ContractError

    frames = {
        ("128x", kind, frame): np.zeros((4, 540, 960), dtype=np.uint16)
        for kind in ("noisy", "denoised", "fused")
        for frame in range(200)
    }
    dataset = DataFirstDataset.from_arrays(
        frames,
        supervision_config=MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)),
    )
    with pytest.raises(ContractError, match="cell"):
        dataset.sample(DataFirstSampleRow("128x", "train", 58, 99, 99))

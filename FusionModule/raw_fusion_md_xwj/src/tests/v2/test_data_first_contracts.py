from __future__ import annotations

import pytest
import torch


def _images(batch: int = 2, height: int = 128, width: int = 160) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260826)
    return {
        name: torch.rand((batch, 4, height, width), generator=generator)
        for name in ("prev_noisy", "curr_noisy", "denoised", "fused")
    }


def test_data_first_input_batch_has_exact_model_keys_and_rejects_md() -> None:
    from raw_fusion.v2.data_first_contracts import DataFirstInputBatch, derive_input_condition

    images = _images()
    condition = derive_input_condition(**images)
    batch = DataFirstInputBatch(**images, c_tilde=condition)
    assert tuple(batch.as_mapping()) == ("prev_noisy", "curr_noisy", "denoised", "fused", "c_tilde")
    with pytest.raises(TypeError, match="exactly"):
        DataFirstInputBatch.from_mapping({**batch.as_mapping(), "md_mask": torch.zeros((2, 1, 128, 160))})


def test_input_condition_is_deterministic_and_has_no_md_dependency() -> None:
    from raw_fusion.v2.data_first_contracts import derive_input_condition

    images = _images()
    first = derive_input_condition(**images)
    second = derive_input_condition(**images)
    assert torch.equal(first, second)
    assert first.shape == (2, 4)
    assert torch.isfinite(first).all()


def test_input_condition_requires_floating_same_shape_images() -> None:
    from raw_fusion.v2.data_first_contracts import derive_input_condition
    from raw_fusion.v2.schemas.common import ContractError

    images = _images()
    images["fused"] = images["fused"][:, :, :-1]
    with pytest.raises(ContractError, match="identical"):
        derive_input_condition(**images)


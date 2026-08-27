from __future__ import annotations

import torch


def test_data_first_checkpoint_round_trip(tmp_path) -> None:
    from raw_fusion.v2.data_first_checkpoint import load_data_first_checkpoint, save_data_first_checkpoint
    from raw_fusion.v2.model import FrequencyFusionConfigV2, FrequencyFusionCore

    model = FrequencyFusionCore(FrequencyFusionConfigV2.production())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "data_first_v2.pt"
    ref = save_data_first_checkpoint(path, model, optimizer, {"global_step": 3}, {"schedule_digest": "abc"})
    loaded = load_data_first_checkpoint(path)
    assert ref.sha256
    assert loaded.payload["protocol"] == "raw_fusion_v2_data_first"
    assert loaded.payload["md_used_as_model_input"] is False

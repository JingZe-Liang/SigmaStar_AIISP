from __future__ import annotations

import numpy as np


def _dataset(tmp_path):
    from raw_fusion.v2.data_first_dataset import DataFirstDataset
    from raw_fusion.v2.data_first_supervision import MOG2SupervisionConfig
    from raw_fusion.v2.md import Mog2ConfigV2

    frames = {
        (condition, kind, frame): np.full((4, 540, 960), 512, dtype=np.uint16)
        for condition in ("128x", "645x")
        for kind in ("noisy", "denoised", "fused")
        for frame in range(200)
    }
    return DataFirstDataset.from_arrays(frames, supervision_config=MOG2SupervisionConfig(mog2=Mog2ConfigV2(history=50)))


def test_data_first_inference_does_not_require_md_and_writes_outputs(tmp_path) -> None:
    import torch
    from raw_fusion.v2.data_first_checkpoint import save_data_first_checkpoint
    from raw_fusion.v2.data_first_infer import run_data_first_inference
    from raw_fusion.v2.model import FrequencyFusionConfigV2, FrequencyFusionCore

    model = FrequencyFusionCore(FrequencyFusionConfigV2.production())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "data_first_v2.pt"
    save_data_first_checkpoint(checkpoint, model, optimizer, {"global_step": 0}, {})
    result = run_data_first_inference(checkpoint, _dataset(tmp_path), tmp_path / "out", conditions=("128x",), max_frames=1)
    assert result.frames == 1
    assert (tmp_path / "out" / "inference_128x.npy").is_file()
    assert result.fallback_fraction == 0.0


def test_invalid_model_output_preserves_denoised_bytes(tmp_path) -> None:
    import torch
    from raw_fusion.v2.data_first_checkpoint import save_data_first_checkpoint
    from raw_fusion.v2.data_first_infer import run_data_first_inference
    from raw_fusion.v2.model import FrequencyFusionConfigV2, FrequencyFusionCore

    model = FrequencyFusionCore(FrequencyFusionConfigV2.production())
    with torch.no_grad():
        model.q_head.bias.fill_(float("nan"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "bad.pt"
    save_data_first_checkpoint(checkpoint, model, optimizer, {"global_step": 0}, {})
    dataset = _dataset(tmp_path)
    result = run_data_first_inference(checkpoint, dataset, tmp_path / "out", conditions=("128x",), max_frames=1)
    output = np.load(tmp_path / "out" / "inference_128x.npy")
    expected = dataset._read("128x", "denoised", 58)
    assert np.array_equal(output[0], expected)
    assert result.fallback_fraction == 1.0


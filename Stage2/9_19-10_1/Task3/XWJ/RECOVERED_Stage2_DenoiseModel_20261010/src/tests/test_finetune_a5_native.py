import numpy as np

from raw_fusion.finetune.a5_native import evaluate_native_outputs


def test_native_identity_has_zero_error() -> None:
    black = np.asarray([254.75, 257.125, 256.875, 255.125], dtype=np.float32)
    reference = np.full((4, 540, 960), 400.0, dtype=np.float32)
    normalized = (reference - black[:, None, None]) / (4095.0 - black[:, None, None])
    video = np.repeat(normalized[None], 2, axis=0)
    motion = np.zeros((2, 540, 960), dtype=np.float32)
    valid = np.ones((540, 960), dtype=bool)
    metrics = evaluate_native_outputs(video, video, reference, valid, motion, black)
    assert metrics["finetuned"]["rmse_dn"] == 0.0
    assert metrics["input"]["rmse_dn"] == 0.0
    assert metrics["legal_fraction"] == 1.0


def test_native_input_mask_excludes_motion() -> None:
    black = np.asarray([254.75, 257.125, 256.875, 255.125], dtype=np.float32)
    reference = np.full((4, 540, 960), 400.0, dtype=np.float32)
    normalized = (reference - black[:, None, None]) / (4095.0 - black[:, None, None])
    video = np.repeat(normalized[None], 2, axis=0)
    motion = np.ones((2, 540, 960), dtype=np.float32)
    valid = np.ones((540, 960), dtype=bool)
    metrics = evaluate_native_outputs(video, video, reference, valid, motion, black)
    assert metrics["finetuned"]["pixels"] == 0

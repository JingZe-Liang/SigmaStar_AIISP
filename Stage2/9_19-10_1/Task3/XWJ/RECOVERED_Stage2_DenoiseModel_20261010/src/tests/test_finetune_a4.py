import numpy as np
import torch

from raw_fusion.finetune.a4 import evaluate_outputs, infer_video


class IdentityWindow(torch.nn.Module):
    def forward(self, x):
        return x


def test_infer_video_blends_identity_windows():
    video = np.random.default_rng(4).random((8, 4, 540, 960), dtype=np.float32)
    output = infer_video(IdentityWindow(), video, torch.device("cpu"))
    np.testing.assert_allclose(output, video, rtol=0, atol=1e-6)


def test_evaluate_outputs_reports_zero_error_for_reference():
    black = np.asarray([266.125, 275.625, 275.375, 267.0], dtype=np.float32)
    reference = np.full((4, 540, 960), 1200.0, dtype=np.float32)
    normalized = (reference - black[:, None, None]) / (4095.0 - black[:, None, None])
    video = np.repeat(normalized[None], 2, axis=0)
    overlap = np.ones((2, 1, 540, 960), dtype=bool)
    motion = np.zeros((2, 540, 960), dtype=np.float32)
    metrics = evaluate_outputs(video, video, reference, overlap, motion, black)
    assert metrics["finetuned"]["rmse_dn"] == 0.0
    assert metrics["legal_fraction"] == 1.0

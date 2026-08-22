"""Small no-data regression checks for the v2 model and three-term objective."""

from __future__ import annotations

import inspect

import torch

from losses import LossWeights, TextureFusionLoss
from model import TextureGateNet


def test_model_uses_three_current_raw_inputs_and_loss_backpropagates() -> None:
    torch.manual_seed(0)
    model = TextureGateNet(base_channels=8)
    assert tuple(inspect.signature(model.forward).parameters) == ("noisy", "base", "temporal")
    noisy = torch.rand(2, 4, 32, 32)
    base = torch.rand(2, 4, 32, 32)
    temporal = torch.rand(2, 4, 32, 32)
    alpha = model(noisy, base, temporal)
    batch = {
        "noisy": noisy,
        "base": base,
        "temporal": temporal,
        "proxy": torch.rand(2, 4, 32, 32),
        "static_mask": torch.ones(2, 1, 32, 32),
        "motion_mask": torch.zeros(2, 1, 32, 32),
    }
    loss, metrics = TextureFusionLoss(LossWeights())(alpha, batch)
    assert torch.isfinite(loss)
    assert set(("gate", "texture", "motion")).issubset(metrics)
    loss.backward()
    assert sum(parameter.grad.abs().sum().item() for parameter in model.parameters() if parameter.grad is not None) > 0.0

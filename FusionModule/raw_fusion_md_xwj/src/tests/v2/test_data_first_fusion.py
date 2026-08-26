from __future__ import annotations

import torch


def test_safe_q_limit_keeps_composition_in_normalized_raw_range() -> None:
    from raw_fusion.v2.data_first_fusion import limit_q_to_raw_range

    denoised = torch.tensor([[[[0.95]], [[0.10]]]])
    delta = torch.tensor([[[[0.50]], [[-0.50]]]])
    limited = limit_q_to_raw_range(denoised, delta, torch.tensor([[[[0.25]]]]))
    composed = denoised + limited * delta
    assert float(limited) == __import__("pytest").approx(0.1)
    assert bool(((composed >= 0.0) & (composed <= 1.0)).all())

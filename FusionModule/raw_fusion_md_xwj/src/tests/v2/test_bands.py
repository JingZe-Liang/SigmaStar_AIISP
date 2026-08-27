from __future__ import annotations

import torch


def test_b2_annihilates_constant_and_linear_interior() -> None:
    from raw_fusion.v2.bands import b2

    constant = torch.ones((1, 4, 65, 67), dtype=torch.float32)
    assert torch.equal(b2(constant), torch.zeros_like(constant))
    x = torch.arange(67, dtype=torch.float32).view(1, 1, 1, 67).expand(1, 4, 65, 67)
    assert torch.max(torch.abs(b2(x)[..., 8:-8, 8:-8])).item() < 1e-5


def test_low_pass_preserves_shape_and_dtype() -> None:
    from raw_fusion.v2.bands import low_pass

    x = torch.randn((2, 4, 17, 19), dtype=torch.float32)
    y = low_pass(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32

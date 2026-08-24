from __future__ import annotations

import pytest
import torch

from raw_fusion.config import ModelConfig
from raw_fusion.model import CausalRawFusionNet


_CHANNELS = (8, 16, 24)


def make_inputs(*, batch: int, height: int, width: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "prev_noisy": torch.rand(batch, 4, height, width, generator=generator),
        "curr_noisy": torch.rand(batch, 4, height, width, generator=generator),
        "denoised": torch.rand(batch, 4, height, width, generator=generator),
        "fused": torch.rand(batch, 4, height, width, generator=generator),
    }


def make_model(*, residual_scale: float = 0.03, use_temporal: bool = True) -> CausalRawFusionNet:
    return CausalRawFusionNet(ModelConfig(_CHANNELS, residual_scale, use_temporal))


def set_nonzero_output_weights(model: CausalRawFusionNet) -> None:
    with torch.no_grad():
        gate = model.gate_head[-1]
        correction = model.correction_head[-1]
        gate.weight.copy_(torch.linspace(-0.08, 0.12, gate.weight.numel()).reshape_as(gate.weight))
        gate.bias.fill_(0.2)
        correction.weight.copy_(
            torch.linspace(-0.1, 0.15, correction.weight.numel()).reshape_as(correction.weight)
        )
        correction.bias.copy_(torch.tensor((-0.3, -0.1, 0.1, 0.3)))


def set_constant_output_logits(
    model: CausalRawFusionNet,
    *,
    gate_logit: float,
    correction_logits: tuple[float, float, float, float],
) -> None:
    with torch.no_grad():
        gate = model.gate_head[-1]
        correction = model.correction_head[-1]
        gate.weight.zero_()
        gate.bias.fill_(gate_logit)
        correction.weight.zero_()
        correction.bias.copy_(torch.tensor(correction_logits))


def assert_finite_nonzero_gradient(parameter: torch.Tensor) -> None:
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) > 0


def assert_finite_zero_gradient(parameter: torch.Tensor) -> None:
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) == 0


def fusion_loss(model: CausalRawFusionNet, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**inputs)
    return output.prediction.square().mean() + output.gate.mean() + output.correction.mean()


def test_forward_shapes_and_gate_broadcast() -> None:
    model = make_model()
    inputs = make_inputs(batch=2, height=65, width=97)

    output = model(**inputs)

    assert output.prediction.shape == (2, 4, 65, 97)
    assert output.base.shape == (2, 4, 65, 97)
    assert output.gate.shape == (2, 1, 65, 97)
    assert output.correction.shape == (2, 4, 65, 97)


def test_initial_model_is_exact_fixed_average() -> None:
    model = make_model().eval()
    inputs = make_inputs(batch=1, height=32, width=48)

    with torch.no_grad():
        output = model(**inputs)

    expected = 0.5 * (inputs["denoised"] + inputs["fused"])
    torch.testing.assert_close(output.gate, torch.full_like(output.gate, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(output.correction, torch.zeros_like(output.correction), rtol=0, atol=0)
    torch.testing.assert_close(output.prediction, expected, rtol=0, atol=0)


def test_candidate_only_nonzero_outputs_do_not_depend_on_noisy() -> None:
    model = make_model(use_temporal=False).eval()
    assert model.temporal_branch is None
    set_nonzero_output_weights(model)
    first = make_inputs(batch=1, height=16, width=16)
    second = {
        **first,
        "prev_noisy": 1.0 - first["prev_noisy"],
        "curr_noisy": 1.0 - first["curr_noisy"],
    }

    with torch.no_grad():
        first_output = model(**first)
        second_output = model(**second)

    assert torch.any(first_output.gate != 0.5)
    assert torch.any(first_output.correction != 0.0)
    torch.testing.assert_close(first_output.prediction, second_output.prediction, rtol=0, atol=0)
    torch.testing.assert_close(first_output.gate, second_output.gate, rtol=0, atol=0)
    torch.testing.assert_close(first_output.correction, second_output.correction, rtol=0, atol=0)


def test_candidate_only_autograd_excludes_noisy_inputs() -> None:
    model = make_model(use_temporal=False)
    assert model.temporal_branch is None
    set_nonzero_output_weights(model)
    inputs = make_inputs(batch=1, height=16, width=16)
    for value in inputs.values():
        value.requires_grad_()

    fusion_loss(model, inputs).backward()

    assert inputs["prev_noisy"].grad is None
    assert inputs["curr_noisy"].grad is None
    assert_finite_nonzero_gradient(inputs["denoised"])
    assert_finite_nonzero_gradient(inputs["fused"])


def test_output_terms_follow_nonzero_fusion_equations() -> None:
    model = make_model().eval()
    set_constant_output_logits(
        model,
        gate_logit=0.75,
        correction_logits=(-1.0, -0.25, 0.5, 1.25),
    )
    inputs = make_inputs(batch=1, height=17, width=23)

    with torch.no_grad():
        output = model(**inputs)

    expected_gate = torch.full_like(output.gate, 0.6791787147521973)
    expected_correction = torch.tensor(
        (-0.022847825, -0.00734756, 0.013863515, 0.025448509),
        dtype=output.correction.dtype,
    ).reshape(1, 4, 1, 1).expand_as(output.correction)
    expected_base = expected_gate * inputs["denoised"] + (1.0 - expected_gate) * inputs["fused"]
    torch.testing.assert_close(output.gate, expected_gate)
    torch.testing.assert_close(output.correction, expected_correction)
    torch.testing.assert_close(output.base, expected_base)
    torch.testing.assert_close(output.prediction, expected_base + expected_correction)


def test_zero_residual_scale_forces_exact_zero_correction() -> None:
    model = make_model(residual_scale=0.0).eval()
    set_constant_output_logits(
        model,
        gate_logit=-0.5,
        correction_logits=(-2.0, -1.0, 1.0, 2.0),
    )
    inputs = make_inputs(batch=1, height=8, width=8)

    with torch.no_grad():
        output = model(**inputs)

    torch.testing.assert_close(output.correction, torch.zeros_like(output.correction), rtol=0, atol=0)
    torch.testing.assert_close(output.prediction, output.base, rtol=0, atol=0)


def test_prediction_is_not_clamped_to_normalized_range() -> None:
    model = make_model(use_temporal=False).eval()
    inputs = make_inputs(batch=1, height=8, width=8)
    inputs["denoised"].fill_(1.25)
    inputs["fused"].fill_(1.25)

    with torch.no_grad():
        output = model(**inputs)

    torch.testing.assert_close(output.prediction, torch.full_like(output.prediction, 1.25))
    assert output.prediction.max() > 1.0


def test_mismatched_input_shapes_fail_clearly() -> None:
    model = make_model()
    inputs = make_inputs(batch=1, height=16, width=16)
    inputs["fused"] = torch.rand(1, 4, 16, 15)

    with pytest.raises(ValueError, match="same shape"):
        model(**inputs)


def test_mixed_input_dtypes_fail_at_entry() -> None:
    model = make_model()
    inputs = make_inputs(batch=1, height=8, width=8)
    inputs["fused"] = inputs["fused"].double()

    with pytest.raises(ValueError, match="same dtype"):
        model(**inputs)


def test_mixed_input_devices_fail_at_entry() -> None:
    model = make_model()
    inputs = make_inputs(batch=1, height=8, width=8)
    inputs["fused"] = inputs["fused"].to("meta")

    with pytest.raises(ValueError, match="same device"):
        model(**inputs)


def test_model_device_mismatch_fails_at_entry() -> None:
    model = make_model()
    inputs = {
        name: value.to("meta")
        for name, value in make_inputs(batch=1, height=8, width=8).items()
    }

    with pytest.raises(ValueError, match="model.*device"):
        model(**inputs)


def test_model_dtype_mismatch_without_autocast_fails_at_entry() -> None:
    model = make_model().double()
    inputs = make_inputs(batch=1, height=8, width=8)

    with pytest.raises(TypeError, match="model.*dtype"):
        model(**inputs)


def test_cpu_autocast_rejects_unsupported_input_dtype_at_entry() -> None:
    model = make_model().eval()
    inputs = {
        name: value.double()
        for name, value in make_inputs(batch=1, height=8, width=8).items()
    }

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast.*dtype"):
            model(**inputs)


def test_cpu_autocast_allows_legal_input_and_parameter_dtypes() -> None:
    model = make_model().eval()
    inputs = {
        name: value.to(torch.bfloat16)
        for name, value in make_inputs(batch=1, height=8, width=8).items()
    }

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(**inputs)

    assert torch.isfinite(output.prediction).all()
    assert output.prediction.dtype == torch.bfloat16


def test_cpu_autocast_float32_inputs_keep_uniform_output_dtype() -> None:
    model = make_model().eval()
    inputs = make_inputs(batch=1, height=8, width=8)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(**inputs)
    assert output.prediction.dtype == torch.float32
    assert output.base.dtype == torch.float32
    assert output.gate.dtype == torch.float32
    assert output.correction.dtype == torch.float32


def test_zero_heads_block_first_step_then_propagate_after_update() -> None:
    model = make_model()
    inputs = make_inputs(batch=2, height=17, width=19)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert model.temporal_branch is not None

    fusion_loss(model, inputs).backward()

    assert_finite_nonzero_gradient(model.gate_head[-1].weight)
    assert_finite_nonzero_gradient(model.correction_head[-1].weight)
    assert_finite_zero_gradient(model.decode2.fuse[0].weight)
    assert_finite_zero_gradient(model.candidate_branch.stem.project[0].weight)
    assert_finite_zero_gradient(model.temporal_branch.noisy_stem.project[0].weight)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    fusion_loss(model, inputs).backward()

    assert_finite_nonzero_gradient(model.decode2.fuse[0].weight)
    assert_finite_nonzero_gradient(model.candidate_branch.stem.project[0].weight)
    assert_finite_nonzero_gradient(model.temporal_branch.noisy_stem.project[0].weight)

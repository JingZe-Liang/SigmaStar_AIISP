from __future__ import annotations

import pytest
import torch
from torch import nn


def _inputs(
    *,
    batch: int = 1,
    height: int = 320,
    width: int = 320,
    c_tilde: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    target_device = torch.device(device)
    tensors = {
        name: torch.rand((batch, 4, height, width), generator=generator, dtype=torch.float32)
        for name in ("prev_noisy", "curr_noisy", "denoised", "fused")
    }
    result = {name: value.to(target_device) for name, value in tensors.items()}
    result["c_tilde"] = (
        torch.zeros((batch, 4), dtype=torch.float32, device=target_device)
        if c_tilde is None
        else c_tilde.to(device=target_device, dtype=torch.float32)
    )
    return result


def _core():
    from raw_fusion.v2.model import FrequencyFusionConfigV2, FrequencyFusionCore

    return FrequencyFusionCore(FrequencyFusionConfigV2.production())


def test_channel_norm_is_independent_of_neighboring_spatial_extent() -> None:
    from raw_fusion.v2.model import ChannelNorm

    layer = ChannelNorm(24)
    center = torch.randn((1, 24, 1, 1), generator=torch.Generator().manual_seed(1))
    large = center.expand(1, 24, 17, 19).clone()
    assert torch.allclose(layer(center)[..., 0, 0], layer(large)[..., 8, 9], atol=1e-6, rtol=0)


def test_initial_film_is_identity_and_zero_class_probability_is_near_point97() -> None:
    model = _core()
    assert torch.count_nonzero(model.film_out.weight) == 0
    assert torch.count_nonzero(model.film_out.bias) == 0
    assert torch.count_nonzero(model.q_head.weight) == 0
    probabilities = torch.softmax(model.q_head.bias.detach(), dim=0)
    assert probabilities[0].item() == pytest.approx(0.97, abs=0.002)


def test_one_state_dict_processes_both_conditions() -> None:
    model = _core().eval()
    ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}
    with torch.no_grad():
        model(**_inputs(c_tilde=torch.tensor([[0.0, 0.0, 0.0, 0.0]])))
        model(**_inputs(c_tilde=torch.tensor([[1.0, -1.0, 0.5, -0.5]])))
    assert ids_before == {name: id(parameter) for name, parameter in model.named_parameters()}


def test_no_condition_hard_disables_film_and_its_gradients() -> None:
    model = _core()
    with torch.no_grad():
        model.film_out.weight.normal_(mean=0.0, std=0.03)
        model.q_head.weight.normal_(mean=0.0, std=0.01)

    first = _inputs(height=128, width=160, c_tilde=torch.tensor([[0.0, 0.5, -0.5, 1.0]]))
    second = _inputs(height=128, width=160, c_tilde=torch.tensor([[2.0, -1.0, 3.0, -2.0]]))
    first_output = model(**first, condition_enabled=False).q_logits_pixel_core
    second_output = model(**second, condition_enabled=False).q_logits_pixel_core

    assert torch.equal(first_output, second_output)
    first_output.square().mean().backward()
    assert model.film_in.weight.grad is None
    assert model.film_in.bias.grad is None
    assert model.film_out.weight.grad is None
    assert model.film_out.bias.grad is None


def test_measured_dependency_radius_is_within_production_halo() -> None:
    from raw_fusion.v2.model import measure_dependency_radius

    assert measure_dependency_radius(_core(), input_shape=(540, 960)) <= 32


def test_measured_dependency_radius_bounds_autograd_support() -> None:
    from raw_fusion.v2.model import CORE_HALO_PACKED, measure_dependency_radius

    torch.manual_seed(91)
    model = _core().eval()
    with torch.no_grad():
        model.q_head.weight.normal_(mean=0.0, std=0.03)
    inputs = _inputs(height=256, width=256)
    inputs["curr_noisy"].requires_grad_(True)
    output_index = 64
    model(**inputs).q_logits_pixel_core[0, 0, output_index, output_index].backward()
    support = (inputs["curr_noisy"].grad.abs().sum(dim=1)[0] > 0).nonzero()
    radius = measure_dependency_radius(model, input_shape=(256, 256))
    origin = output_index + CORE_HALO_PACKED
    assert support[:, 0].min().item() >= origin - radius
    assert support[:, 0].max().item() <= origin + radius
    assert support[:, 1].min().item() >= origin - radius
    assert support[:, 1].max().item() <= origin + radius


def test_unknown_spatial_operator_is_rejected_by_radius_scan() -> None:
    from raw_fusion.v2.model import measure_dependency_radius
    from raw_fusion.v2.schemas.common import ContractError

    model = _core()
    model.unregistered_spatial_operator = nn.MaxPool2d(3, stride=1, padding=1)
    with pytest.raises(ContractError, match="dependency radius"):
        measure_dependency_radius(model, input_shape=(540, 960))


@pytest.mark.parametrize("normalization", [nn.GroupNorm(1, 24), nn.BatchNorm2d(24)])
def test_architecture_rejects_group_or_batch_norm(normalization: nn.Module) -> None:
    from raw_fusion.v2.model import validate_core_architecture
    from raw_fusion.v2.schemas.common import ContractError

    model = _core()
    model.norm = normalization
    with pytest.raises(ContractError, match="normalization"):
        validate_core_architecture(model)


@pytest.mark.parametrize("module_name", ["gain_embedding", "condition_expert_branch"])
def test_architecture_rejects_gain_embedding_or_branch(module_name: str) -> None:
    from raw_fusion.v2.model import validate_core_architecture
    from raw_fusion.v2.schemas.common import ContractError

    model = _core()
    setattr(model, module_name, nn.Identity())
    with pytest.raises(ContractError, match="gain-specific"):
        validate_core_architecture(model)


@pytest.mark.parametrize("head_name", ["gate_head", "correction_head", "residual_head"])
def test_architecture_rejects_gate_correction_or_nonlogit_head(head_name: str) -> None:
    from raw_fusion.v2.model import validate_core_architecture
    from raw_fusion.v2.schemas.common import ContractError

    model = _core()
    setattr(model, head_name, nn.Conv2d(24, 1, 1))
    with pytest.raises(ContractError, match="q logits only"):
        validate_core_architecture(model)


@pytest.mark.parametrize(("height", "width", "expected"), [(320, 320, (256, 256)), (256, 256, (192, 192))])
def test_core_only_returns_the_valid_32_pixel_core(height: int, width: int, expected: tuple[int, int]) -> None:
    model = _core().eval()
    with torch.no_grad():
        output = model(**_inputs(height=height, width=width))
    assert output.q_logits_pixel_core.shape == (1, 4, *expected)


def test_core_rejects_non_model_inputs_and_noncore_shapes() -> None:
    from raw_fusion.v2.schemas.common import ContractError

    model = _core()
    with pytest.raises(ContractError, match="only"):
        model(**_inputs(height=64, width=256))
    inputs = _inputs(height=256, width=256)
    inputs["p_hard"] = torch.zeros((1, 1, 256, 256))
    with pytest.raises(TypeError, match="unexpected keyword"):
        model(**inputs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_core_has_finite_conditioned_gradients(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = _core().to(device)
    with torch.no_grad():
        model.q_head.weight.normal_(mean=0.0, std=0.01)
    inputs = _inputs(height=128, width=160, device=device)
    output = model(**inputs).q_logits_pixel_core
    output.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

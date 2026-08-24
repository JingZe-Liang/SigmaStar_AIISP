from __future__ import annotations

import math

import pytest
import torch

from raw_fusion.metrics import MetricAccumulator, baseline_predictions, compute_frame_metrics


def test_fixed_average_baseline_and_psnr() -> None:
    denoised = torch.zeros(1, 4, 2, 2)
    fused = torch.ones_like(denoised)
    baselines = baseline_predictions(denoised, fused)
    torch.testing.assert_close(baselines["average"], torch.full_like(denoised, 0.5))
    metrics = compute_frame_metrics(
        baselines["average"],
        baselines["average"],
        torch.ones_like(denoised, dtype=torch.bool),
    )
    assert math.isinf(metrics["psnr"])
    assert metrics["mae"] == 0.0


def test_metrics_include_gradient_mae_and_mask_invalid_pixels() -> None:
    prediction = torch.tensor([[[[0.0, 0.5, 0.5]], [[0.0, 0.5, 0.5]], [[0.0, 0.5, 0.5]], [[0.0, 0.5, 0.5]]]])
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    valid[..., 0] = False
    metrics = compute_frame_metrics(prediction, target, valid)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["mse"] == pytest.approx(0.25)
    assert metrics["gradient_mae"] == pytest.approx(0.0)


def test_metric_accumulator_averages_finite_values_and_keeps_inf_semantics() -> None:
    accumulator = MetricAccumulator()
    accumulator.add({"mae": 1.0, "psnr": math.inf})
    accumulator.add({"mae": 3.0, "psnr": math.inf})
    summary = accumulator.compute()
    assert summary == {"mae": 2.0, "psnr": math.inf}


def test_metric_accumulator_ignores_nonfinite_values_when_finite_values_exist() -> None:
    accumulator = MetricAccumulator()
    accumulator.add({"mae": 1.0, "psnr": math.inf})
    accumulator.add({"mae": 3.0, "psnr": 10.0})
    summary = accumulator.compute()
    assert summary["mae"] == 2.0
    assert summary["psnr"] == 10.0


def test_empty_metric_mask_raises_value_error() -> None:
    values = torch.zeros(1, 4, 2, 2)
    with pytest.raises(ValueError, match="valid mask"):
        compute_frame_metrics(values, values, torch.zeros_like(values, dtype=torch.bool))


def test_baseline_rejects_integer_or_mixed_dtype_candidates() -> None:
    floating = torch.zeros(1, 4, 2, 2)
    with pytest.raises(TypeError, match="floating"):
        baseline_predictions(floating.to(torch.int16), floating.to(torch.int16))
    with pytest.raises(ValueError, match="dtype"):
        baseline_predictions(floating, floating.double())


def test_metrics_reject_mixed_dtype_and_non_boolean_mask() -> None:
    prediction = torch.zeros(1, 4, 2, 2)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    with pytest.raises(ValueError, match="dtype"):
        compute_frame_metrics(prediction, prediction.double(), mask)
    with pytest.raises(TypeError, match="boolean"):
        compute_frame_metrics(prediction, prediction, mask.float())

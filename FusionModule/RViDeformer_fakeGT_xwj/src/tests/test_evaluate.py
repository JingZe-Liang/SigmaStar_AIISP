from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from raw_fusion.evaluate import (
    evaluate_models,
    main,
    parse_model_specs,
    validate_saturation_margin_dn,
    write_evaluation_json,
)
from raw_fusion.model import FusionOutput


class AverageModel(nn.Module):
    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        prediction = 0.5 * (denoised + fused)
        gate = torch.full_like(prediction[:, :1], 0.25)
        correction = torch.full_like(prediction, -0.125)
        return FusionOutput(
            prediction=prediction,
            base=prediction,
            gate=gate,
            correction=correction,
        )


class FrameDependentGateModel(nn.Module):
    def forward(
        self,
        prev_noisy: torch.Tensor,
        curr_noisy: torch.Tensor,
        denoised: torch.Tensor,
        fused: torch.Tensor,
    ) -> FusionOutput:
        gate_value = 0.1 if float(curr_noisy.mean()) < 0.25 else 0.9
        prediction = 0.5 * (denoised + fused)
        gate = torch.full_like(prediction[:, :1], gate_value)
        correction = torch.full_like(prediction, 0.25)
        return FusionOutput(
            prediction=prediction,
            base=prediction,
            gate=gate,
            correction=correction,
        )


def _frame(
    *, value: float = 0.0, target_value: float | None = None, frame_index: int = 7
) -> dict[str, torch.Tensor | int]:
    packed = torch.full((1, 4, 4, 4), value, dtype=torch.float32)
    return {
        "frame_index": frame_index,
        "prev_noisy": packed,
        "curr_noisy": packed,
        "denoised": packed,
        "fused": packed,
        "target": torch.full_like(packed, value if target_value is None else target_value),
    }


def test_evaluation_contains_fixed_baselines_and_named_model() -> None:
    report = evaluate_models(
        {"full": AverageModel()},
        [_frame()],
        saturation_threshold=1.0,
    )

    assert set(report["methods"]) == {"denoised", "fused", "average", "full"}
    assert report["methods"]["average"]["aggregate"]["mae"] == 0.0
    assert report["methods"]["full"]["frames"][0]["frame_index"] == 7


def test_evaluation_records_model_diagnostics_and_json_safe_infinite_psnr(tmp_path: Path) -> None:
    report = evaluate_models(
        {"full": AverageModel()},
        [_frame()],
        saturation_threshold=1.0,
    )

    diagnostics = report["methods"]["full"]["diagnostics"]
    assert diagnostics == {
        "frames": [{
            "frame_index": 7,
            "gate_mean": 0.25,
            "gate_p10": 0.25,
            "gate_p50": 0.25,
            "gate_p90": 0.25,
            "correction_abs_mean": 0.125,
        }],
        "aggregate": {
            "gate_mean_frame_mean": 0.25,
            "gate_p10_frame_mean": 0.25,
            "gate_p50_frame_mean": 0.25,
            "gate_p90_frame_mean": 0.25,
            "correction_abs_mean_frame_mean": 0.125,
        },
    }
    assert report["methods"]["average"]["frames"][0]["psnr"] == "inf"
    destination = tmp_path / "report.json"
    write_evaluation_json(destination, report)
    assert json.loads(destination.read_text(encoding="utf-8"))["methods"]["average"]["aggregate"]["psnr"] == "inf"


def test_evaluation_derives_aggregate_psnr_from_aggregate_mse() -> None:
    report = evaluate_models(
        {},
        [
            _frame(value=0.0, target_value=0.0, frame_index=1),
            _frame(value=0.0, target_value=0.5, frame_index=2),
        ],
        saturation_threshold=1.0,
    )

    aggregate = report["methods"]["average"]["aggregate"]
    assert aggregate["mse"] == pytest.approx(0.125)
    assert aggregate["psnr"] == pytest.approx(9.0308998699)


def test_model_diagnostics_preserve_per_frame_quantiles_and_name_aggregate_means() -> None:
    report = evaluate_models(
        {"full": FrameDependentGateModel()},
        [
            _frame(value=0.0, frame_index=3),
            _frame(value=0.5, frame_index=4),
        ],
        saturation_threshold=1.0,
    )

    diagnostics = report["methods"]["full"]["diagnostics"]
    assert diagnostics["frames"] == [
        {
            "frame_index": 3,
            "gate_mean": pytest.approx(0.1),
            "gate_p10": pytest.approx(0.1),
            "gate_p50": pytest.approx(0.1),
            "gate_p90": pytest.approx(0.1),
            "correction_abs_mean": 0.25,
        },
        {
            "frame_index": 4,
            "gate_mean": pytest.approx(0.9),
            "gate_p10": pytest.approx(0.9),
            "gate_p50": pytest.approx(0.9),
            "gate_p90": pytest.approx(0.9),
            "correction_abs_mean": 0.25,
        },
    ]
    assert diagnostics["aggregate"] == {
        "gate_mean_frame_mean": pytest.approx(0.5),
        "gate_p10_frame_mean": pytest.approx(0.5),
        "gate_p50_frame_mean": pytest.approx(0.5),
        "gate_p90_frame_mean": pytest.approx(0.5),
        "correction_abs_mean_frame_mean": 0.25,
    }


def test_model_spec_parser_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate model name"):
        parse_model_specs(["full=one.json,one.pt", "full=two.json,two.pt"])


def test_saturation_margin_contract_rejects_mismatched_models() -> None:
    with pytest.raises(ValueError, match="saturation_margin_dn"):
        validate_saturation_margin_dn({"candidate": 32, "full": 64})


def test_formal_cli_requires_candidate_and_full() -> None:
    with pytest.raises(ValueError, match="candidate.*full"):
        main(
            [
                "--model",
                "candidate=config.json,checkpoint.pt",
                "--sequence",
                "synthetic",
                "--frames",
                "1:1",
                "--output",
                "report.json",
            ]
        )

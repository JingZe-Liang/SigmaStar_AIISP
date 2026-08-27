from __future__ import annotations

import json
import numpy as np


def test_compare_against_denoised_reports_zero_for_identical_prediction(tmp_path) -> None:
    from raw_fusion.v2.data_first_compare import compare_against_denoised

    denoised = np.full((2, 4, 8, 8), 512, dtype=np.uint16)
    np.save(tmp_path / "inference_128x.npy", denoised)
    result = compare_against_denoised(tmp_path, {"128x": denoised}, tmp_path / "comparison.json")
    assert result.aggregate["mae"] == 0.0
    assert result.aggregate["rmse"] == 0.0
    assert result.aggregate["fallback_fraction"] == 0.0
    assert json.loads((tmp_path / "comparison.json").read_text())["protocol"] == "raw_fusion_v2_data_first_comparison"


def test_compare_reports_nonzero_error_and_temporal_statistics(tmp_path) -> None:
    from raw_fusion.v2.data_first_compare import compare_against_denoised

    denoised = np.zeros((2, 4, 8, 8), dtype=np.uint16)
    prediction = denoised.copy()
    prediction[1] = 100
    np.save(tmp_path / "inference_128x.npy", prediction)
    result = compare_against_denoised(tmp_path, {"128x": denoised}, tmp_path / "comparison.json")
    assert result.aggregate["mae"] > 0.0
    assert result.aggregate["temporal_prediction"] > 0.0


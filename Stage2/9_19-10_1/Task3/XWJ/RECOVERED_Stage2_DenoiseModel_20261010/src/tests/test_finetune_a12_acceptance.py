import numpy as np
import pytest

from raw_fusion.finetune.a12_acceptance import static_error_sums, stress_sums, summarize_static


def test_identity_and_bias_separate_detail_from_offset() -> None:
    target = np.zeros((4, 12, 12), np.float32)
    target[:, :, 6:] = 100
    valid = np.ones((12, 12), bool)
    identity = summarize_static(static_error_sums(target, target, valid))
    offset = summarize_static(static_error_sums(target + 5, target, valid))
    assert identity["rmse_dn"] == 0
    assert offset["bias_dn"] == 5
    assert offset["highpass_error_rmse_dn"] == 0
    assert offset["signed_gradient_error_rmse_dn"] == 0
    assert offset["lowpass_error_rmse_dn"] == 5


def test_mask_excludes_invalid_neighbourhood_from_highpass() -> None:
    target = np.zeros((4, 12, 12), np.float32)
    prediction = target.copy()
    prediction[:, 5, 5] = 1000
    valid = np.ones((12, 12), bool)
    valid[5, 5] = False
    result = summarize_static(static_error_sums(prediction, target, valid))
    assert result["highpass_error_rmse_dn"] == 0
    assert result["rmse_dn"] == 0


def test_shifted_edge_is_not_rewarded_for_equal_hf_energy() -> None:
    target = np.zeros((4, 12, 12), np.float32)
    target[:, :, 6:] = 100
    shifted = np.zeros_like(target)
    shifted[:, :, 7:] = 100
    result = summarize_static(static_error_sums(shifted, target, np.ones((12, 12), bool)))
    assert result["signed_gradient_error_rmse_dn"] > 0
    assert result["highpass_error_rmse_dn"] > 0


def test_empty_mask_is_missing_evidence_not_zero_error() -> None:
    frame = np.zeros((4, 12, 12), np.float32)
    assert summarize_static(static_error_sums(frame, frame, np.zeros((12, 12), bool))) == {"status": "no_valid_support", "values": 0}


def test_thin_detail_class_does_not_erode_valid_neighbours() -> None:
    target = np.zeros((4, 12, 12), np.float32)
    target[:, :, 6:] = 100
    region = np.zeros((12, 12), bool)
    region[:, 5] = True
    result = summarize_static(static_error_sums(target, target, np.ones((12, 12), bool), region))
    assert result["values"] == 40
    assert result["rmse_dn"] == 0


def test_nonfinite_prediction_rejected() -> None:
    target = np.zeros((4, 12, 12), np.float32)
    invalid = target.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        static_error_sums(invalid, target, np.ones((12, 12), bool))


def test_temporal_statistics_do_not_join_separate_clips() -> None:
    first = np.ones((8, 4, 12, 12), np.float32)
    second = first * 100
    assert stress_sums(first, first, first / 4095)["output_temporal_abs"] == 0
    assert stress_sums(second, second, second / 4095)["output_temporal_abs"] == 0
    assert stress_sums(np.concatenate([first, second]), np.concatenate([first, second]), np.concatenate([first, second]) / 4095)["output_temporal_abs"] > 0

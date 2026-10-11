"""Unit tests for the A0 adaptation table and filename metadata parser."""

import pytest

from raw_fusion.finetune.adapters import (
    BIT_SHIFT_HIGH_12,
    BIT_SHIFT_LOW_12,
    calibration_adapter,
    parse_filename_metadata,
    resolve_adapter,
    resolve_exposure_reference,
)

SCENE_NAME = (
    "raw_stream_1920x1080_16bit@RG_[Shutter=79999,SenserG=131072,IspG=5167,"
    "R=2120,G=1024,B=1956].raw"
)
CALIB_NAME = (
    "raw_stream_normal_1920x1080_16_GR_0205173829970_"
    "[FN=10,US=10400,AG=8192,DG=1025,BV=-25381,R=2316,G=1024,B=1983].raw"
)


def test_scene_filename_metadata() -> None:
    meta = parse_filename_metadata(SCENE_NAME)
    assert (meta.width, meta.height) == (1920, 1080)
    assert meta.cfa_name == "RG"
    assert meta.shutter_us == 79999
    assert meta.sensor_gain == 131072
    assert meta.isp_gain == 5167
    assert (meta.wb_r, meta.wb_g, meta.wb_b) == (2120, 1024, 1956)
    assert meta.frames_stated is None  # scene filenames carry no frame count
    assert meta.analog_gain is None


def test_calibration_filename_metadata() -> None:
    meta = parse_filename_metadata(CALIB_NAME)
    assert (meta.width, meta.height) == (1920, 1080)
    assert meta.cfa_name == "GR"
    assert meta.frames_stated == 10  # the filename lies (actual 50); recorded as stated only
    assert meta.shutter_us == 10400
    assert meta.analog_gain == 8192
    assert meta.digital_gain == 1025
    assert meta.bias == -25381
    assert meta.isp_gain is None


def test_adapters_per_source() -> None:
    reference = resolve_adapter("04", "shdarkroom", "10x")
    assert reference.role == "reference"
    assert reference.bit_shift == BIT_SHIFT_HIGH_12
    assert reference.cfa == "RGGB"

    noisy = resolve_adapter("04", "shdarkroom", "645x")
    assert noisy.role == "noisy"

    outdoor = resolve_adapter("04", "mis20s1_outdoor", "512x")
    assert outdoor.role == "noisy" and outdoor.bit_shift == BIT_SHIFT_HIGH_12

    calibration = calibration_adapter("color")
    assert calibration.cfa == "GRBG"  # calibration streams carry the flipped phase
    assert calibration_adapter("black").role == "calibration_black"

    auxiliary = resolve_adapter("03", "MIS20S1_2D3D", "645x")
    assert auxiliary.role == "auxiliary"
    assert auxiliary.bit_shift == BIT_SHIFT_LOW_12
    assert auxiliary.reader_kind == "frame_directory"


def test_adapter_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Phase C"):
        resolve_adapter("05", "whatever", "1024x")


def test_exposure_reference_scope() -> None:
    assert resolve_exposure_reference("04", "shdarkroom", "645x")
    assert not resolve_exposure_reference("04", "shdarkroom", "10x")  # the reference itself
    assert not resolve_exposure_reference("04", "mis20s1_outdoor", "128x")
    assert not resolve_exposure_reference("03", "MIS20S1_2D3D", "645x")

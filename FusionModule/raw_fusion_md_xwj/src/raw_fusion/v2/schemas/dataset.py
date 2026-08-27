from __future__ import annotations

from .common import *


def _target(value, context="TargetSequence"):
    m = validate_exact(value, {"noisy", "denoised", "fused", "preview"}, context)
    for key in ("noisy", "denoised", "fused"):
        validate_raw_asset_mapping(m[key], f"{context}.{key}")
    p = validate_exact(m["preview"], {"white_balance_rgb", "isp_gain"}, f"{context}.preview")
    if not isinstance(p["white_balance_rgb"], list) or not isinstance(p["isp_gain"], (int, float)):
        raise ContractError(f"{context}.preview: invalid ISP values")
    return m


def _support(value, context="SupportSequence"):
    m = validate_exact(value, {"noisy", "used_frame_range"}, context)
    validate_raw_asset_mapping(m["noisy"], f"{context}.noisy")
    if not isinstance(m["used_frame_range"], list) or len(m["used_frame_range"]) != 2 or any(not isinstance(x, int) for x in m["used_frame_range"]):
        raise ContractError(f"{context}.used_frame_range: expected [start,end]")
    return m


def validate_dataset_v2(value):
    m = validate_exact(value, {"schema_version", "protocol", "raw_contract", "target_sequences", "support_sequences"}, "DatasetV2")
    if m["schema_version"] != 2: raise ContractError("DatasetV2: schema_version must be 2")
    validate_protocol(m["protocol"], FUSION_PROTOCOL, "DatasetV2.protocol")
    raw = validate_exact(m["raw_contract"], {"sensor_width", "sensor_height", "dtype", "white_level", "cfa_pattern", "packed_order", "signals"}, "DatasetV2.raw_contract")
    if raw["sensor_width"] != 1920 or raw["sensor_height"] != 1080 or raw["dtype"] != "<u2" or raw["white_level"] != 4095 or raw["cfa_pattern"] != "RGGB":
        raise ContractError("DatasetV2.raw_contract: fixed 1920x1080 <u2 RGGB white=4095 contract required")
    if raw["packed_order"] != ["R", "Gr", "B", "Gb"]:
        raise ContractError("DatasetV2.raw_contract.packed_order must be [R, Gr, B, Gb]")
    signals = validate_exact(raw["signals"], {"noisy", "denoised", "fused", "prediction"}, "DatasetV2.raw_contract.signals")
    expected_signal = {
        "noisy": (4, 252), "denoised": (0, 300), "fused": (0, 300), "prediction": (0, 252),
    }
    for key in signals:
        sig = validate_exact(signals[key], {"right_shift", "black_level"}, f"DatasetV2.raw_contract.signals.{key}")
        expect_int(sig["right_shift"], f"...{key}.right_shift"); expect_int(sig["black_level"], f"...{key}.black_level")
        if (sig["right_shift"], sig["black_level"]) != expected_signal[key]:
            raise ContractError(f"DatasetV2.raw_contract.signals.{key}: unexpected shift/black level")
    targets = expect_mapping(m["target_sequences"], "DatasetV2.target_sequences")
    require_exact_keys(targets, {"128x", "645x"}, "DatasetV2.target_sequences")
    for key in targets:
        target = _target(targets[key], f"DatasetV2.target_sequences.{key}")
        for signal in ("noisy", "denoised", "fused"):
            if target[signal]["frame_count"] != 200:
                raise ContractError(f"DatasetV2.target_sequences.{key}.{signal}: frame_count must be 200")
    supports = expect_mapping(m["support_sequences"], "DatasetV2.support_sequences")
    require_exact_keys(supports, {"10x", "64x", "313x"}, "DatasetV2.support_sequences")
    expected_support_count = {"10x": 200, "64x": 200, "313x": 300}
    for key in supports:
        support = _support(supports[key], f"DatasetV2.support_sequences.{key}")
        if support["noisy"]["frame_count"] != expected_support_count[key]:
            raise ContractError(f"DatasetV2.support_sequences.{key}: unexpected frame_count")
        if support["used_frame_range"] != [0, 199]:
            raise ContractError(f"DatasetV2.support_sequences.{key}.used_frame_range must be [0,199]")
    return m


def _split_part(value, context):
    m = validate_exact(value, {"pre_roll_frames", "target_frames", "post_guard_frames", "references"}, context)
    for key in ("pre_roll_frames", "target_frames", "post_guard_frames"):
        if not isinstance(m[key], list) or any(not isinstance(x, int) for x in m[key]): raise ContractError(f"{context}.{key}: expected integer list")
    refs = validate_exact(m["references"], {"g", "e", "v"}, f"{context}.references")
    for key in refs:
        if not isinstance(refs[key], list) or len(refs[key]) != 2 or any(not isinstance(x, int) for x in refs[key]): raise ContractError(f"{context}.references.{key}: expected range")
    return m


def validate_split_v2(value):
    m = validate_exact(value, {"schema_version", "protocol", "train", "validation", "audit_rois"}, "SplitV2")
    if m["schema_version"] != 2: raise ContractError("SplitV2: schema_version must be 2")
    validate_protocol(m["protocol"], FUSION_PROTOCOL, "SplitV2.protocol")
    train = _split_part(m["train"], "SplitV2.train")
    validation = _split_part(m["validation"], "SplitV2.validation")
    if train["pre_roll_frames"] != [56, 57] or train["target_frames"] != [58, 93] or train["post_guard_frames"] != [94, 95]:
        raise ContractError("SplitV2.train: fixed frame ranges required")
    if validation["pre_roll_frames"] != [96, 97] or validation["post_guard_frames"] != [126, 127]:
        raise ContractError("SplitV2.validation: fixed pre-roll/post-guard ranges required")
    if validation["target_frames"][0] != 98 or validation["target_frames"][1] > 125:
        raise ContractError("frame 128 validation target would cross into train reference")
    if train["references"] != {"g": [128, 143], "e": [144, 159], "v": [160, 175]}:
        raise ContractError("SplitV2.train.references: fixed ranges required")
    if validation["references"] != {"g": [176, 183], "e": [184, 191], "v": [192, 199]}:
        raise ContractError("SplitV2.validation.references: fixed ranges required")
    rois = validate_exact(m["audit_rois"], {"grass", "wall"}, "SplitV2.audit_rois")
    for key in rois:
        roi = validate_exact(rois[key], {"x_half_open", "y_half_open"}, f"SplitV2.audit_rois.{key}")
        for axis in ("x_half_open", "y_half_open"):
            if not isinstance(roi[axis], list) or len(roi[axis]) != 2 or any(not isinstance(x, int) for x in roi[axis]): raise ContractError(f"{axis}: invalid range")
    return m

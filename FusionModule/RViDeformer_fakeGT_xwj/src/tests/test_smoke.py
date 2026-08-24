from __future__ import annotations

from pathlib import Path

import pytest

from raw_fusion.smoke import SmokeReport


def test_smoke_report_requires_all_acceptance_fields(tmp_path: Path) -> None:
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"png")
    report = SmokeReport(
        validation_passed=True,
        train_step_finite=True,
        initial_loss=1.0,
        final_loss=0.7,
        full_frame_bytes=4_147_200,
        full_frame_min=252,
        full_frame_max=4095,
        preview_path=preview_path,
    )
    assert report.loss_reduction == pytest.approx(0.3)
    assert report.accepted


def test_smoke_report_rejects_insufficient_overfit(tmp_path: Path) -> None:
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"png")
    report = SmokeReport(
        validation_passed=True,
        train_step_finite=True,
        initial_loss=1.0,
        final_loss=0.85,
        full_frame_bytes=4_147_200,
        full_frame_min=252,
        full_frame_max=4095,
        preview_path=preview_path,
    )
    assert not report.accepted


def test_smoke_report_rejects_invalid_output_range(tmp_path: Path) -> None:
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"png")
    report = SmokeReport(
        validation_passed=True,
        train_step_finite=True,
        initial_loss=1.0,
        final_loss=0.5,
        full_frame_bytes=4_147_200,
        full_frame_min=251,
        full_frame_max=4095,
        preview_path=preview_path,
    )
    assert not report.accepted

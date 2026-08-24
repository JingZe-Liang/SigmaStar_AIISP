from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from raw_fusion.preview import build_comparison_frame, encode_video_ffmpeg, main, simple_isp


def _constant_rggb(red: int, green: int, blue: int) -> np.ndarray:
    raw = np.empty((8, 8), dtype=np.uint16)
    raw[0::2, 0::2] = red
    raw[0::2, 1::2] = green
    raw[1::2, 0::2] = green
    raw[1::2, 1::2] = blue
    return raw


def test_simple_isp_keeps_red_plane_in_rgb_red_channel_without_mutating_input() -> None:
    raw = _constant_rggb(red=3000, green=1000, blue=500)
    original = raw.copy()

    rgb = simple_isp(
        raw,
        black=0,
        white=4095,
        white_balance=(1.0, 1.0, 1.0),
        exposure=1.0,
    )

    assert np.array_equal(raw, original)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (8, 8, 3)
    assert rgb[..., 0].mean() > rgb[..., 1].mean() > rgb[..., 2].mean()
    assert np.array_equal(
        rgb,
        simple_isp(raw, 0, 4095, (1.0, 1.0, 1.0), 1.0),
    )


@pytest.mark.parametrize(
    ("raw", "black", "white", "white_balance", "exposure"),
    [
        (np.zeros((8, 8, 1), dtype=np.uint16), 0, 4095, (1.0, 1.0, 1.0), 1.0),
        (np.zeros((8, 8), dtype=np.uint8), 0, 4095, (1.0, 1.0, 1.0), 1.0),
        (np.zeros((7, 8), dtype=np.uint16), 0, 4095, (1.0, 1.0, 1.0), 1.0),
        (np.full((8, 8), 5000, dtype=np.uint16), 0, 4095, (1.0, 1.0, 1.0), 1.0),
        (np.zeros((8, 8), dtype=np.uint16), 4095, 4095, (1.0, 1.0, 1.0), 1.0),
        (np.zeros((8, 8), dtype=np.uint16), 0, 65535, (1.0, 1.0, 1.0), 1.0),
        (np.zeros((8, 8), dtype=np.uint16), 0, 4095, (0.0, 1.0, 1.0), 1.0),
        (np.zeros((8, 8), dtype=np.uint16), 0, 4095, (1.0, 1.0, 1.0), float("inf")),
    ],
)
def test_simple_isp_rejects_invalid_inputs(
    raw: np.ndarray,
    black: int,
    white: int,
    white_balance: tuple[float, float, float],
    exposure: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        simple_isp(raw, black, white, white_balance, exposure)


def test_comparison_frame_preserves_insert_order_and_rejects_mismatched_geometry() -> None:
    first = np.full((8, 12, 3), 10, np.uint8)
    second = np.full((8, 12, 3), 20, np.uint8)
    comparison = build_comparison_frame({"denoised": first, "fused": second})

    assert comparison.shape == (8, 24, 3)
    assert np.all(comparison[:, :12] == 10)
    assert np.all(comparison[:, 12:] == 20)
    with pytest.raises(ValueError, match="same shape"):
        build_comparison_frame({"a": first, "b": np.zeros((7, 12, 3), np.uint8)})


def test_encode_video_ffmpeg_builds_required_command_and_propagates_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pattern = tmp_path / "frame_%04d.png"
    output = tmp_path / "comparison.mp4"
    commands: list[list[str]] = []

    def successful_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        output.touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("raw_fusion.preview.subprocess.run", successful_run)
    encode_video_ffmpeg(pattern, output, 30)
    assert commands == [[
        "ffmpeg", "-y", "-framerate", "30", "-i", str(pattern), "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", str(output),
    ]]

    monkeypatch.setattr(
        "raw_fusion.preview.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, "", "encoder failure"),
    )
    with pytest.raises(RuntimeError, match="encoder failure"):
        encode_video_ffmpeg(pattern, output, 30)


def test_preview_cli_writes_one_synthetic_frame(tmp_path: Path) -> None:
    raw_path = tmp_path / "frame.raw"
    output_path = tmp_path / "nested" / "frame.png"
    _constant_rggb(3000, 1000, 500).tofile(raw_path)

    main(
        [
            "--input", str(raw_path),
            "--output", str(output_path),
            "--width", "8",
            "--height", "8",
            "--black", "0",
            "--white", "4095",
            "--white-balance", "1,1,1",
            "--exposure", "1",
        ]
    )

    assert output_path.is_file()
    assert cv2.imread(str(output_path), cv2.IMREAD_COLOR).shape == (8, 8, 3)

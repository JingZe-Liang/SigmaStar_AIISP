from pathlib import Path

import numpy as np
import pytest

from raw_fusion.raw import (
    RawFrameDirectoryReader,
    RawStreamReader,
    normalize_raw,
    pack_rggb,
    quantize_normalized,
    unpack_rggb,
)


def test_pack_rggb_uses_r_gr_b_gb_order() -> None:
    mosaic = np.array([[10, 20, 11, 21], [40, 30, 41, 31]], dtype=np.uint16)

    packed = pack_rggb(mosaic)

    np.testing.assert_array_equal(packed[:, 0, 0], np.array([10, 20, 30, 40]))
    np.testing.assert_array_equal(unpack_rggb(packed), mosaic)


def test_noisy_shift_precedes_normalization() -> None:
    stored = np.array([[252 << 4, 4095 << 4]], dtype="<u2")

    normalized = normalize_raw(stored >> 4, 252, 4095)

    np.testing.assert_allclose(normalized, np.array([[0.0, 1.0]], dtype=np.float32))


def test_stream_reader_rejects_partial_frame(tmp_path: Path) -> None:
    path = tmp_path / "partial.raw"
    path.write_bytes(b"\x00" * 15)

    with pytest.raises(ValueError, match="期望字节数"):
        RawStreamReader(path, width=2, height=2, frame_count=2, shift=0)


def test_quantization_is_little_endian_and_right_aligned() -> None:
    raw = quantize_normalized(np.array([[0.0, 1.0]], np.float32), 252, 4095)

    assert raw.dtype == np.dtype("<u2")
    assert raw.tolist() == [[252, 4095]]


def test_quantization_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="非有限"):
        quantize_normalized(np.array([[np.nan, np.inf, -np.inf]], np.float32), 252, 4095)


def test_readers_reject_shift_16(tmp_path: Path) -> None:
    path = tmp_path / "stream.raw"
    path.write_bytes(np.zeros(4, dtype="<u2").tobytes())

    with pytest.raises(ValueError, match="shift"):
        RawStreamReader(path, width=2, height=2, frame_count=1, shift=16)
    with pytest.raises(ValueError, match="shift"):
        RawFrameDirectoryReader(tmp_path, pattern="frame_{index}.raw", width=2, height=2, shift=16)


def test_directory_reader_rejects_negative_indices(tmp_path: Path) -> None:
    reader = RawFrameDirectoryReader(
        tmp_path, pattern="frame_{index}.raw", width=2, height=2, shift=0
    )

    with pytest.raises(IndexError, match="negative"):
        reader.read_frame(-1)
    with pytest.raises(IndexError, match="negative"):
        reader.read_crop(-1, 0, 0, 1, 1)


def test_stream_reader_shifts_and_returns_independent_arrays(tmp_path: Path) -> None:
    path = tmp_path / "stream.raw"
    values = np.arange(12, dtype="<u2").reshape(3, 2, 2) << 2
    path.write_bytes(values.tobytes())
    reader = RawStreamReader(path, width=2, height=2, frame_count=3, shift=2)

    frame = reader.read_frame(1)
    crop = reader.read_crop(2, 0, 1, 2, 1)
    expected_frame = frame.copy()
    frame[0, 0] = 999

    np.testing.assert_array_equal(expected_frame, values[1] >> 2)
    np.testing.assert_array_equal(crop, (values[2] >> 2)[:, 1:2])
    assert type(frame) is np.ndarray
    assert type(crop) is np.ndarray
    assert not np.shares_memory(frame, reader._frames)
    assert not np.shares_memory(crop, reader._frames)


def test_stream_reader_rejects_out_of_bounds_reads(tmp_path: Path) -> None:
    path = tmp_path / "stream.raw"
    path.write_bytes(np.zeros(2 * 2 * 2, dtype="<u2").tobytes())
    reader = RawStreamReader(path, width=2, height=2, frame_count=2, shift=0)

    with pytest.raises(IndexError):
        reader.read_frame(2)
    with pytest.raises(ValueError):
        reader.read_crop(0, 1, 0, 2, 2)


def test_frame_directory_reader_formats_index_and_validates_file_size(
    tmp_path: Path,
) -> None:
    first = np.array([[1, 2], [3, 4]], dtype="<u2")
    second = np.array([[5, 6], [7, 8]], dtype="<u2")
    (tmp_path / "frame_0003.raw").write_bytes(first.tobytes())
    (tmp_path / "frame_0004.raw").write_bytes(second.tobytes())
    reader = RawFrameDirectoryReader(
        tmp_path, pattern="frame_{index:04d}.raw", width=2, height=2, shift=1
    )

    np.testing.assert_array_equal(reader.read_frame(3), first >> 1)
    np.testing.assert_array_equal(reader.read_crop(4, 0, 0, 1, 1), (second >> 1)[:1, :1])

    (tmp_path / "frame_0005.raw").write_bytes(b"\x00" * 7)
    with pytest.raises(ValueError, match="期望字节数"):
        RawFrameDirectoryReader(
            tmp_path, pattern="frame_{index:04d}.raw", width=2, height=2, shift=0
        ).read_frame(5)

"""Stream reader and packing tests: synthetic packing logic + real-file reads.

Real-file tests read the dataset strictly read-only (memmap mode='r').
"""

from pathlib import Path

import numpy as np

from raw_fusion.finetune.adapters import calibration_adapter, resolve_adapter
from raw_fusion.finetune.stream_io import open_stream, pack_frame, pack_frames

DATASET = Path("/HardDisk/jingzeliang/Data/Company_Dataset/dataset")
SHDARKROOM = DATASET / "04_Sigmastar_7_30" / "shdarkroom"
GROUP_03 = DATASET / "03_MIS20S1_2D3D"


def _gain_dir(gain: str) -> Path:
    return SHDARKROOM / gain


def _single_raw(directory: Path) -> Path:
    return sorted(directory.glob("*.raw"))[0]


def test_pack_rggb_phase() -> None:
    # Synthetic RGGB frame: each 2x2 cell constant (r, gr, gb, b), built in the
    # raw high-aligned domain (dn12 << 4) so the adapter's shift recovers DN.
    r, gr, gb, b = 100, 200, 300, 400
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[0::2, 0::2] = r << 4
    frame[0::2, 1::2] = gr << 4
    frame[1::2, 0::2] = gb << 4
    frame[1::2, 1::2] = b << 4
    adapter = resolve_adapter("04", "shdarkroom", "10x")
    packed = pack_frame(frame, adapter)
    assert packed.shape == (4, 3, 4)
    np.testing.assert_array_equal(packed[0], r)
    np.testing.assert_array_equal(packed[1], gr)
    np.testing.assert_array_equal(packed[2], gb)
    np.testing.assert_array_equal(packed[3], b)


def test_pack_grbg_phase_flips_to_canonical() -> None:
    # Calibration sensors are GRBG: row0 = Gr R, row1 = B Gb.
    r, gr, gb, b = 10, 20, 30, 40
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[0::2, 0::2] = gr << 4
    frame[0::2, 1::2] = r << 4
    frame[1::2, 0::2] = b << 4
    frame[1::2, 1::2] = gb << 4
    adapter = calibration_adapter("color")
    packed = pack_frame(frame, adapter)
    np.testing.assert_array_equal(packed[0], r)  # canonical R
    np.testing.assert_array_equal(packed[1], gr)
    np.testing.assert_array_equal(packed[2], gb)
    np.testing.assert_array_equal(packed[3], b)


def test_pack_applies_bit_shift() -> None:
    frame = np.full((4, 8), 4095 << 4, dtype=np.uint16)
    adapter = resolve_adapter("04", "shdarkroom", "10x")
    packed = pack_frames(frame[None], adapter)
    assert int(packed.max()) == 4095  # high-aligned stream lands on 12-bit DN


def test_reader_rejects_undivisible_file(tmp_path: Path) -> None:
    from raw_fusion.finetune.stream_io import RawStreamReader

    bad = tmp_path / "bad.raw"
    bad.write_bytes(b"\x00" * (8 * 8 * 2 + 3))
    try:
        RawStreamReader(bad, 8, 8)
    except ValueError as error:
        assert "not a multiple" in str(error)
    else:
        raise AssertionError("expected ValueError for undivisible file size")


def test_real_shdarkroom_reader() -> None:
    adapter = resolve_adapter("04", "shdarkroom", "10x")
    reader, meta = open_stream(_single_raw(_gain_dir("10x")), adapter)
    assert (meta.width, meta.height) == (1920, 1080)
    assert reader.frames == 200  # verified by byte division in the survey
    frame = reader.read_frame(0)
    assert frame.shape == (1080, 1920) and frame.dtype == np.uint16
    # High alignment: sampled raw values must be exact multiples of 16.
    sample = frame[::17, ::19].ravel()
    assert bool(np.all(sample % 16 == 0))
    packed = pack_frame(frame, adapter)
    assert packed.shape == (4, 540, 960)
    chunks = list(reader.iter_chunks(64))
    assert [c.shape[0] for c in chunks] == [64, 64, 64, 8]


def test_real_03_frame_directory_reader() -> None:
    adapter = resolve_adapter("03", "MIS20S1_2D3D", "645x")
    variant_dir = next(GROUP_03.glob("*645x/denoised"))
    reader, meta = open_stream(variant_dir, adapter)
    assert (meta.width, meta.height) == (1920, 1080)
    assert reader.frames == 200
    frame = reader.read_frame(7)
    assert frame.shape == (1080, 1920)
    # Low alignment: values must NOT all be multiples of 16.
    sample = frame[::17, ::19].ravel()
    assert not bool(np.all(sample % 16 == 0))

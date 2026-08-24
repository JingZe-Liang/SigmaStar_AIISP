from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from raw_fusion.config import DatasetConfig, RawLayout, SequenceConfig
import raw_fusion.data as data_module
from raw_fusion.data import (
    FusionPatchDataset,
    build_frame_indices,
    validate_dataset,
)


_PATCH_PLANES = np.array(
    [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        [[100, 101, 102, 103], [104, 105, 106, 107], [108, 109, 110, 111], [112, 113, 114, 115]],
        [[200, 201, 202, 203], [204, 205, 206, 207], [208, 209, 210, 211], [212, 213, 214, 215]],
        [[300, 301, 302, 303], [304, 305, 306, 307], [308, 309, 310, 311], [312, 313, 314, 315]],
    ],
    dtype="<u2",
)


@pytest.fixture
def synthetic_sequence(tmp_path: Path) -> Path:
    return _write_patch_sequence(tmp_path / "sequence")


@pytest.fixture
def validation_sequence(tmp_path: Path) -> Path:
    return _write_validation_sequence(tmp_path / "validation", frame_count=200)


@pytest.fixture
def dataset_config(validation_sequence: Path) -> DatasetConfig:
    return _make_config(validation_sequence, frame_count=200)


def _write_patch_sequence(root: Path) -> Path:
    root.mkdir()
    target_dir = root / "target"
    target_dir.mkdir()
    for index in range(3):
        noisy = _mosaic_from_packed(_PATCH_PLANES + 252 + index * 10)
        denoised = _mosaic_from_packed(_PATCH_PLANES + 300 + index * 20)
        fused = _mosaic_from_packed(_PATCH_PLANES + 300 + index * 30)
        target = _mosaic_from_packed(_PATCH_PLANES + 252 + index * 40)
        _append_frame(root / "noisy.raw", np.left_shift(noisy, 4))
        _append_frame(root / "denoised.raw", denoised)
        _append_frame(root / "fused.raw", fused)
        target.tofile(target_dir / f"out_{index:04d}.raw")
    return root


def _write_validation_sequence(root: Path, *, frame_count: int) -> Path:
    root.mkdir()
    target_dir = root / "target"
    target_dir.mkdir()
    for index in range(frame_count):
        target = np.full((8, 8), 252 + index, dtype="<u2")
        noisy = np.full((8, 8), (20 + index) << 4, dtype="<u2")
        denoised = np.asarray(target + 48, dtype="<u2")
        fused = np.asarray(target + 49, dtype="<u2")
        _append_frame(root / "noisy.raw", noisy)
        _append_frame(root / "denoised.raw", denoised)
        _append_frame(root / "fused.raw", fused)
        target.tofile(target_dir / f"out_{index:04d}.raw")
    return root


def test_build_frame_indices_is_closed_interval() -> None:
    assert build_frame_indices((1, 3)) == (1, 2, 3)
    with pytest.raises(ValueError, match="闭区间"):
        build_frame_indices((3, 1))


def test_dataset_pairs_exact_previous_and_current_noisy_by_index(synthetic_sequence: Path) -> None:
    dataset = make_dataset(
        synthetic_sequence, frame_range=(1, 1), samples_per_epoch=1, force_transform=(False, False, False)
    )
    sample = dataset[0]

    assert sample["frame_index"].item() == 1
    _assert_sample_signals(sample, _expected_frame_one())

def test_non_full_packed_crop_preserves_all_cfa_planes_and_transforms(synthetic_sequence: Path) -> None:
    dataset = make_dataset(
        synthetic_sequence,
        frame_range=(1, 1),
        patch_size=2,
        seed=9,
        force_transform=(True, True, True),
    )
    sample = dataset[0]

    assert sample["frame_index"].item() == 1
    _assert_sample_signals(sample, _expected_offset_crop_after_all_transforms())


@pytest.mark.parametrize(
    "force_transform",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
    ids=("horizontal", "vertical", "transpose", "combined"),
)
def test_all_signals_receive_exact_identical_spatial_transform(
    synthetic_sequence: Path, force_transform: tuple[bool, bool, bool]
) -> None:
    dataset = make_dataset(
        synthetic_sequence, frame_range=(1, 1), force_transform=force_transform
    )
    sample = dataset[0]
    expected = {
        signal: _transform_expected(value, force_transform)
        for signal, value in _expected_frame_one().items()
    }

    _assert_sample_signals(sample, expected)


def test_epoch_and_index_make_sampling_deterministic(synthetic_sequence: Path) -> None:
    first = make_dataset(synthetic_sequence, seed=19)
    second = make_dataset(synthetic_sequence, seed=19)
    first.set_epoch(3)
    second.set_epoch(3)

    assert torch.equal(first[11]["curr_noisy"], second[11]["curr_noisy"])


def test_dataset_rejects_frame_zero_to_preserve_causality(synthetic_sequence: Path) -> None:
    with pytest.raises(ValueError, match="至少为 1"):
        make_dataset(synthetic_sequence, frame_range=(0, 1))


def test_validation_reports_all_signal_ranges(dataset_config: DatasetConfig) -> None:
    report = validate_dataset(dataset_config)
    sequence = report.sequences["synthetic"]

    assert sequence.frame_counts == {"noisy": 200, "denoised": 200, "fused": 200, "target": 200}
    assert sequence.checked_indices == (0, 1, 99, 100, 198, 199)
    assert sequence.minimums["noisy"] == 20
    assert sequence.maximums["target"] == 451


def test_target_file_validation_returns_verified_count(dataset_config: DatasetConfig) -> None:
    sequence = dataset_config.sequences["synthetic"]

    assert data_module._validate_all_target_files(dataset_config.layout, sequence) == 200



def test_validation_detects_nonzero_noisy_low_bits(dataset_config: DatasetConfig) -> None:
    corrupt_one_noisy_sample(dataset_config, value=257)

    with pytest.raises(ValueError, match="低 4 bit"):
        validate_dataset(dataset_config)


def test_validation_rejects_missing_non_sampled_target_frame(dataset_config: DatasetConfig) -> None:
    (dataset_config.sequences["synthetic"].pseudo_gt_dir / "out_0050.raw").unlink()

    with pytest.raises(ValueError, match="伪 GT"):
        validate_dataset(dataset_config)


def test_validation_rejects_missized_non_sampled_target_frame(dataset_config: DatasetConfig) -> None:
    (dataset_config.sequences["synthetic"].pseudo_gt_dir / "out_0050.raw").write_bytes(b"\0")

    with pytest.raises(ValueError, match="期望字节数"):
        validate_dataset(dataset_config)


def make_dataset(
    synthetic_sequence: Path,
    *,
    frame_range: tuple[int, int] = (1, 2),
    patch_size: int = 4,
    samples_per_epoch: int = 32,
    seed: int = 0,
    force_transform: tuple[bool, bool, bool] | None = None,
) -> FusionPatchDataset:
    return FusionPatchDataset(
        _make_config(synthetic_sequence, frame_count=3),
        sequence_name="synthetic",
        frame_range=frame_range,
        patch_size_packed=patch_size,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        force_transform=force_transform,
    )


def corrupt_one_noisy_sample(config: DatasetConfig, value: int) -> None:
    with config.sequences["synthetic"].noisy_stream.open("r+b") as stream:
        stream.write(np.asarray([value], dtype="<u2").tobytes())


def _make_config(root: Path, *, frame_count: int) -> DatasetConfig:
    return DatasetConfig(
        layout=RawLayout(
            width=8,
            height=8,
            frame_count=frame_count,
            dtype="<u2",
            white_level=4095,
            noisy_black_level=252,
            candidate_black_level=300,
            target_black_level=252,
            noisy_shift=4,
            cfa_pattern="RGGB",
            pseudo_gt_pattern="out_{index:04d}.raw",
        ),
        sequences={
            "synthetic": SequenceConfig(
                name="synthetic",
                noisy_stream=root / "noisy.raw",
                denoised_stream=root / "denoised.raw",
                fused_stream=root / "fused.raw",
                pseudo_gt_dir=root / "target",
                white_balance=(1.0, 1.0, 1.0),
                isp_gain=1.0,
            )
        },
    )


def _append_frame(path: Path, frame: np.ndarray) -> None:
    with path.open("ab") as stream:
        stream.write(frame.tobytes())


def _mosaic_from_packed(packed: np.ndarray) -> np.ndarray:
    mosaic = np.empty((8, 8), dtype="<u2")
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 1::2] = packed[2]
    mosaic[1::2, 0::2] = packed[3]
    return mosaic


def _expected_frame_one() -> dict[str, torch.Tensor]:
    return {
        "prev_noisy": torch.tensor(_PATCH_PLANES, dtype=torch.float32) / 3843.0,
        "curr_noisy": torch.tensor(_PATCH_PLANES + 10, dtype=torch.float32) / 3843.0,
        "denoised": torch.tensor(_PATCH_PLANES + 20, dtype=torch.float32) / 3795.0,
        "fused": torch.tensor(_PATCH_PLANES + 30, dtype=torch.float32) / 3795.0,
        "target": torch.tensor(_PATCH_PLANES + 40, dtype=torch.float32) / 3843.0,
    }


def _expected_offset_crop_after_all_transforms() -> dict[str, torch.Tensor]:
    # Literal [R, Gr, B, Gb] values for crop (packed_top=1, packed_left=2)
    # after horizontal, vertical, and transpose transforms.
    base = torch.tensor(
        [
            [[11, 7], [10, 6]],
            [[111, 107], [110, 106]],
            [[211, 207], [210, 206]],
            [[311, 307], [310, 306]],
        ],
        dtype=torch.float32,
    )
    return {
        "prev_noisy": base / 3843.0,
        "curr_noisy": (base + 10.0) / 3843.0,
        "denoised": (base + 20.0) / 3795.0,
        "fused": (base + 30.0) / 3795.0,
        "target": (base + 40.0) / 3843.0,
    }


def _transform_expected(
    expected: torch.Tensor, force_transform: tuple[bool, bool, bool]
) -> torch.Tensor:
    horizontal, vertical, transpose = force_transform
    transformed = expected
    if horizontal:
        transformed = transformed.flip(-1)
    if vertical:
        transformed = transformed.flip(-2)
    if transpose:
        transformed = transformed.transpose(-2, -1)
    return transformed


def _assert_sample_signals(sample: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for signal, value in expected.items():
        torch.testing.assert_close(sample[signal], value, rtol=0.0, atol=1e-7)

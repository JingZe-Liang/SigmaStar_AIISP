from __future__ import annotations

import numpy as np

from data.company_raw import CompanyRawPatchDataset
from noise_model.raw_io import RawSpec, encode_raw16


def test_company_raw_dataset_decodes_left_aligned_dn12(tmp_path):
    spec = RawSpec(width=8, height=6, cfa="RGGB")
    dn12 = np.arange(2 * spec.height * spec.width, dtype=np.float32).reshape(2, spec.height, spec.width)
    path = tmp_path / "stream.raw"
    encode_raw16(dn12, spec).tofile(path)

    dataset = CompanyRawPatchDataset(path, spec, patch_size=2, samples_per_epoch=2, deterministic=True)
    patch, location = dataset[0]

    assert patch.shape == (4, 2, 2)
    assert patch.dtype == np.float32
    assert 0 <= location["top"] <= spec.height // 2 - 2
    assert 0 <= location["left"] <= spec.width // 2 - 2
    # RGGB packing has R at [0, 0] and B at [1, 1] of each Bayer quad.
    assert patch[0, 0, 0] == dn12[0, 0, 0]
    assert patch[3, 0, 0] == dn12[0, 1, 1]

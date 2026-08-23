import unittest
from pathlib import Path

import numpy as np

from dataset_io import (
    AllSourceFramesDataset,
    PairedFusionDataset,
    RawStreamReader,
    discover_dataset,
    pack_bayer,
    validate_dataset,
)


DATASET_ROOT = Path(__file__).resolve().parent / "DATASET"


class BayerPackingTests(unittest.TestCase):
    def test_rggb_pack_and_odd_crop_preserve_global_channels(self):
        tile = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        mosaic = np.tile(tile, (2, 2))
        expected = np.stack(
            [
                np.full((2, 2), 1, dtype=np.uint16),
                np.full((2, 2), 2, dtype=np.uint16),
                np.full((2, 2), 3, dtype=np.uint16),
                np.full((2, 2), 4, dtype=np.uint16),
            ]
        )
        np.testing.assert_array_equal(pack_bayer(mosaic, "RGGB"), expected)

        odd_origin_crop = np.tile(tile, (2, 3))[:, 1:5]
        np.testing.assert_array_equal(
            pack_bayer(odd_origin_crop, "RGGB", origin=(0, 1)), expected
        )

    def test_grbg_pack_uses_consistent_r_g1_g2_b_order(self):
        mosaic = np.tile(np.array([[2, 1], [4, 3]], dtype=np.uint16), (2, 2))
        packed = pack_bayer(mosaic, "GRBG")
        self.assertEqual([int(channel[0, 0]) for channel in packed], [1, 2, 3, 4])


@unittest.skipUnless(DATASET_ROOT.is_dir(), "Phase2 dataset is not present")
class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = discover_dataset(DATASET_ROOT)

    def test_complete_catalog_counts(self):
        non_archive_files = [
            path for path in self.catalog.physical_files if path.suffix.lower() != ".zip"
        ]
        self.assertEqual(len(non_archive_files), 1634)
        self.assertEqual(
            len(self.catalog.physical_files),
            1634 + int(self.catalog.archive_path is not None),
        )
        self.assertEqual(len(self.catalog.source_streams), 29)
        self.assertEqual(self.catalog.source_frame_count, 3193)
        self.assertEqual(len(self.catalog.fusion_sequences), 2)
        self.assertEqual(self.catalog.paired_frame_count, 400)

    def test_shallow_validation(self):
        report = validate_dataset(self.catalog)
        self.assertTrue(report["ok"])
        self.assertFalse(report["errors"])

    def test_source_conversion_is_exact_integer_shift(self):
        spec = self.catalog.fusion_sequences[0].source
        reader = RawStreamReader(spec)
        raw16 = reader.read_frame(0, crop=(0, 0, 32, 32), convert_to_12bit=False)
        raw12 = reader.read_frame(0, crop=(0, 0, 32, 32))
        np.testing.assert_array_equal(raw12, np.right_shift(raw16, 4))

    def test_all_source_dataset_has_every_frame(self):
        dataset = AllSourceFramesDataset(self.catalog)
        self.assertEqual(len(dataset), 3193)
        first = dataset[0]
        last = dataset[-1]
        self.assertEqual(first["frame_index"], 0)
        self.assertEqual(
            last["frame_index"], self.catalog.source_streams[-1].frame_count - 1
        )

    def test_paired_dataset_has_all_400_frames(self):
        dataset = PairedFusionDataset(self.catalog, crop_size=64)
        self.assertEqual(len(dataset), 400)
        for item in (dataset[0], dataset[-1]):
            self.assertEqual(tuple(item["source"].shape), (4, 32, 32))
            self.assertEqual(tuple(item["denoised"].shape), (4, 32, 32))
            self.assertEqual(tuple(item["fused"].shape), (4, 32, 32))
            self.assertGreaterEqual(float(item["source"].min()), 0.0)
            self.assertLessEqual(float(item["source"].max()), 4095.0)

    def test_temporal_dataset_only_drops_unavailable_boundaries(self):
        dataset = PairedFusionDataset(
            self.catalog, crop_size=64, temporal_radius=1
        )
        self.assertEqual(len(dataset), 396)
        item = dataset[0]
        self.assertEqual(item["frame_index"], 1)
        self.assertIn("source_prev", item)
        self.assertIn("source_next", item)


if __name__ == "__main__":
    unittest.main()

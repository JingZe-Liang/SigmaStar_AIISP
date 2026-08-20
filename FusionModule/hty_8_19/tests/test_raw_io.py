import unittest

import numpy as np

from dnr_fusion.raw_io import normalize_raw, pack_rggb, unpack_rggb


class RawIoTests(unittest.TestCase):
    def test_rggb_pack_round_trip(self) -> None:
        mosaic = np.arange(8 * 10, dtype=np.uint16).reshape(8, 10)
        packed = pack_rggb(mosaic)
        self.assertEqual(packed.shape, (4, 4, 5))
        np.testing.assert_array_equal(unpack_rggb(packed), mosaic)

    def test_channel_order(self) -> None:
        mosaic = np.zeros((4, 4), dtype=np.uint16)
        mosaic[0::2, 0::2] = 10
        mosaic[0::2, 1::2] = 20
        mosaic[1::2, 1::2] = 30
        mosaic[1::2, 0::2] = 40
        packed = pack_rggb(mosaic)
        self.assertEqual([int(channel[0, 0]) for channel in packed], [10, 20, 30, 40])

    def test_normalization_preserves_small_negative_values(self) -> None:
        packed = np.array([[[290, 300, 4095]]], dtype=np.uint16)
        normalized = normalize_raw(packed, black=300, white=4095)
        self.assertLess(float(normalized[0, 0, 0]), 0.0)
        self.assertEqual(float(normalized[0, 0, 1]), 0.0)
        self.assertAlmostEqual(float(normalized[0, 0, 2]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()


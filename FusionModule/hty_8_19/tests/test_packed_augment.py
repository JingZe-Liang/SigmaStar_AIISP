import unittest

import numpy as np
import torch

from dnr_fusion.packed_augment import _horizontal, _rotate, _vertical


class PackedAugmentTests(unittest.TestCase):
    def setUp(self):
        # Channel constants make CFA permutations directly observable.
        self.item = torch.stack(
            [torch.full((2, 3), float(index)) for index in range(4)]
        )

    def test_horizontal_swaps_cfa_columns(self):
        result = _horizontal(self.item)
        self.assertEqual(result[:, 0, 0].tolist(), [1.0, 0.0, 3.0, 2.0])

    def test_vertical_swaps_cfa_rows(self):
        result = _vertical(self.item)
        self.assertEqual(result[:, 0, 0].tolist(), [3.0, 2.0, 1.0, 0.0])

    def test_rotation_permutation(self):
        result = _rotate(self.item, 1)
        self.assertEqual(result[:, 0, 0].tolist(), [1.0, 2.0, 3.0, 0.0])


if __name__ == "__main__":
    unittest.main()

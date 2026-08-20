import unittest

import torch

from dnr_fusion.confidence import SafetyParams
from dnr_fusion.features_v2 import build_threshold_normalized_features
from dnr_fusion.infer_v2 import stabilize_gate


class V2SafetyTests(unittest.TestCase):
    def test_gate_falls_immediately_and_rises_slowly(self):
        previous = torch.full((1, 1, 2, 2), 0.8)
        lower = torch.full_like(previous, 0.2)
        higher = torch.full_like(previous, 1.0)
        hard = torch.zeros_like(previous)
        fallen = stabilize_gate(lower, previous, hard, rise_alpha=0.08, fall_alpha=1.0)
        risen = stabilize_gate(higher, previous, hard, rise_alpha=0.08, fall_alpha=1.0)
        self.assertTrue(torch.allclose(fallen, lower))
        self.assertTrue(torch.allclose(risen, torch.full_like(previous, 0.816)))

    def test_hard_mask_always_zero(self):
        previous = torch.full((1, 1, 2, 2), 0.8)
        instantaneous = torch.full_like(previous, 0.9)
        hard = torch.zeros_like(previous)
        hard[:, :, 0, 1] = 1
        result = stabilize_gate(instantaneous, previous, hard, rise_alpha=0.08, fall_alpha=1.0)
        self.assertEqual(float(result[0, 0, 0, 1]), 0.0)

    def test_threshold_normalized_features_have_expected_scale(self):
        params = SafetyParams(
            motion_threshold_dn=10.0,
            disagreement_threshold_dn=20.0,
            dynamic_range_dn=1000.0,
        )
        source = torch.zeros(1, 4, 2, 2)
        denoised = torch.zeros_like(source)
        fused = torch.full_like(source, 80.0 / 1000.0)
        previous = torch.zeros_like(source)
        features = build_threshold_normalized_features(
            source, denoised, fused, previous, params
        )
        # Candidate delta is 80/(4*20) = 1 and is clipped to one.
        self.assertTrue(torch.allclose(features[:, 16:20], torch.ones_like(features[:, 16:20])))
        self.assertEqual(tuple(features.shape), (1, 24, 2, 2))


if __name__ == "__main__":
    unittest.main()

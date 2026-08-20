import unittest

import torch

from dnr_fusion.confidence import (
    SafetyParams,
    conservative_gate,
    fuse_candidates,
    safety_confidence,
)
from dnr_fusion.model import SafeGateUNet


class SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = SafetyParams(
            motion_threshold_dn=10.0,
            disagreement_threshold_dn=50.0,
            dynamic_range_dn=3795.0,
            local_kernel=5,
            dilation_kernel=9,
        )

    def test_motion_region_is_hard_fallback(self) -> None:
        previous = torch.full((1, 4, 32, 32), 0.1)
        current = previous.clone()
        current[:, :, 12:20, 12:20] = 0.5
        candidate = current + 0.03
        confidence, diagnostics = safety_confidence(
            current, previous, candidate, self.params
        )
        self.assertEqual(float(confidence[0, 0, 16, 16]), 0.0)
        self.assertEqual(float(diagnostics["hard_mask"][0, 0, 16, 16]), 1.0)

        predicted = torch.ones_like(confidence)
        gate = conservative_gate(predicted, confidence)
        output = fuse_candidates(current, candidate, gate)
        torch.testing.assert_close(output[:, :, 16, 16], current[:, :, 16, 16])

    def test_static_region_allows_temporal_candidate(self) -> None:
        current = torch.full((1, 4, 32, 32), 0.1)
        candidate = current + 0.001
        confidence, diagnostics = safety_confidence(
            current, current, candidate, self.params
        )
        self.assertGreater(float(confidence.mean()), 0.8)
        self.assertEqual(float(diagnostics["hard_mask"].sum()), 0.0)

    def test_model_output_shape_and_safe_initialization(self) -> None:
        model = SafeGateUNet(input_channels=24, width=8)
        inputs = torch.zeros((2, 24, 63, 65))
        gate = model.gate(inputs)
        self.assertEqual(gate.shape, (2, 1, 63, 65))
        self.assertLess(float(gate.mean()), 0.03)


if __name__ == "__main__":
    unittest.main()


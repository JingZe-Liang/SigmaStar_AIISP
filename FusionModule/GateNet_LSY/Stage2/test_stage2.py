from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


STAGE2_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STAGE2_ROOT.parents[2]
PHASE2_ROOT = WORKSPACE_ROOT / "Phase2"
for path in (STAGE2_ROOT, PHASE2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_io import discover_dataset  # noqa: E402
from fusion_loss import WeakFusionLoss  # noqa: E402
from training_dataset import FusionTrainingDataset  # noqa: E402

from fusion_loss_stage2 import Stage2FusionLoss  # noqa: E402
from gatenet_stage2 import FEATURE_CHANNELS, GateNetStage2, build_gate_features  # noqa: E402


DATASET_ROOT = PHASE2_ROOT / "DATASET"
MD_ROOT = PHASE2_ROOT / "DERIVED" / "md_mog2"


class Stage2ModelTests(unittest.TestCase):
    def test_compact_model_shapes_and_parameter_count(self):
        model = GateNetStage2()
        features = torch.randn(2, FEATURE_CHANNELS, 32, 32)
        alpha, motion_logit = model(features, return_motion=True)
        self.assertEqual(tuple(alpha.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(motion_logit.shape), (2, 1, 32, 32))
        self.assertTrue(torch.all((alpha >= 0.0) & (alpha <= 1.0)))
        self.assertLess(sum(p.numel() for p in model.parameters()), 3000)


@unittest.skipUnless(DATASET_ROOT.is_dir() and MD_ROOT.is_dir(), "Training data is absent")
class Stage2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = FusionTrainingDataset(
            discover_dataset(DATASET_ROOT),
            MD_ROOT,
            sequence_ids=("128x",),
            samples_per_epoch=4,
            crop_size=64,
            training=False,
        )

    def test_md_supervises_training_but_is_not_a_model_feature(self):
        batch = next(iter(DataLoader(self.dataset, batch_size=4)))
        features = build_gate_features(
            batch["denoised"],
            batch["fused"],
            batch["source"],
            batch["source_prev"],
            batch["source_next"],
            batch["noise_sigma"],
        )
        self.assertEqual(features.shape[1], FEATURE_CHANNELS)
        model = GateNetStage2()
        alpha, motion_logit = model(features, return_motion=True)
        loss, metrics = Stage2FusionLoss(WeakFusionLoss())(
            alpha, motion_logit, batch
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("motion_aux", metrics)
        loss.backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

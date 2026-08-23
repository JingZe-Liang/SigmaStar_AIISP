import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_io import discover_dataset
from fusion_loss import WeakFusionLoss
from gatenet import FEATURE_CHANNELS, GateNet, build_gate_features
from training_dataset import FusionTrainingDataset


BASE = Path(__file__).resolve().parent
DATASET_ROOT = BASE / "DATASET"
MD_ROOT = BASE / "DERIVED" / "md_mog2"


@unittest.skipUnless(DATASET_ROOT.is_dir() and MD_ROOT.is_dir(), "Training data is absent")
class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        catalog = discover_dataset(DATASET_ROOT)
        cls.dataset = FusionTrainingDataset(
            catalog,
            MD_ROOT,
            sequence_ids=("128x",),
            samples_per_epoch=4,
            crop_size=64,
            training=False,
        )

    def test_sample_contains_aligned_training_inputs(self):
        sample = self.dataset[0]
        for key in (
            "source",
            "source_prev",
            "source_next",
            "denoised",
            "fused",
            "proxy",
        ):
            self.assertEqual(tuple(sample[key].shape), (4, 32, 32))
        self.assertEqual(tuple(sample["motion"].shape), (1, 32, 32))
        self.assertEqual(tuple(sample["noise_sigma"].shape), (4, 1, 1))

    def test_model_and_loss_have_finite_nonzero_gradients(self):
        batch = next(iter(DataLoader(self.dataset, batch_size=4)))
        features = build_gate_features(
            batch["denoised"],
            batch["fused"],
            batch["source"],
            batch["source_prev"],
            batch["source_next"],
            batch["motion"],
            batch["noise_sigma"],
        )
        self.assertEqual(features.shape[1], FEATURE_CHANNELS)
        model = GateNet()
        alpha = model(features)
        loss, metrics = WeakFusionLoss()(alpha, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(metrics["static_fraction"]), 0.0)
        self.assertGreater(float(metrics["motion_fraction"]), 0.0)
        loss.backward()
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)


if __name__ == "__main__":
    unittest.main()

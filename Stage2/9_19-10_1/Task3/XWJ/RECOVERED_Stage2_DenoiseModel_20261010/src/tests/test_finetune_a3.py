"""A3 model loading and objective tests."""

from pathlib import Path

import torch

from raw_fusion.finetune.a3 import (
    A3Config,
    PhaseADataset,
    build_rvideformer,
    compute_loss,
    load_official_state,
    restore_checkpoint,
    save_checkpoint,
)


_CHECKPOINT = Path("/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/checkpoints/RViDeformer-L_CRVD.pth")


def test_official_checkpoint_matches_local_training_architecture() -> None:
    state = load_official_state(_CHECKPOINT)
    model = build_rvideformer()
    model.load_state_dict(state, strict=True)


def test_a3_loss_is_finite_with_spatial_mask() -> None:
    prediction = torch.full((1, 8, 4, 64, 64), 0.5, requires_grad=True)
    target = torch.full_like(prediction, 0.4)
    mask = torch.ones((1, 8, 1, 64, 64))
    loss, metrics = compute_loss(prediction, target, mask, 0.1, 0.1)
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "charbonnier", "gradient", "highpass"}
    loss.backward()
    assert prediction.grad is not None


def test_synthetic_temporal_window_has_independent_noise_frames() -> None:
    config = A3Config(
        dataset_root="/HardDisk/jingzeliang/Data/Company_Dataset/dataset",
        supervision_dir="artifacts/finetune_mis20s1/supervision",
        noise_dir="artifacts/finetune_mis20s1/noise_model",
        output_dir="artifacts/finetune_mis20s1/training",
        checkpoint=str(_CHECKPOINT),
        device="cpu",
    )
    dataset = PhaseADataset(config, samples=2)
    inputs, _, _ = dataset[1]
    assert inputs.shape == (8, 4, 64, 64)
    assert not torch.equal(inputs[0], inputs[1])


def test_resume_restores_model_ema_optimizer_and_step(tmp_path: Path) -> None:
    config = A3Config(
        dataset_root="dataset",
        supervision_dir="supervision",
        noise_dir="noise",
        output_dir=str(tmp_path),
        checkpoint=str(_CHECKPOINT),
        device="cpu",
    )
    model = torch.nn.Linear(2, 2)
    ema = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        model.weight.fill_(0.25)
        ema.weight.fill_(0.5)
    path = tmp_path / "resume.pth"
    save_checkpoint(path, model, ema, optimizer, 17, config)
    with torch.no_grad():
        model.weight.zero_()
        ema.weight.zero_()
    restored = restore_checkpoint(path, model, ema, optimizer, torch.device("cpu"))
    assert restored == 17
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.25))
    assert torch.allclose(ema.weight, torch.full_like(ema.weight, 0.5))

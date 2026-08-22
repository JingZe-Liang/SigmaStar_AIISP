from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from losses import NAFBPNWeakFusionLoss
from model import NAFBPNMotionFusionNet, extract_model_state


def main() -> int:
    parser = argparse.ArgumentParser(description="NAF-BPN weak-supervision invariants")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    model = NAFBPNMotionFusionNet(
        num_basis=int(config.get("num_basis", 15)),
        kernel_size=int(config.get("kernel_size", 7)),
        width=int(config.get("model_width", 32)),
    )
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(extract_model_state(payload), strict=True)
    model.eval()
    batch = [torch.rand(2, 1, 64, 64) for _ in range(4)]
    with torch.no_grad():
        output = model(*batch)
    if output.shape != batch[0].shape or not torch.isfinite(output).all():
        raise AssertionError("NAF-BPN forward 输出尺寸或数值错误")

    weak_batch = {
        "image_2dnr": batch[0],
        "image_3dnr": batch[1],
        "noisy_current": batch[2],
        "proxy": torch.rand_like(batch[0]),
        "motion_target": torch.zeros_like(batch[0]),
        "valid_signal": torch.ones_like(batch[0]),
        "temporal_difference": torch.zeros_like(batch[0]),
        "temporal_range": torch.zeros_like(batch[0]),
        "noise_sigma": torch.ones_like(batch[0]),
    }
    loss, metrics = NAFBPNWeakFusionLoss()(output, weak_batch)
    if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in metrics.values()):
        raise AssertionError("弱监督 loss 产生非有限值")
    print(
        "验证通过: NAF-BPN 四路部署接口、弱监督 loss、严格 checkpoint 加载；"
        f"参数量={sum(parameter.numel() for parameter in model.parameters())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

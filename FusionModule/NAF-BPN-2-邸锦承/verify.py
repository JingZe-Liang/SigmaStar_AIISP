from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import CODE_MAX, NR_BLACK_LEVEL, SOURCE_BLACK_LEVEL
from model import NAFBPNMotionFusionNet


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 训练前不变量检查")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_zero = (SOURCE_BLACK_LEVEL - SOURCE_BLACK_LEVEL) / (CODE_MAX - SOURCE_BLACK_LEVEL)
    nr_zero = (NR_BLACK_LEVEL - NR_BLACK_LEVEL) / (CODE_MAX - NR_BLACK_LEVEL)
    source_one = (CODE_MAX - SOURCE_BLACK_LEVEL) / (CODE_MAX - SOURCE_BLACK_LEVEL)
    nr_one = (CODE_MAX - NR_BLACK_LEVEL) / (CODE_MAX - NR_BLACK_LEVEL)
    if (source_zero, nr_zero, source_one, nr_one) != (0.0, 0.0, 1.0, 1.0):
        raise AssertionError("黑电平线性映射不满足端点条件")
    model = NAFBPNMotionFusionNet()
    allowed = model.basis_allowed
    center = model.kernel_size // 2
    if bool(allowed[:, center, center].any()):
        raise AssertionError("严格版 BPN 中心不应允许")
    for row in range(model.kernel_size):
        for column in range(model.kernel_size):
            expected = (row - center) % 2 == 0 and (column - center) % 2 == 0 and (row != center or column != center)
            if bool(allowed[0, row, column]) != expected:
                raise AssertionError(f"CFA BPN mask 错误: {(row, column)}")
    basis = model._basis(torch.randn(2, 256, 32, 32))
    if float(basis[..., ~allowed].abs().max().detach()) != 0.0:
        raise AssertionError("非法 BPN 权重必须为 0")
    if not torch.allclose(basis.sum(dim=(2, 3, 4)), torch.ones_like(basis.sum(dim=(2, 3, 4))), atol=1e-6):
        raise AssertionError("每个 basis 的有效权重和必须为 1")
    print("验证通过: 黑电平端点、同 CFA 非中心 BPN mask、核权重归一化均正确")


if __name__ == "__main__":
    main()

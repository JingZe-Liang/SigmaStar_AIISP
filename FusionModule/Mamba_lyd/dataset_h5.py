"""Compatibility wrapper for dataset code used by the model package.

目前预处理实现已经稳定放在 `MyNet.data_prepare` 下，这里只做一层轻薄转发，
让后续模型、训练脚本和验证脚本统一从 `MyNet.ai_fusion` 入口导入。
"""

from MyNet.data_prepare.dataset_h5 import H5FusionDataset, H5FusionDatasetConfig

__all__ = [
    "H5FusionDataset",
    "H5FusionDatasetConfig",
]

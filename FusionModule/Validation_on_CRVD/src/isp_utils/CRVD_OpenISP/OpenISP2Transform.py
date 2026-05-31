import torch
import numpy as np
from typing import Optional
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from torch.utils.data import DataLoader

"""
【 OpenISP2transform 使用指南 】

1. 功能概述：
   本工具主要用于衔接 PyTorch 深度学习流水线与 NumPy 格式的 ISP 处理流程（如 OpenISP2 或 RawPy）。
   核心功能是将神经网络输出的 4 维视频序列 Tensor (B, T, H, W) 平铺为 3 维的 NumPy 数组 (B*T, H, W)。

2. 输入输出规范：
   - 输入 (Tensor): 必须为 4 维张量，形状为 (Batch_Size, Time_Steps, Height, Width)。
   - 输出 (NumPy): 形状为 (Batch_Size * Time_Steps, Height, Width)。

3. 数据格式支持：
   - 自动处理设备转换：支持输入 CUDA 或 CPU 上的 Tensor，输出统一为 CPU NumPy 数组。
   - 数据截断：默认通过 `to_uint16=True` 兼容 CRVD 等 RAW 数据集的 [0, 65535] 表示范围。
   - 梯度处理：内部自动执行 detach()，不会影响反向传播。

4. 快速调用示例：
   transformer = OpenISP2transform()
   # 方式 A: 调用 forward 方法
   numpy_data = transformer.forward(batch_tensor, to_uint16=True)
   # 方式 B: 直接像函数一样调用
   numpy_data = transformer(batch_tensor, to_uint16=True)

5. 配合 SequenceWise_Load 使用建议：
   在序列模式下，DataLoader 返回 (B, 7, 1080, 1920)。
   使用此工具转换后得到 (B*7, 1080, 1920)，可直接送入循环进行单帧 ISP 处理。
"""

class OpenISP2transform:
    def __init__(self):
        pass

    def forward(self,
                tensor: torch.Tensor,
                dtype: Optional[np.dtype] = None,
                to_uint16: bool = True) -> np.ndarray:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"输入应为torch.Tensor，但得到 {type(tensor)}")

        if tensor.dim() != 4:
            raise ValueError(f"输入张量应为4维 (B, T, H, W)，但得到 {tensor.dim()} 维")

        B, T, H, W = tensor.shape

        if tensor.is_cuda:
            numpy_array = tensor.detach().cpu().numpy()
        else:
            numpy_array = tensor.detach().numpy()

        numpy_array = numpy_array.reshape(-1, H, W)

        if to_uint16:
            numpy_array = numpy_array.astype(np.uint16)
        elif dtype is not None:
            numpy_array = numpy_array.astype(dtype)

        return numpy_array

    def __call__(self,
                 tensor: torch.Tensor,
                 dtype: Optional[np.dtype] = None,
                 to_uint16: bool = False) -> np.ndarray:
        return self.forward(tensor, dtype, to_uint16)


if __name__ == "__main__":
    isp2 = OpenISP2transform()
    noisy_root = "D:/A_Data/CRVD_dataset/indoor_raw_noisy"
    gt_root = "D:/A_Data/CRVD_dataset/indoor_raw_gt"

    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1, 2,3,4,5,6],
        iso_levels=[1600,3200],
        num_frames=7,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(loader))

    batch_noisy, batch_gt = next(iter(loader))
    print(batch_noisy.shape, batch_gt.shape)

    numpy_array = isp2.forward(batch_noisy)
    print(f"输入形状: {batch_noisy.shape}")
    print(f"输出形状: {numpy_array.shape}")
    print(f"输出类型: {numpy_array.dtype}")

    numpy_array2 = isp2(batch_noisy, to_uint16=True)
    print(f"转换为uint16后类型: {numpy_array2.dtype}")

    if torch.cuda.is_available():
        batch_noisy_gpu = batch_noisy.cuda()
        numpy_array_gpu = isp2(batch_noisy_gpu)
        print(f"GPU张量转换后形状: {numpy_array_gpu.shape}")

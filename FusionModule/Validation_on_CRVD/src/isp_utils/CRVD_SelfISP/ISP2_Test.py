# test_isp_with_dataloader_simple.py
"""
使用DataLoader加载CRVD序列，通过ISP处理并显示7帧RAW和sRGB对比
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# 添加模块路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入自定义模块
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_SelfISP.ISP2 import ISPModule


def main():
    # 设置路径（修改为你的实际路径）
    noisy_root = "D:/A_Data/CRVD_dataset/indoor_raw_noisy"
    gt_root = "D:/A_Data/CRVD_dataset/indoor_raw_gt"

    print("1. Creating dataset and DataLoader...")

    # 创建序列数据集
    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[8],
        iso_levels=[3200],
        sequence_mode=True
    )

    print(f"Dataset size: {len(dataset)} sequences")

    # 创建DataLoader，batch_size=2
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    print("2. Loading one batch from DataLoader...")

    # 获取一个batch
    batch_noisy, batch_gt = next(iter(dataloader))


    # batch_gt = batch_gt
    # batch_noisy = batch_noisy / (4095-240)

    print(f"Batch shape: noisy={batch_noisy.shape}, gt={batch_gt.shape}")
    print(f"Batch value range: [{batch_noisy.min():.0f}, {batch_noisy.max():.0f}]")

    # 创建ISP模块
    isp = ISPModule()
    srgb_noisy = isp(batch_noisy)

    print("3. Processing batch with ISP...")

    # 只处理noisy数据 (2, 7, H, W) -> (2, 7, 3, H, W)
    with torch.no_grad():
        srgb_noisy = isp(batch_noisy)  # (2, 7, 3, H, W)

    print(f"ISP output shape: {srgb_noisy.shape}")
    print(f"ISP output range: [{srgb_noisy.min():.4f}, {srgb_noisy.max():.4f}]")

    # 取出第一个batch（第一个序列）的7帧
    first_batch_raw = batch_noisy[0]  # (7, H, W) 原始RAW数据
    first_batch_srgb = srgb_noisy[0]  # (7, 3, H, W) sRGB数据

    print(f"\nFirst batch shape: RAW={first_batch_raw.shape}, sRGB={first_batch_srgb.shape}")

    # 准备显示函数
    def tensor_to_image_rgb(tensor):
        """RGB tensor转numpy图像"""
        img = tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        # 注意：ISP已经做了伽马校正，输出范围在[0,1]
        img = img.clip(0, 1)  # 确保在[0,1]范围
        return img

    print("4. Displaying 7 frames: RAW (grayscale) vs sRGB (color)...")

    # 创建大图显示7帧对比
    fig, axes = plt.subplots(7, 2, figsize=(8, 20))

    for i in range(7):  # 遍历7帧
        # RAW图像（灰度） - 显示原始值，ISP内部会做黑电平校正和归一化
        raw_img = first_batch_raw[i].numpy()

        # sRGB图像（彩色） - ISP已处理
        srgb_img = tensor_to_image_rgb(first_batch_srgb[i])

        # 显示RAW帧（灰度） - 显示原始传感器值
        axes[i, 0].imshow(raw_img, cmap='gray')
        axes[i, 0].set_title(f'Frame {i + 1}: RAW ({raw_img.min():.0f}-{raw_img.max():.0f})')
        axes[i, 0].axis('off')

        # 显示sRGB帧（彩色） - ISP处理后
        axes[i, 1].imshow(srgb_img)
        axes[i, 1].set_title(f'Frame {i + 1}: sRGB ({srgb_img.min():.2f}-{srgb_img.max():.2f})')
        axes[i, 1].axis('off')

    plt.suptitle('7 Frames: RAW (left, sensor values) vs ISP Processed sRGB (right, [0,1])', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 打印数值统计
    print("\n--- Frame Statistics ---")
    for i in range(7):
        raw_frame = first_batch_raw[i].numpy()
        srgb_frame = first_batch_srgb[i].numpy()

        print(f"Frame {i + 1}:")
        print(f"  RAW - min={raw_frame.min():.0f}, max={raw_frame.max():.0f}, mean={raw_frame.mean():.1f}")
        print(f"  sRGB - min={srgb_frame.min():.3f}, max={srgb_frame.max():.3f}, mean={srgb_frame.mean():.3f}")


if __name__ == "__main__":
    main()
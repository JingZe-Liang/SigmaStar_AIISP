import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# 确保你的路径引用正确
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2


def verify_crvd_loading_fixed():
    # 1. 路径配置 (请根据你的实际盘符检查)
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"

    # 2. 实例化数据集 (序列模式)
    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1],
        iso_levels=[1600],
        sequence_mode=True
    )

    # 3. 读取第一组序列 (7, H, W)
    noisy_seq, gt_seq = dataset[0]

    # 4. 打印位深信息 (修复了之前的 ValueError)
    print(f"✅ 成功读取序列！")
    print(f"📊 形状 (Frames, H, W): {noisy_seq.shape}")
    # 修正格式化代码：使用 .2f 即可
    print(f"💡 Noisy 序列原始值范围 (12-bit): [{noisy_seq.min():.2f}, {noisy_seq.max():.2f}]")
    print(f"💡 GT 序列原始值范围 (12-bit):    [{gt_seq.min():.2f}, {gt_seq.max():.2f}]")

    # 5. 模拟量化：如果我们要喂给 VBM3D (PNG 8-bit)
    # 12-bit (0-4095) -> 8-bit (0-255)
    noisy_8bit = (noisy_seq / 16.0).astype(np.uint8)
    print(f"📏 缩放至 8-bit 后范围: [{noisy_8bit.min()}, {noisy_8bit.max()}]")

    # 6. 可视化第一帧
    isp = OpenCV_ISP2(show_preview=False)
    noisy_tensor = torch.from_numpy(noisy_seq).unsqueeze(0)  # (1, 7, H, W)
    color_seq = isp(noisy_tensor)[0]  # (7, H, W, 3)

    plt.figure(figsize=(10, 6))
    plt.imshow(color_seq[0])
    plt.title(f"Scene 1 | ISO 1600 | Frame 1 Preview\nRaw Max: {noisy_seq.max():.2f}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    verify_crvd_loading_fixed()
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_OpenCV_ISP.Tensor_To_Numpy_Reshape import OpenISP2transform

"""
【 说明 】

1. FastISP 类:
   - 职责：执行底层图像处理算子。
   - 输入：(H, W) NumPy 数组。
   - 输出：(H, W, 3) sRGB 格式 NumPy 数组。
   - 算法：黑电平扣除 -> AWB 增益 -> OpenCV 快速 Demosaic -> Gamma 校正。

2. OpenCV_ISP2 类:
   - 职责：管理数据流。
   - 局限性 ： 必须手动确定 每一个样本的帧数T
   - 流程：接收 Tensor -> 调用 OpenISP2transform 展开 -> 循环调用 FastISP -> 维度恢复与显示。
   - 输出：（batch,7（T）,1080,1920）
"""

class FastISP:
    def __init__(self, black_level=240, white_level=4095, r_gain=1.9, b_gain=1.5):
        self.black_level = black_level
        self.white_level = white_level
        self.r_gain = r_gain
        self.b_gain = b_gain

    def process_single_frame(self, frame_np):
        # 1. 归一化与黑电平处理
        img = (frame_np.astype(np.float32) - self.black_level) / (self.white_level - self.black_level)
        img = np.clip(img, 0, 1)

        # 2. 白平衡增益 (针对 GBRG 修正)
        img[1::2, 0::2] *= self.r_gain
        img[0::2, 1::2] *= self.b_gain
        img = np.clip(img, 0, 1)

        # 3. 去马赛克 (使用 GR 模式解决“僵尸脸”问题)
        rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BayerGR2RGB)

        # 4. Gamma 校正
        rgb = np.power(rgb / 255.0, 1 / 2.2) * 255
        return rgb.astype(np.uint8)


class OpenCV_ISP2:
    """
    【 说明 】


    2. OpenCV_ISP2 类:
       - 接收（B,T,H,W）的tensor，未BLC，未norm
       - 输出 Numpy（B , T , H , W , 3 ）
       - 操作 ：对其进行ISP（包含BLC，norm）
       - 局限性 ： 必须specify每一个样本的帧数T
       - 流程：接收 Tensor -> 调用 OpenISP2transform 展开 -> 循环调用 FastISP -> 维度恢复与显示。
    """
    def __init__(self, show_preview=False):
        self.isp_engine = FastISP()
        self.show_preview = show_preview

    def forward(self, x: torch.Tensor):
        # 1. 形状转换 (B, T, H, W) -> (BT, H, W)
        isp2transform = OpenISP2transform()
        flat_frames = isp2transform(x)

        # 2. 获取维度信息
        T = 7
        BT = flat_frames.shape[0]
        B = BT // T
        H, W = flat_frames.shape[1], flat_frames.shape[2]

        # 3. 序列处理
        processed_list = []
        for i in range(BT):
            rgb_frame = self.isp_engine.process_single_frame(flat_frames[i])
            processed_list.append(rgb_frame)

        # 4. 组合并恢复五维结构 (B, T, H, W, 3)
        processed_array = np.array(processed_list)
        processed_reshaped = processed_array.reshape(B, T, H, W, 3)

        # 5. 可视化预览
        if self.show_preview:
            self._display(processed_reshaped[0, 0])

        return processed_reshaped

    def _display(self, img):
        plt.figure(figsize=(8, 5))
        plt.imshow(img)
        plt.title("Modular Fast ISP - Normalized Color")
        plt.axis('off')
        plt.show()

    def __call__(self, tensor):
        return self.forward(tensor)

if __name__ == "__main__":
    noisy_root = "D:/A_Data/CRVD_dataset/indoor_raw_noisy"
    gt_root = "D:/A_Data/CRVD_dataset/indoor_raw_gt"

    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1, 2],
        iso_levels=[1600,3200],
        num_frames=7,
        # sequence_mode = False
    )

    print(f"Total samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch_noisy, batch_gt = next(iter(loader))

    t = OpenCV_ISP2()
    m = t(batch_noisy)
    print()
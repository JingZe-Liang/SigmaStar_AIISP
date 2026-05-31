import cv2
import numpy as np
import torch


def denoise_2d(raw1: torch.Tensor, t: int) -> torch.Tensor:
    """
    1. 输入：RAW1    ：Tensor（BT , H , W），已经归一化；T
    2. 输出：RAW1_2 : Tensor（BT , H , W），依旧是 0-1
    """
    device = raw1.device
    dtype = raw1.dtype
    bt, h, w = raw1.shape

    # 转为 uint8 [0, 255]
    raw_np = (raw1.cpu().numpy() * 255).astype(np.uint8)

    # 逐帧双边滤波
    denoised_frames = []
    for i in range(bt):
        frame = raw_np[i]
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=75, sigmaSpace=75)
        denoised_frames.append(denoised)

    # 转回 [0, 1]
    raw1_2 = np.stack(denoised_frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(raw1_2).to(device).to(dtype)

if __name__ == "__main__":
    # 模拟已归一化数据
    raw1 = torch.rand(32, 256, 256)  # BT=32, 值域 [0, 1]
    T = 8

    # 处理前
    print(f"处理前 - shape: {raw1.shape}")
    print(f"处理前 - min: {raw1.min().item():.4f}, max: {raw1.max().item():.4f}")

    # 2D 降噪
    raw1_2 = denoise_2d(raw1, t=T)

    # 处理后
    print(f"\n处理后 - shape: {raw1_2.shape}")
    print(f"处理后 - min: {raw1_2.min().item():.4f}, max: {raw1_2.max().item():.4f}")

    # 降噪效果评估
    diff = (raw1 - raw1_2).abs()
    print(f"\n平均差异: {diff.mean().item():.6f}")
    print(f"最大差异: {diff.max().item():.6f}")
    print(f"受影响像素比例: {(diff > 0.01).float().mean().item():.2%}")


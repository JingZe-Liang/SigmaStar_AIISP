from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

# =================【输入 / 输出配置】=================
# FILE_PATH：要检查的 .npy RAW 文件路径。
# 这个脚本不写文件，输出主要是终端结论和可视化窗口。
# ====================================================
FILE_PATH = Path(r"D:\DeepLearning\RAWProcessing\A001_03052215_C005_000000_bayer.npy")


def check_npy_is_raw(file_path: str | Path) -> None:
    """从维度、数值分布和像素跳变特征判断 .npy 是否像 Bayer RAW。"""
    file_path = Path(file_path)
    # 【输入标记】这里读取待检测的 .npy 文件。
    data = np.load(file_path)

    print(f"--- 正在检测 NPY 文件：{file_path} ---")
    print(f"数据维度：{data.shape}")
    print(f"数据类型：{data.dtype}")

    if data.ndim == 2:
        print("[特征] 当前是 2D 矩阵，符合 Bayer RAW 的常见排布。")
    elif data.ndim == 3 and (data.shape[2] == 4 or data.shape[0] == 4):
        print("[特征] 当前是 4 通道 packed RAW，后续 Bayer 检测不再继续。")
        return
    elif data.ndim == 3 and (data.shape[2] == 3 or data.shape[0] == 3):
        print("[警告] 当前更像 RGB 图像，而不是原始 RAW。")
        return
    else:
        print("[警告] 数据维度不符合常见 RAW 形式。")
        return

    height, width = data.shape
    patch = data[height // 2 : height // 2 + 100, width // 2 : width // 2 + 100].astype(np.float32)
    diff_h = float(np.mean(np.abs(patch[:, 0::2] - patch[:, 1::2])))
    mean_val = float(np.mean(patch))
    mosaic_ratio = diff_h / (mean_val + 1e-6)

    print(f"相邻像素跳变比例：{mosaic_ratio:.4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(data[height // 2 : height // 2 + 50, width // 2 : width // 2 + 50], cmap="gray", interpolation="nearest")
    plt.title("Zoomed-in Patch")

    if mosaic_ratio > 0.1:
        print(">> [结论] 大概率是真实 Bayer RAW，检测到了明显的马赛克像素跳变。")
    else:
        print(">> [结论] 更像普通灰度图，或已经经过平滑处理。")

    plt.show()


if __name__ == "__main__":
    check_npy_is_raw(FILE_PATH)

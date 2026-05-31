import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def detect_vbm3d_results():
    root = tk.Tk()
    root.withdraw()

    default_path = os.path.join(os.path.expanduser("~"), "Desktop", "VBM3D_3DNR_Results")

    file_path = filedialog.askopenfilename(
        initialdir=default_path,
        title="选择 VBM3D 去噪后的 TIFF 文件",
        filetypes=[("TIFF 图像", "*.tif;*.tiff"), ("所有文件", "*.*")]
    )

    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img_array = np.array(img).astype(np.float32)

        print("-" * 40)
        print(f"【VBM3D 产物深度检测】")
        print(f"文件名: {os.path.basename(file_path)}")

        # 基础维度信息
        h, w = img_array.shape[:2]
        channels = img_array.shape[2] if img_array.ndim == 3 else 1
        print(f"分辨率: {w} x {h}")
        if channels > 1:
            print(f"通道数: {channels} (彩色图像)")
        print(f"数据类型 (dtype): {img_array.dtype}")

        # 数值范围检测
        v_min = np.min(img_array)
        v_max = np.max(img_array)
        v_mean = np.mean(img_array)

        print(f"像素最小值: {v_min:.2f}")
        print(f"像素最大值: {v_max:.2f}")
        print(f"平均亮度: {v_mean:.2f}")

        # 位深逻辑判定
        if v_max > 255:
            print("判定结果: 确认保留了 12-bit/16-bit 高精度数据 (4095 域)")
        else:
            print("判定结果: 数据似乎被压缩到了 8-bit 范围 (255 域)")

        # ---------- 新增：真实位深检测 ----------
        # 展平所有像素值（如果是彩色，则将所有通道的值合并）
        flat_vals = img_array.ravel()
        unique_vals = np.unique(flat_vals)
        num_unique = len(unique_vals)
        print(f"\n【灰度级分析】")
        print(f"不同像素值数量: {num_unique} (理论最大4096)")

        # 根据唯一值数量判断
        if num_unique > 3000:
            print("真实性: 很可能为真实12位图像 (灰度级丰富)")
        elif num_unique > 1000:
            print("真实性: 可能为12位但有压缩或后处理")
        elif num_unique > 256:
            print("真实性: 疑似从8位拉伸 (灰度级较少)")
        else:
            print("真实性: 极可能为8位或更低图像")

        # 分析灰度级间隔特征（如果唯一值足够多才分析，避免除零）
        if num_unique > 10:
            sorted_vals = np.sort(unique_vals)
            gaps = np.diff(sorted_vals)
            median_gap = np.median(gaps)
            std_gap = np.std(gaps)
            print(f"灰度级间隔中位数: {median_gap:.2f}")
            print(f"间隔标准差: {std_gap:.2f}")

            # 判断是否存在均匀拉伸的迹象
            if std_gap < 0.5 and median_gap > 1.5:
                # 间隔非常一致且较大，可能是线性拉伸
                print("间隔特征: 灰度级间隔均匀，可能是从低位深线性拉伸而来")
                if 15 < median_gap < 17:
                    print("  └─ 符合8位→12位拉伸 (间隔≈16)")
                elif 3 < median_gap < 5:
                    print("  └─ 符合10位→12位拉伸 (间隔≈4)")
                else:
                    print(f"  └─ 间隔{median_gap:.1f}，可能为其他拉伸")
            elif median_gap == 1:
                print("间隔特征: 正常连续灰度级")
            else:
                print("间隔特征: 无明显拉伸迹象")
        else:
            print("灰度级过少，无法进行间隔分析")
        # ----------------------------------------

        # Matplotlib 预览
        plt.figure(figsize=(10, 8))
        if channels == 1:
            plt.imshow(img_array, cmap='gray', vmin=0, vmax=4095)
            plt.colorbar(label='Pixel Value (12-bit Range)')
        else:
            # 彩色图像：归一化显示（不改变原始数据）
            plt.imshow(np.clip(img_array / 4095, 0, 1))
            plt.colorbar(label='Normalized Value')
        plt.title(f"Preview: {os.path.basename(file_path)}")
        plt.axis('off')

        print("\n提示: 图片已在 PyCharm 的 SciView/Plots 窗口打开。")
        print("-" * 40)
        plt.show()

    except Exception as e:
        print(f"检测失败: {e}")


if __name__ == "__main__":
    detect_vbm3d_results()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def process_specific_raw(file_path):
    # --- 1. 参数提取 ---
    # 针对你提供的文件名：raw_stream_normal_3200x1800_16_RG_...[R=1155,G=1024,B=3691].raw
    width = 3200
    height = 1800

    # 固定的增益（也可以根据文件名正则提取，这里先按你给的数值写死）
    r_gain = 1155 / 1024.0
    g_gain = 1024 / 1024.0
    b_gain = 3691 / 1024.0

    print(f"处理文件: {os.path.basename(file_path)}")
    print(f"应用增益: R={r_gain:.2f}, G={g_gain:.2f}, B={b_gain:.2f}")

    # --- 2. 安全读取数据 ---
    try:
        # 使用 np.fromfile 读取
        raw_data = np.fromfile(file_path, dtype=np.uint16)

        # 很多 SigmaStar 的 RAW 文件末尾会有元数据，我们只取图像部分
        expected_size = width * height
        if raw_data.size < expected_size:
            raise ValueError(f"数据量不足! 期望 {expected_size}, 实际 {raw_data.size}")

        raw_img = raw_data[:expected_size].reshape((height, width)).astype(np.float32)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # --- 3. 核心 ISP 还原步骤 ---

    # A. 白平衡 (Bayer Domain WB)
    # 布局为 RGGB
    # R  G
    # G  B
    wb_mask = np.ones((height, width), dtype=np.float32)
    wb_mask[0::2, 0::2] = r_gain  # R
    wb_mask[0::2, 1::2] = g_gain  # G
    wb_mask[1::2, 0::2] = g_gain  # G
    wb_mask[1::2, 1::2] = b_gain  # B

    balanced_raw = raw_img * wb_mask

    # B. 亮度缩放 (SigmaStar RAW 在暗室下数值极低，需要拉伸)
    # 线性拉伸到 uint16 范围
    img_max = balanced_raw.max()
    if img_max > 0:
        balanced_raw = (balanced_raw / img_max * 65535).astype(np.uint16)
    else:
        balanced_raw = balanced_raw.astype(np.uint16)

    # C. 去马赛克 (Demosaic)
    # 模式为 RGGB
    color_img = cv2.cvtColor(balanced_raw, cv2.COLOR_BayerRG2RGB)

    # D. 后处理 (Gamma & Tone Mapping)
    final_img = color_img.astype(np.float32) / 65535.0
    # 增加曝光增强（针对 Darkroom）
    final_img = np.clip(final_img * 1.2, 0, 1)
    # Gamma 2.2 校正
    final_img = np.power(final_img, 1 / 2.2)

    # --- 4. 可视化 ---
    plt.figure(figsize=(12, 8))
    plt.imshow(final_img)
    plt.title(f"SigmaStar Reconstructed\n{os.path.basename(file_path)}")
    plt.axis('off')
    plt.show()


# --- 关键修正点：使用 r"" 避免路径转义错误 ---
path = r"E:\SigmaStar\Fortest\sc635hai\raw_stream_normal_3200x1800_16_RG_0205213025917_131x.raw"
process_specific_raw(path)
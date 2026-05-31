"""
RAW视频去噪工具模块
提供基于OpenCV的多种去噪方法（快速NL均值、双边滤波），支持GBRG Bayer格式的RAW数据。
包含分通道处理、PSNR计算以及数据集测试的主函数。
"""

import warnings
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2
import bm3d
# --- [新增内容：导入视觉评价指标库] ---
from skimage.metrics import structural_similarity as ssim_func
import os
import subprocess
import cv2
import numpy as np
from pathlib import Path

def detect_dataset_structure(root_path, depth=3):
    """
    探测数据集的文件组织模式
    """
    # 修正了引号嵌套问题
    print(f"\n{'=' * 20} 探测路径: {root_path} {'=' * 20}")
    root = Path(root_path)

    if not root.exists():
        print(f"错误: 路径 {root_path} 不存在。")
        return

    def walk_dir(current_path, current_depth):
        if current_depth > depth:
            return

        try:
            items = list(current_path.iterdir())
        except PermissionError:
            print("  " * current_depth + " [无权限访问]")
            return

        dirs = [d for d in items if d.is_dir()]
        files = [f for f in items if f.is_file()]

        indent = "  " * (current_depth - 1)
        print(f"{indent}文件夹: {current_path.name} (子文件夹数: {len(dirs)}, 文件数: {len(files)})")

        if files:
            # 同样注意这里的引号
            print(f"{indent}  示例文件: {[f.name for f in files[:3]]}")
            suffixes = set(f.suffix for f in files)
            print(f"{indent}  文件后缀: {suffixes}")

        for d in dirs[:5]:
            walk_dir(d, current_depth + 1)
        if len(dirs) > 5:
            print(f"{indent}  ... 还有 {len(dirs) - 5} 个文件夹未列出")

    walk_dir(root, 1)
# ----------------------------------------------------------------------
# [新增方法]：像素级运动检测 (Motion Detection)
# 基于帧间通道差异计算每一个像素的权重。差异大则权重低（保留细节），差异小则权重高（强降噪）。
# ----------------------------------------------------------------------
def compute_motion_weights(curr_channels, ref_channels, threshold=0.05, sensitivity=10.0):
    """
    输入: curr_channels, ref_channels (B, 4, Hh, Wh)
    输出: alpha (B, 4, Hh, Wh) 取值 [0, 1]
    """
    # 计算差异并归一化到 [0, 1] (假设 12位数据最大值4095)
    diff = torch.abs(curr_channels.float() - ref_channels.float()) / 4095.0

    # 映射公式：alpha = 1 - tanh(sensitivity * max(0, diff - threshold))
    # threshold 以下视为噪声/静止，alpha 接近 1；超过 threshold 视为运动，alpha 迅速下降
    alpha = 1.0 - torch.tanh(sensitivity * torch.clamp(diff - threshold, min=0))

    # 设定 alpha 上限，防止完全锁死不更新
    return alpha * 0.95

# ----------------------------------------------------------------------
# [新增方法]：基于 MD 权重的自适应递归时域降噪
# 该方法替代传统的固定比例融合，使用像素级 alpha 实时控制。
# ----------------------------------------------------------------------
def denoise_temporal_recursive_adaptive(raw_tensor, threshold=0.05, sensitivity=10.0):
    """
    Y_t = alpha * Y_{t-1} + (1 - alpha) * X_t
    """
    channels = split_bayer_gbrg(raw_tensor)
    B, F, C, Hh, Wh = channels.shape
    output_channels = torch.empty_like(channels)

    # 首帧处理
    output_channels[:, 0] = channels[:, 0]

    for f in range(1, F):
        curr_x = channels[:, f]
        prev_y = output_channels[:, f - 1]

        # 计算像素级融合权重
        alpha = compute_motion_weights(curr_x, prev_y, threshold, sensitivity)

        # 逐像素融合
        output_channels[:, f] = alpha * prev_y + (1.0 - alpha) * curr_x

    return merge_bayer_gbrg(output_channels)

# ----------------------------------------------------------------------
# Bayer 分通道工具函数
# ----------------------------------------------------------------------
def split_bayer_gbrg(raw_tensor):
    """
    将GBRG Bayer格式的RAW张量拆分为4个通道（G,R,B,G）
    输入形状: (B, F, H, W)   H,W为偶数
    输出形状: (B, F, 4, H//2, W//2)
    """
    B, F, H, W = raw_tensor.shape
    assert H % 2 == 0 and W % 2 == 0
    raw_reshaped = raw_tensor.view(B, F, H // 2, 2, W // 2, 2)
    # G左上
    ch0 = raw_reshaped[..., 0, :, 0]
    # B右上
    ch1 = raw_reshaped[..., 0, :, 1]
    # R左下
    ch2 = raw_reshaped[..., 1, :, 0]
    # G右下
    ch3 = raw_reshaped[..., 1, :, 1]
    channels = torch.stack([ch0, ch1, ch2, ch3], dim=2)
    return channels


def merge_bayer_gbrg(channels_tensor):
    """
    将4通道张量合并回GBRG Bayer格式
    输入形状: (B, F, 4, Hh, Wh)
    输出形状: (B, F, H, W) 其中 H=2*Hh, W=2*Wh
    """
    B, F, C, Hh, Wh = channels_tensor.shape
    assert C == 4
    ch0, ch1, ch2, ch3 = torch.unbind(channels_tensor, dim=2)
    H, W = Hh * 2, Wh * 2
    raw_tensor = torch.empty(
        (B, F, H, W), dtype=channels_tensor.dtype, device=channels_tensor.device
    )
    raw_tensor[..., 0::2, 0::2] = ch0
    raw_tensor[..., 0::2, 1::2] = ch1
    raw_tensor[..., 1::2, 0::2] = ch2
    raw_tensor[..., 1::2, 1::2] = ch3
    return raw_tensor


# ----------------------------------------------------------------------
# 快速NL均值去噪（8位精度）
# ----------------------------------------------------------------------
def denoise_channels_fastNL(
    channels_tensor, h=10, templateWindowSize=7, searchWindowSize=21, bit_depth=12
):
    """
    对四通道张量进行快速NL均值去噪，使用16位中间表示以保留高位深信息。
    输入：
        channels_tensor : torch.float32，形状 (B,F,4,Hh,Wh)，取值范围 [0, 2^bit_depth-1]
        h : 去噪强度参数（针对8位图像的经验值，内部自动缩放）
        templateWindowSize, searchWindowSize : OpenCV fastNlMeansDenoising 的窗口参数
        bit_depth : 原始数据位深
    返回：
        与输入同形状、同类型的去噪张量
    """
    is_tensor = torch.is_tensor(channels_tensor)
    if is_tensor:
        device = channels_tensor.device
        dtype = channels_tensor.dtype
        np_data = channels_tensor.cpu().numpy()
    else:
        np_data = channels_tensor

    max_val = (1 << bit_depth) - 1  # 例如 4095
    scale_16 = 65535.0 / max_val  # 缩放到16位范围的比例因子

    # 将数据缩放到16位并转为uint16
    np_uint16 = (np_data * scale_16).round().astype(np.uint16)

    # 缩放h参数到16位范围，保持去噪强度与原始数据范围一致
    h_scaled = float(h * (65535.0 / max_val))

    B, F, C, Hh, Wh = np_uint16.shape
    denoised_uint16 = np.empty_like(np_uint16)

    use_16bit = True  # 标记当前是否使用16位处理
    for b in range(B):
        for f in range(F):
            for c in range(C):
                img = np.ascontiguousarray(np_uint16[b, f, c])
                if use_16bit:
                    try:
                        # 尝试16位去噪（OpenCV支持16位，但部分版本可能不支持）
                        denoised = cv2.fastNlMeansDenoising(
                            img,
                            None,
                            np.array([h_scaled], dtype=np.float32),  # 关键修改
                            templateWindowSize,
                            searchWindowSize,
                            normType=cv2.NORM_L1,
                        )
                    except cv2.error as e:
                        warnings.warn(
                            f"16-bit denoising failed, falling back to 8-bit. Error: {e}"
                        )
                        use_16bit = False
                        # 降级到8位处理当前图像
                        img_8 = (np_data[b, f, c] * (255.0 / max_val)).round().astype(
                            np.uint8
                        )
                        h_8 = float(h)  # 原h值适用于8位
                        denoised_8 = cv2.fastNlMeansDenoising(
                            img_8, None, h_8, templateWindowSize, searchWindowSize
                        )
                        # 将8位结果映射回16位以便后续统一处理
                        denoised = (
                            denoised_8.astype(np.float32) * (65535.0 / 255.0)
                        ).round().astype(np.uint16)
                else:
                    # 已经降级，继续用8位处理
                    img_8 = (np_data[b, f, c] * (255.0 / max_val)).round().astype(
                        np.uint8
                    )
                    h_8 = float(h)
                    denoised_8 = cv2.fastNlMeansDenoising(
                        img_8, None, h_8, templateWindowSize, searchWindowSize
                    )
                    denoised = (
                        denoised_8.astype(np.float32) * (65535.0 / 255.0)
                    ).round().astype(np.uint16)

                denoised_uint16[b, f, c] = denoised

    # 将去噪结果从16位范围映射回原始范围
    np_final = denoised_uint16.astype(np.float32) / scale_16  # 得到 [0, max_val]

    if is_tensor:
        np_final = torch.from_numpy(np_final).to(device).to(dtype)
    return np_final


def denoise_raw_gbrg(
    raw_tensor, h=10, templateWindowSize=7, searchWindowSize=21, bit_depth=12
):
    """
    对RAW视频张量（B,F,H,W）进行GBRG分通道快速NL均值去噪
    参数含义同 denoise_channels_fastNL
    """
    channels = split_bayer_gbrg(raw_tensor)
    denoised_channels = denoise_channels_fastNL(
        channels, h, templateWindowSize, searchWindowSize, bit_depth
    )
    denoised_raw = merge_bayer_gbrg(denoised_channels)
    return denoised_raw


# ----------------------------------------------------------------------
# 双边滤波去噪
# ----------------------------------------------------------------------
def denoise_channels_bilateral(channels_tensor, d=5, sigmaColor=50, sigmaSpace=50, bit_depth=12):
    """
    对四通道张量进行双边滤波去噪，自动归一化到[0,1]处理。
    输入：
        channels_tensor : torch.float32，形状 (B,F,4,Hh,Wh)，取值范围 [0, 2^bit_depth-1]
        d : 滤波直径（正整数）
        sigmaColor : 颜色空间标准差（按8位图像习惯给出，内部自动转换）
        sigmaSpace : 坐标空间标准差
        bit_depth : 原始位深
    返回：
        与输入相同形状、相同类型的去噪张量
    """
    is_tensor = torch.is_tensor(channels_tensor)
    if is_tensor:
        device = channels_tensor.device
        dtype = channels_tensor.dtype
        np_data = channels_tensor.cpu().numpy()
    else:
        np_data = channels_tensor

    max_val = (1 << bit_depth) - 1
    # 归一化到 [0,1]
    np_norm = (np_data / max_val).astype(np.float32)
    sigmaColor_norm = sigmaColor / max_val  # 转换sigmaColor到[0,1]范围

    B, F, C, Hh, Wh = np_norm.shape
    denoised_norm = np.empty_like(np_norm)

    for b in range(B):
        for f in range(F):
            for c in range(C):
                img = np_norm[b, f, c]
                img = np.ascontiguousarray(img)
                filtered = cv2.bilateralFilter(img, d, sigmaColor_norm, sigmaSpace)
                denoised_norm[b, f, c] = filtered

    # 反归一化回原始范围
    denoised = denoised_norm * max_val

    if is_tensor:
        denoised = torch.from_numpy(denoised).to(device).to(dtype)
    return denoised


def denoise_raw_bilateral(raw_tensor, d=5, sigmaColor=50, sigmaSpace=50, bit_depth=12):
    """
    对RAW视频张量（B,F,H,W）进行GBRG分通道双边滤波去噪
    参数同 denoise_channels_bilateral
    """
    channels = split_bayer_gbrg(raw_tensor)
    denoised_channels = denoise_channels_bilateral(
        channels, d, sigmaColor, sigmaSpace, bit_depth
    )
    return merge_bayer_gbrg(denoised_channels)


def denoise_raw_bilateral_direct(raw_tensor, d=5, sigmaColor=50, sigmaSpace=50, bit_depth=12):
    """
    直接对RAW图像应用双边滤波（不拆分通道），用于与分通道方法对比。
    参数含义同 denoise_channels_bilateral
    """
    is_tensor = torch.is_tensor(raw_tensor)
    if is_tensor:
        device = raw_tensor.device
        dtype = raw_tensor.dtype
        np_data = raw_tensor.cpu().numpy()
    else:
        np_data = raw_tensor

    max_val = (1 << bit_depth) - 1
    np_norm = (np_data / max_val).astype(np.float32)
    sigmaColor_norm = sigmaColor / 255.0

    B, F, H, W = np_norm.shape
    denoised_norm = np.empty_like(np_norm)

    for b in range(B):
        for f in range(F):
            img = np_norm[b, f]
            img = np.ascontiguousarray(img)
            filtered = cv2.bilateralFilter(img, d, sigmaColor_norm, sigmaSpace)
            denoised_norm[b, f] = filtered

    denoised = denoised_norm * max_val

    if is_tensor:
        denoised = torch.from_numpy(denoised).to(device).to(dtype)
    return denoised


# ----------------------------------------------------------------------
# 评估指标
# ----------------------------------------------------------------------
def calculate_psnr(img1, img2, max_val=4095.0):
    """
    计算两张图像之间的峰值信噪比（PSNR）
    支持 torch.Tensor 或 numpy.ndarray，形状可为 (H,W) 或 (B,F,H,W)
    max_val : 图像的最大可能值（对于12位RAW数据为4095）
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)

# --- [新增内容：SSIM 与 RGB 域综合指标方法] ---
def calculate_ssim(img1, img2, data_range=1.0):
    """计算两张彩色图像之间的结构相似性（SSIM）"""
    if torch.is_tensor(img1): img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2): img2 = img2.detach().cpu().numpy()
    # multichannel=True 现改为 channel_axis=2
    return ssim_func(img1, img2, data_range=data_range, channel_axis=2)

def calculate_rgb_visual_metrics(denoised_rgb, gt_rgb):
    """在 RGB 空间综合测量 PSNR 和 SSIM"""
    p = calculate_psnr(denoised_rgb, gt_rgb, max_val=1.0)
    s = calculate_ssim(denoised_rgb, gt_rgb, data_range=1.0)
    return p, s
# --- [新增内容结束] ---

# ----------------------------------------------------------------------
# BM3D 去噪
# ----------------------------------------------------------------------
def denoise_channels_bm3d(channels_tensor, sigma_psd=0.1, bit_depth=12):
    """
    对四通道张量进行 BM3D 去噪。
    输入：
        channels_tensor : torch.float32，形状 (B,F,4,Hh,Wh)，取值范围 [0, 2^bit_depth-1]
        sigma_psd : 噪声标准差（相对于 [0,1] 范围），例如 noise_std / max_val
        bit_depth : 原始位深
    返回：
        与输入相同形状、相同类型的去噪张量
    """
    is_tensor = torch.is_tensor(channels_tensor)
    if is_tensor:
        device = channels_tensor.device
        dtype = channels_tensor.dtype
        np_data = channels_tensor.cpu().numpy()
    else:
        np_data = channels_tensor

    max_val = (1 << bit_depth) - 1
    # 归一化到 [0,1]
    np_norm = (np_data / max_val).astype(np.float64)  # BM3D 通常用 float64

    B, F, C, Hh, Wh = np_norm.shape
    denoised_norm = np.empty_like(np_norm)

    for b in range(B):
        for f in range(F):
            for c in range(C):
                img = np_norm[b, f, c]
                # BM3D 去噪，sigma_psd 是相对于 [0,1] 的噪声标准差
                denoised = bm3d.bm3d(img, sigma_psd=sigma_psd)
                denoised_norm[b, f, c] = denoised

    # 反归一化回原始范围
    denoised = denoised_norm * max_val

    if is_tensor:
        denoised = torch.from_numpy(denoised.astype(np.float32)).to(device).to(dtype)
    return denoised


def denoise_raw_bm3d(raw_tensor, sigma_psd=0.1, bit_depth=12):
    """
    对 RAW 视频张量 (B,F,H,W) 进行 GBRG 分通道 BM3D 去噪。
    sigma_psd : 噪声标准差（相对于 [0,1] 范围），需根据实际噪声水平调整。
    """
    channels = split_bayer_gbrg(raw_tensor)
    denoised_channels = denoise_channels_bm3d(channels, sigma_psd, bit_depth)
    return merge_bayer_gbrg(denoised_channels)


# ----------------------------------------------------------------------
# 时域滤波（3DNR）验证实验
# ----------------------------------------------------------------------
def denoise_temporal_recursive(raw_tensor):
    """
    递归式时域平均降噪：当前帧与上一帧的输出结果取平均。
    公式: Y_t = (X_t + Y_{t-1}) / 2
    输入: raw_tensor (B, F, H, W)
    输出: denoised_raw (B, F, H, W)
    """
    # 1. 拆分通道 (B, F, 4, H/2, W/2)
    channels = split_bayer_gbrg(raw_tensor)
    B, F, C, Hh, Wh = channels.shape

    # 创建输出容器
    output_channels = torch.empty_like(channels)

    # 2. 递归处理
    for f in range(F):
        if f == 0:
            # 第一帧保留原状（初始化递归）
            output_channels[:, f] = channels[:, f]
        else:
            # 当前帧 X_t 与 上一帧输出 Y_{t-1} 取平均
            hj = 6
            hj = hj / 10
            output_channels[:, f] = channels[:, f] * hj+ output_channels[:, f - 1] * (1- hj)

    # 3. 合并回 Bayer 格式
    return merge_bayer_gbrg(output_channels)
def main_temporal_test() -> None:
    # 路径配置
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"

    # 1. 加载数据集 (序列模式)
    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1],  # 测试场景1
        iso_levels=[3200],  # 测试ISO 3200
        num_frames=7,
        sequence_mode=True
    )

    # batch_size=1 方便逐个序列观察
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    isp = OpenCV_ISP2(show_preview=False)

    print(f"Starting Temporal Denoise Test. Total sequences: {len(loader)}")

    isp = OpenCV_ISP2(show_preview=False)

    for batch_idx, (noisy_seq, gt_seq) in enumerate(loader):
        # 1. 核心降噪：对整个序列执行简单时域平均
        # noisy_seq shape: (1, 7, 1080, 1920)
        denoised_seq = denoise_temporal_simple_average(noisy_seq)

        # 2. 一次性将整个序列转为彩色 (B, T, H, W, 3)
        # 这样符合 ISP 内部 T=7 的预期，避免 reshape 报错
        print(f"Converting sequence {batch_idx} to color...")
        color_gt_seq = isp(gt_seq)  # 输出形状 (1, 7, 1080, 1920, 3)
        color_noisy_seq = isp(noisy_seq)
        color_denoised_seq = isp(denoised_seq)

        # 3. 逐帧遍历显示
        for f_idx in range(7):
            # 从彩色序列中直接取帧 [batch_0, frame_f]
            img_gt = color_gt_seq[0, f_idx]
            img_noisy = color_noisy_seq[0, f_idx]
            img_denoised = color_denoised_seq[0, f_idx]

            # 计算这一帧的 PSNR (在 RAW 域计算)
            psnr_n = calculate_psnr(noisy_seq[0, f_idx], gt_seq[0, f_idx])
            psnr_d = calculate_psnr(denoised_seq[0, f_idx], gt_seq[0, f_idx])

            # 绘图展示
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img_gt)
            axes[0].set_title(f"Frame {f_idx}: GT")
            axes[0].axis('off')

            axes[1].imshow(img_noisy)
            axes[1].set_title(f"Noisy (PSNR: {psnr_n:.2f}dB)")
            axes[1].axis('off')

            axes[2].imshow(img_denoised)
            axes[2].set_title(f"Simple 3DNR (PSNR: {psnr_d:.2f}dB)")
            axes[2].axis('off')

            plt.suptitle(f"Seq {batch_idx} Frame {f_idx} | Two-Frame Average Test")
            plt.show()

            print(f"Frame {f_idx}: Noisy {psnr_n:.2f} -> Denoised {psnr_d:.2f}")

        # 只测试一个 sequence 即可
        break
def denoise_temporal_simple_average(raw_tensor):
    """
    最简单的时域降噪：当前帧与上一帧直接取平均。
    输入: raw_tensor (B, F, H, W) - F 为帧数
    输出: denoised_raw (B, F, H, W)
    """
    # 1. 分离通道 (B, F, 4, H/2, W/2)
    channels = split_bayer_gbrg(raw_tensor)
    B, F, C, Hh, Wh = channels.shape

    # 创建输出容器，默认复制输入内容（处理首帧）
    denoised_channels = channels.clone()

    # 2. 从第2帧开始，每帧与前一帧做平均
    # 注意：这里不是递归，每一帧只使用原始的当前帧和原始的上一帧
    for f in range(1, F):
        # O_t = (I_t + I_{t-1}) / 2
        denoised_channels[:, f] = (channels[:, f] + channels[:, f - 1]) * 0.5

    # 3. 合并回 Bayer 格式
    return merge_bayer_gbrg(denoised_channels)
def main_recursive_comparison() -> None:
    # 路径配置与初始化
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"
    isp = OpenCV_ISP2(show_preview=False)

    # 1. 加载一个序列 (sequence_mode=True)
    dataset = CRVDDataset(noisy_root, gt_root, scenes=[1], iso_levels=[3200], sequence_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    noisy_seq, gt_seq = next(iter(loader))

    # 2. 处理对象统一为 Noisy 序列
    denoised_2dnr = denoise_raw_bilateral(noisy_seq)
    denoised_3dnr = denoise_temporal_recursive(noisy_seq)

    # 3. 图像转彩色用于显示
    # 修复点：isp 返回已经是 numpy，直接取 [0] 即可
    color_gt = isp(gt_seq)[0]  # (7, H, W, 3)
    color_2dnr = isp(denoised_2dnr)[0]
    color_3dnr = isp(denoised_3dnr)[0]

    # 4. 边缘 1/4 面积 PSNR 掩码逻辑
    H, W = gt_seq.shape[2], gt_seq.shape[3]
    m_h, m_w = int(H * 0.067), int(W * 0.067)
    mask = np.ones((H, W), dtype=bool)
    mask[m_h:-m_h, m_w:-m_w] = False

    # 5. 确保计算 PSNR 时使用 Numpy 数组
    # 如果输入是 torch.Tensor，先转为 numpy
    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    np_gt = to_np(gt_seq[0])
    np_noisy = to_np(noisy_seq[0])
    np_2dnr = to_np(denoised_2dnr[0])
    np_3dnr = to_np(denoised_3dnr[0])

    # 6. 绘图 3行 7列
    fig, axes = plt.subplots(3, 7, figsize=(28, 14))

    for col in range(7):
        p_gt_mask = np_gt[col][mask]
        p_noisy_mask = np_noisy[col][mask]
        p_2dnr_mask = np_2dnr[col][mask]
        p_3dnr_mask = np_3dnr[col][mask]

        psnr_orig = calculate_psnr(p_gt_mask, p_noisy_mask)
        psnr_2dnr = calculate_psnr(p_gt_mask, p_2dnr_mask)
        psnr_3dnr = calculate_psnr(p_gt_mask, p_3dnr_mask)

        # 第一行: GT (标注原始 Noisy PSNR)
        axes[0, col].imshow(color_gt[col])
        axes[0, col].set_title(f"GT F{col}\n(Noisy PSNR: {psnr_orig:.2f})", fontsize=10)
        axes[0, col].axis('off')

        # 第二行: 2DNR (Bilateral)
        axes[1, col].imshow(color_2dnr[col])
        axes[1, col].set_title(f"2DNR (Bilateral)\nPSNR: {psnr_2dnr:.2f}", fontsize=10)
        axes[1, col].axis('off')

        # 第三行: 3DNR (Recursive)
        axes[2, col].imshow(color_3dnr[col])
        axes[2, col].set_title(f"3DNR (Recursive)\nPSNR: {psnr_3dnr:.2f}", fontsize=10)
        axes[2, col].axis('off')

    plt.suptitle("Denoising Comparison on Static Borders (1/4 Area PSNR)", fontsize=16)
    plt.tight_layout()
    plt.show()
def main_recursive_comparison_v2() -> None:
    # 路径与环境初始化
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"
    isp = OpenCV_ISP2(show_preview=False)

    # 1. 加载序列数据
    dataset = CRVDDataset(noisy_root, gt_root, scenes=[1], iso_levels=[3200], sequence_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    noisy_seq, gt_seq = next(iter(loader))  # (1, 7, 1080, 1920)

    # 2. 对 Noisy 序列执行两种降噪
    # 2DNR: 空间双边滤波
    denoised_2dnr = denoise_raw_bilateral(noisy_seq)
    # 3DNR: 递归时域平均
    denoised_3dnr = denoise_temporal_recursive(noisy_seq)

    # 3. 图像转彩色 (ISP 返回通常为 Numpy 数组)
    color_gt = isp(gt_seq)[0]
    color_2dnr = isp(denoised_2dnr)[0]
    color_3dnr = isp(denoised_3dnr)[0]

    # 4. 定义边缘 PSNR 掩码 (1 - 7/8*7/8 面积)
    H, W = gt_seq.shape[2], gt_seq.shape[3]
    # 边距为 1/16
    m_h, m_w = H // 16, W // 16
    mask = np.ones((H, W), dtype=bool)
    mask[m_h:-m_h, m_w:-m_w] = False

    # 辅助函数：确保数据转为 Numpy 且处于正确维度
    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    np_gt = to_np(gt_seq[0])
    np_noisy = to_np(noisy_seq[0])
    np_2dnr = to_np(denoised_2dnr[0])
    np_3dnr = to_np(denoised_3dnr[0])

    # 5. 绘图 3行 7列
    fig, axes = plt.subplots(3, 7, figsize=(28, 14))

    for col in range(7):
        # 提取当前帧边缘像素
        p_gt_edge = np_gt[col][mask]
        p_noisy_edge = np_noisy[col][mask]
        p_2dnr_edge = np_2dnr[col][mask]
        p_3dnr_edge = np_3dnr[col][mask]

        # 计算边缘区域的 PSNR
        psnr_orig = calculate_psnr(p_gt_edge, p_noisy_edge)
        psnr_2dnr = calculate_psnr(p_gt_edge, p_2dnr_edge)
        psnr_3dnr = calculate_psnr(p_gt_edge, p_3dnr_edge)

        # 第一行: GT (标注原始 Noisy PSNR 以作为基准对照)
        axes[0, col].imshow(color_gt[col])
        axes[0, col].set_title(f"GT Frame {col}\nBase PSNR: {psnr_orig:.2f}", color='blue')
        axes[0, col].axis('off')

        # 第二行: 2DNR (Bilateral)
        axes[1, col].imshow(color_2dnr[col])
        axes[1, col].set_title(f"2DNR (Bilateral)\nPSNR: {psnr_2dnr:.2f}")
        axes[1, col].axis('off')

        # 第三行: 3DNR (Recursive)
        axes[2, col].imshow(color_3dnr[col])
        # 标注 3DNR 相对原始 Noisy 的增益
        gain = psnr_3dnr - psnr_orig
        axes[2, col].set_title(f"3DNR (Recursive)\nPSNR: {psnr_3dnr:.2f} (+{gain:.2f})", color='red')
        axes[2, col].axis('off')

    plt.suptitle(f"Recursive 3DNR vs 2DNR: Border PSNR (Area: 1 - 49/64)", fontsize=18)
    plt.tight_layout()
    plt.show()

# --- [新增内容：支持视觉指标显示的终极对比主函数] ---
def main_visual_recursive_comparison() -> None:
    """
    视觉增强版对比实验：
    1. 同时展示 3DNR (递归) 与 2DNR (双边)。
    2. 计算边缘 1/16 RAW PSNR。
    3. 计算边缘区域的 RGB 域 PSNR 和 SSIM 视觉效果指数。
    """
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"
    isp = OpenCV_ISP2(show_preview=False)

    # 1. 加载序列数据
    dataset = CRVDDataset(noisy_root, gt_root, scenes=[1], iso_levels=[3200], sequence_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    noisy_seq, gt_seq = next(iter(iter(loader)))

    # 2. 降噪处理
    denoised_2dnr = denoise_raw_bilateral(noisy_seq)

    # 替换为新的自适应 3DNR 逻辑
    denoised_3dnr = denoise_temporal_recursive_adaptive(noisy_seq)

    # 3. ISP 转换为彩色 (B, T, H, W, 3)
    color_gt = isp(gt_seq)[0]
    color_2dnr = isp(denoised_2dnr)[0]
    color_3dnr = isp(denoised_3dnr)[0]

    # 4. 定义边缘掩码 (计算 1/16 宽度边缘)
    H, W = gt_seq.shape[2], gt_seq.shape[3]
    m_h, m_w = H // 16, W // 16
    mask = np.ones((H, W), dtype=bool)
    mask[m_h:-m_h, m_w:-m_w] = False

    def to_np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else x
    np_gt, np_noisy = to_np(gt_seq[0]), to_np(noisy_seq[0])
    np_2dnr, np_3dnr = to_np(denoised_2dnr[0]), to_np(denoised_3dnr[0])

    fig, axes = plt.subplots(3, 7, figsize=(28, 16))

    for col in range(7):
        # --- A. 计算 RAW 域边缘指标 ---
        p_raw_orig = calculate_psnr(np_gt[col][mask], np_noisy[col][mask])
        p_raw_2d = calculate_psnr(np_gt[col][mask], np_2dnr[col][mask])
        p_raw_3d = calculate_psnr(np_gt[col][mask], np_3dnr[col][mask])

        # --- B. 计算 RGB 域边缘视觉指标 ---
        # 计算 2DNR 边缘 SSIM
        _, s_map_2d = ssim_func(color_2dnr[col], color_gt[col], data_range=1.0, channel_axis=2, full=True)
        rgb_s_2d = s_map_2d[mask].mean()
        # 计算 2DNR 边缘 PSNR
        rgb_p_2d = calculate_psnr(color_2dnr[col][mask], color_gt[col][mask], max_val=1.0)

        # 计算 3DNR 边缘 SSIM
        _, s_map_3d = ssim_func(color_3dnr[col], color_gt[col], data_range=1.0, channel_axis=2, full=True)
        rgb_s_3d = s_map_3d[mask].mean()
        # 计算 3DNR 边缘 PSNR
        rgb_p_3d = calculate_psnr(color_3dnr[col][mask], color_gt[col][mask], max_val=1.0)

        # --- C. 绘图展示 ---
        # 第一行: GT (标注原始噪声参考)
        axes[0, col].imshow(color_gt[col])
        axes[0, col].set_title(f"GT F{col}\nNoisy RAW PSNR: {p_raw_orig:.2f}", color='blue', fontsize=9)
        axes[0, col].axis('off')

        # 第二行: 2DNR (双边滤波)
        axes[1, col].imshow(color_2dnr[col])
        axes[1, col].set_title(f"2DNR (Bilateral)\nBorder RAW PSNR: {p_raw_2d:.2f}\nBorder RGB SSIM: {rgb_s_2d:.4f}", fontsize=9)
        axes[1, col].axis('off')

        # 第三行: 3DNR (递归时域)
        axes[2, col].imshow(color_3dnr[col])
        axes[2, col].set_title(f"3DNR (Adaptive)\nBorder RAW PSNR: {p_raw_3d:.2f}\nBorder RGB SSIM: {rgb_s_3d:.4f}", color='red', fontsize=9)
        axes[2, col].axis('off')

    plt.suptitle("Motion-Adaptive Recursive 3DNR vs 2DNR: Border Visual Quality", fontsize=18)
    plt.tight_layout()
    plt.show()
# --- [新增内容结束] ---

# ----------------------------------------------------------------------
# [新增内容]：像素级运动权重控制与自适应递归函数
# ----------------------------------------------------------------------
def compute_motion_weights(curr_channels, ref_channels, threshold=0.05, sensitivity=10.0):
    """
    计算像素级融合权重 alpha。
    alpha 接近 1 表示静止（重度时域滤波），alpha 接近 0 表示运动（不进行时域滤波）。
    """
    # 差异计算 (假设 12bit 数据)
    diff = torch.abs(curr_channels.float() - ref_channels.float()) / 4095.0
    # 映射逻辑
    alpha = 1.0 - torch.tanh(sensitivity * torch.clamp(diff - threshold, min=0))
    return alpha * 0.95

def denoise_temporal_recursive_adaptive(raw_tensor, threshold=0.05, sensitivity=12.0):
    """
    新的递归时域降噪：接受MD权重，逐像素控制融合比例。
    """
    channels = split_bayer_gbrg(raw_tensor)
    B, F, C, Hh, Wh = channels.shape
    output_channels = torch.empty_like(channels)

    # 首帧初始化
    output_channels[:, 0] = channels[:, 0]

    for f in range(1, F):
        curr_frame = channels[:, f]
        prev_output = output_channels[:, f - 1]

        # 获取像素级融合权重
        alpha = compute_motion_weights(curr_frame, prev_output, threshold, sensitivity)

        # 执行融合: Y_t = alpha * Y_{t-1} + (1-alpha) * X_t
        output_channels[:, f] = alpha * prev_output + (1.0 - alpha) * curr_frame

    return merge_bayer_gbrg(output_channels)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# 2DNR数据集构建
# ----------------------------------------------------------------------
def construct_2dnr_datasets():
    """
    使用 dataset = CRVDDataset 读出 noisy 和 gt 处理。
    构建两个新数据集：2DNR_bilateral 和 2DNR_bm3d。
    模仿 GT 的结构: sceneX/ISOxxxx/frameX_clean_and_slightly_denoised.tiff
    """
    # 1. 路径与配置
    base_root = Path("E:/CRVD_dataset")
    noisy_root = str(base_root / "indoor_raw_noisy")
    gt_root = str(base_root / "indoor_raw_gt")

    # 定义输出目标
    save_roots = {
        "bilateral": base_root / "2DNR_bilateral",
        "bm3d": base_root / "2DNR_bm3d"
    }

    # 2. 实例化数据集 (直接利用 CRVDDataset 的读取逻辑)
    # 遍历所有场景 (1-11) 和 所有 ISO 级别
    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=list(range(1, 12)),
        iso_levels=[1600, 3200, 6400, 12800, 25600],
        sequence_mode=True  # 序列模式一次处理 7 帧
    )

    # batch_size=1 以便准确对应 samples 里的路径信息
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"开始构建 2DNR 数据集，总计序列数: {len(dataset)}")

    for idx, (noisy_seq, gt_seq) in enumerate(loader):
        # 获取当前样本的元数据 (scene 和 iso)
        meta = dataset.samples[idx]
        scene_id = meta['scene']
        iso_val = meta['iso']

        # --- A. 计算实际噪声标准差 sigma ---
        # noisy_seq 形状为 (1, 7, H, W)，转为 float 进行计算
        diff = noisy_seq.float() - gt_seq.float()
        actual_sigma = torch.std(diff).item()
        sigma_norm = actual_sigma / 4095.0

        # --- B. 2DNR 处理 ---
        # 1. 双边滤波 (sigmaColor = actual_sigma * 2)
        denoised_bilateral = denoise_raw_bilateral(
            noisy_seq,
            d=5,
            sigmaColor=round(actual_sigma * 2),
            sigmaSpace=50
        )

        # 2. BM3D 处理
        denoised_bm3d = denoise_raw_bm3d(noisy_seq, sigma_psd=sigma_norm)

        # --- C. 模仿 GT 结构保存 ---
        results = {
            "bilateral": denoised_bilateral[0],  # 去掉 batch 维 -> (7, H, W)
            "bm3d": denoised_bm3d[0]
        }

        for method, result_seq in results.items():
            # 构造路径: E:/CRVD_dataset/2DNR_xxx/sceneX/ISOxxxx/
            target_dir = save_roots[method] / f"scene{scene_id}" / f"ISO{iso_val}"
            target_dir.mkdir(parents=True, exist_ok=True)

            for f_idx in range(7):
                frame_num = f_idx + 1
                # 模仿 GT 的文件名规范
                file_name = f"frame{frame_num}_clean_and_slightly_denoised.tiff"
                save_path = target_dir / file_name

                # 数据处理：限制范围 [0, 4095] 并转为 uint16 保存
                img_data = result_seq[f_idx].cpu().numpy()
                img_uint16 = np.clip(img_data, 0, 4095).astype(np.uint16)

                cv2.imwrite(str(save_path), img_uint16)

        if (idx + 1) % 5 == 0:
            print(f"已完成: {idx + 1}/{len(dataset)} | 当前: Scene {scene_id} ISO {iso_val}")

    print("✅ 所有 2DNR 数据集构建完成。")

def test_construct_mini():
    """
    快速测试函数：仅处理 1 个序列，用于验证路径、文件名和位深。
    """
    # 1. 基础路径配置
    base_root = Path("E:/CRVD_dataset")
    noisy_root = str(base_root / "indoor_raw_noisy")
    gt_root = str(base_root / "indoor_raw_gt")

    # 2. 实例化数据集 - 仅选择 Scene 1 和 ISO 1600 进行快速测试
    test_dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1],
        iso_levels=[1600],
        sequence_mode=True  # 返回 (7, H, W)
    )

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    save_folders = ["2DNR_bilateral", "2DNR_bm3d"]

    print("--- 开始快速测试 (仅处理 Scene 1, ISO 1600) ---")

    for idx, (noisy_seq, gt_seq) in enumerate(loader):
        # 计算噪声 Sigma 及其归一化值
        diff = noisy_seq.float() - gt_seq.float()
        actual_sigma = torch.std(diff).item()
        sigma_norm = actual_sigma / 4095.0  # 12bit 归一化

        for folder in save_folders:
            # A. 执行降噪逻辑
            if "bilateral" in folder:
                # 严格按照要求：sigmaColor = actual_sigma * 2
                denoised = denoise_raw_bilateral(
                    noisy_seq,
                    d=5,
                    sigmaColor=round(actual_sigma * 2),
                    sigmaSpace=50
                )
            else:
                # BM3D 逻辑
                denoised = denoise_raw_bm3d(noisy_seq, sigma_psd=sigma_norm)

            # B. 模仿 GT 结构创建路径
            # 目标文件名: frameX_clean_and_slightly_denoised.tiff
            save_path = base_root / folder / "scene1" / "ISO1600"
            save_path.mkdir(parents=True, exist_ok=True)

            result_np = denoised[0].cpu().numpy()  # 移除 batch 维

            for f_idx in range(7):
                frame_name = f"frame{f_idx + 1}_clean_and_slightly_denoised.tiff"
                # 转回 12bit uint16 并保存
                img_to_save = np.clip(result_np[f_idx], 0, 4095).astype(np.uint16)
                cv2.imwrite(str(save_path / frame_name), img_to_save)

            print(f"已完成测试保存: {folder}")

    print("--- 测试结束，请检查 E:/CRVD_dataset/ 目录下的文件夹 ---")

def verify_and_visualize_new_datasets():
    """
    检测新数据集的程序：
    读取 Noisy, GT, Bilateral 和 BM3D 数据集的第三帧，
    计算 PSNR 并利用 ISP 进行彩色可视化展示。
    """
    import torch
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 1. 初始化路径与工具
    base_root = Path("E:/CRVD_dataset")
    paths = {
        "Noisy": base_root / "indoor_raw_noisy/scene10/ISO1600/frame3_noisy0.tiff",
        "GT": base_root / "indoor_raw_gt/scene10/ISO1600/frame3_clean_and_slightly_denoised.tiff",
        "Bilateral": base_root / "2DNR_bilateral/scene10/ISO1600/frame3_clean_and_slightly_denoised.tiff",
        "BM3D": base_root / "2DNR_bm3d/scene10/ISO1600/frame3_clean_and_slightly_denoised.tiff"
    }

    # 实例化 ISP 工具
    isp = OpenCV_ISP2(show_preview=False)

    # 2. 读取数据 (保持 uint16 原始 12-bit 深度)
    imgs_raw = {}
    for name, p in paths.items():
        if not p.exists():
            print(f"错误: 找不到路径 {p}")
            return
        img = cv2.imread(str(p), -1)  # 读取原始数据
        if img is None:
            print(f"读取失败: {p}")
            return
        imgs_raw[name] = img.astype(np.float32)

    # 3. 计算 PSNR (以 GT 为基准)
    # 调用 Denosie.py 中定义的 calculate_psnr
    psnr_noisy = calculate_psnr(imgs_raw["Noisy"], imgs_raw["GT"])
    psnr_bilateral = calculate_psnr(imgs_raw["Bilateral"], imgs_raw["GT"])
    psnr_bm3d = calculate_psnr(imgs_raw["BM3D"], imgs_raw["GT"])

    # 4. 可视化转换 (修复 Reshape 错误)
    color_results = {}
    for name, raw_np in imgs_raw.items():
        # 核心修复：
        # 1. 构造 (1, 1, H, W) 张量
        # 2. 如果 ISP 强制要求 T=7 才能 reshape，我们需要将单帧复制 7 次
        # 3. 或者根据报错提示，构造一个符合 ISP 内部 reshape 逻辑的 tensor
        raw_tensor = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # 为了适配某些写死 T=7 的 ISP 逻辑，我们这里 padding 成 T=7
        temp_seq = raw_tensor.repeat(1, 7, 1, 1)  # (1, 7, H, W)

        with torch.no_grad():
            # 此时输入 (1, 7, H, W)，输出应为 (1, 7, H, W, 3)
            processed = isp(temp_seq)
            # 取第一帧的结果即可
            color_results[name] = processed[0, 0]

    # 5. 绘图展示
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    display_configs = [
        ("Noisy", psnr_noisy),
        ("GT", None),
        ("Bilateral", psnr_bilateral),
        ("BM3D", psnr_bm3d)
    ]

    for i, (name, psnr_val) in enumerate(display_configs):
        axes[i].imshow(color_results[name])
        title = f"{name}"
        if psnr_val is not None:
            title += f"\nPSNR: {psnr_val:.2f} dB"
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    print("检测可视化完成。")

# ----------------------------------------------------------------------
# 3DNR数据集构建
# ----------------------------------------------------------------------
def construct_vbm3d_3dnr_mini(scene_id=1, iso_val=1600):
    """
    VBM3D 3DNR 实验方法：
    模仿 test_construct_mini 逻辑，针对特定场景和 ISO 进行高精度 3DNR 处理。
    1. 自动计算各通道 Sigma。
    2. 使用 16-bit TIFF 保持 12-bit 精度，无损回收 float32 结果。
    3. 最终在桌面生成去噪产物并打印 PSNR 提升总结。
    """
    import shutil
    # --- 1. 环境与路径硬核配置 ---
    bin_dir = Path(r"E:/Reasearch/Code/vbm3d_1/build/bin")
    exe_path = str(bin_dir / "VBM3Ddenoising.exe")
    # 临时工作区，用于存放分通道的 TIFF 序列
    workspace = Path(r"E:/vbm3d_experiment_temp")
    # 桌面产物路径
    desktop_save = Path(os.path.join(os.path.expanduser("~"), "Desktop")) / "VBM3D_3DNR_Experiment"

    # 清理旧环境
    if workspace.exists(): shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    if not desktop_save.exists(): desktop_save.mkdir(parents=True)

    # --- 2. 加载数据集 (使用 CRVDDataset 序列模式) ---
    base_root = Path("E:/CRVD_dataset")
    dataset = CRVDDataset(
        noisy_root=str(base_root / "indoor_raw_noisy"),
        gt_root=str(base_root / "indoor_raw_gt"),
        scenes=[scene_id],
        iso_levels=[iso_val],
        sequence_mode=True
    )

    if len(dataset) == 0:
        print(f"错误: 未找到 Scene {scene_id} ISO {iso_val} 的数据")
        return

    noisy_seq, gt_seq = dataset[0]  # (7, 1080, 1920)
    noisy_tensor = torch.from_numpy(noisy_seq).unsqueeze(0)  # (1, 7, 1080, 1920)
    gt_tensor = torch.from_numpy(gt_seq).unsqueeze(0)

    # --- 3. 分通道逻辑处理 (G1, R, B, G2) ---
    print(f"--- 开始 VBM3D 3DNR 实验 [Scene {scene_id} | ISO {iso_val}] ---")
    noisy_ch_seq = split_bayer_gbrg(noisy_tensor)[0]  # (7, 4, 540, 960)
    gt_ch_seq = split_bayer_gbrg(gt_tensor)[0]

    ch_names = ["G1", "R", "B", "G2"]
    denoised_ch_list = []

    for c in range(4):
        ch_dir = workspace / ch_names[c]
        ch_dir.mkdir()

        # A. 动态计算该通道的物理 Sigma
        diff = noisy_ch_seq[:, c] - gt_ch_seq[:, c]
        sigma_val = torch.std(diff).item()

        print(f"  > 正在处理通道 {ch_names[c]} (Sigma: {sigma_val:.2f})...")

        # B. 导出 16-bit 输入序列
        for f in range(7):
            img_uint16 = noisy_ch_seq[f, c].numpy().astype(np.uint16)
            cv2.imwrite(str(ch_dir / f"in_{f + 1:04d}.tif"), img_uint16)

        # C. 调用 VBM3D 算子 (注意：产物后缀为 .tiff，命名为 %03d)
        # 清理 bin 目录防止干扰
        for f_old in bin_dir.glob("deno_*.tiff"): os.remove(f_old)

        input_pattern = str(ch_dir / "in_%04d.tif").replace("\\", "/")
        cmd = [exe_path, "-i", input_pattern, "-f", "1", "-l", "7", "-sigma", f"{sigma_val:.2f}", "-add", "false"]

        subprocess.run(cmd, cwd=str(bin_dir), capture_output=True, check=True)

        # D. 无损回收 float32 结果
        recovered_frames = []
        for f_idx in range(1, 8):
            prod_path = bin_dir / f"deno_{f_idx:03d}.tiff"  # 算子生成的特定命名格式
            # 使用 cv2.IMREAD_UNCHANGED 保持 float32 精度
            img_deno = cv2.imread(str(prod_path), cv2.IMREAD_UNCHANGED)
            recovered_frames.append(torch.from_numpy(img_deno))

        denoised_ch_list.append(torch.stack(recovered_frames))

    # --- 4. 合并与结果评价 ---
    # 构造 (1, 7, 4, 540, 960)
    merged_input = torch.stack(denoised_ch_list, dim=1).unsqueeze(0)
    denoised_raw = merge_bayer_gbrg(merged_input)[0].numpy()  # (7, 1080, 1920)

    # 结果统计
    print(f"\n--- 实验总结: PSNR 变化 ---")
    isp = OpenCV_ISP2(show_preview=False)

    for f in range(7):
        psnr_n = calculate_psnr(noisy_seq[f], gt_seq[f])
        psnr_d = calculate_psnr(denoised_raw[f], gt_seq[f])
        print(f"  Frame {f + 1}: Noisy {psnr_n:.2f}dB -> 3DNR {psnr_d:.2f}dB (Gain: {psnr_d - psnr_n:+.2f}dB)")

        # 保存去噪后的 12-bit RAW 到桌面产物文件夹
        final_save_path = desktop_save / f"scene{scene_id}_ISO{iso_val}_frame{f + 1}_VBM3D.tiff"
        cv2.imwrite(str(final_save_path), np.clip(denoised_raw[f], 0, 4095).astype(np.uint16))

    # --- 5. 快速可视化对比 ---
    with torch.no_grad():
        rgb_noisy = isp(torch.from_numpy(noisy_seq).unsqueeze(0))[0, 3]  # 取中间帧
        rgb_deno = isp(torch.from_numpy(denoised_raw).unsqueeze(0))[0, 3]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(rgb_noisy)
    axes[0].set_title("Noisy Input (Frame 4)")
    axes[1].imshow(rgb_deno)
    axes[1].set_title("VBM3D 3DNR Output (Frame 4)")
    plt.suptitle(f"VBM3D Experiment: Scene {scene_id} ISO {iso_val}")
    plt.show()

    print(f"\n✅ 实验完成。产物已保存至: {desktop_save}")


def construct_3dnr_vbm3d_datasets():
    """
    使用 VBM3D 算子构建 3DNR 数据集。
    [特性]:
    1. 自动适配 3位/4位 数字补齐 (001 vs 0001) [cite: 15]
    2. 自动适配 .tif/.tiff 后缀
    3. 动态计算 Bayer 分通道 Sigma
    4. 12-bit 无损 float32 回收
    """
    import shutil
    # --- 1. 环境路径配置 ---
    base_root = Path("E:/CRVD_dataset")
    bin_dir = Path(r"E:/Reasearch/Code/vbm3d_1/build/bin")
    exe_path = str(bin_dir / "VBM3Ddenoising.exe")
    save_root = base_root / "3DNR_vbm3d"

    # 临时工作空间
    workspace = Path(r"E:/vbm3d_dataset_workspace")
    if workspace.exists(): shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # 2. 实例化数据集 (遍历所有场景与所有 ISO)
    # 根据文档，该实现支持多线程加速 [cite: 10, 11]
    dataset = CRVDDataset(
        noisy_root=str(base_root / "indoor_raw_noisy"),
        gt_root=str(base_root / "indoor_raw_gt"),
        scenes=list(range(1, 12)),
        iso_levels=[1600, 3200, 6400, 12800, 25600],
        sequence_mode=True
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"🚀 3DNR 构建流水线启动。总序列: {len(dataset)}")

    for idx, (noisy_seq, gt_seq) in enumerate(loader):
        meta = dataset.samples[idx]
        scene_id, iso_val = meta['scene'], meta['iso']
        target_dir = save_root / f"scene{scene_id}" / f"ISO{iso_val}"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[任务 {idx + 1}/{len(dataset)}] 处理中: Scene {scene_id} | ISO {iso_val}")

        # --- A. 分通道准备 ---
        noisy_ch_seq = split_bayer_gbrg(noisy_seq)[0]
        gt_ch_seq = split_bayer_gbrg(gt_seq)[0]

        ch_names = ["G1", "R", "B", "G2"]
        denoised_ch_results = []

        for c in range(4):
            ch_dir = workspace / f"{ch_names[c]}"
            if ch_dir.exists(): shutil.rmtree(ch_dir)
            ch_dir.mkdir()

            # 动态计算通道 Sigma
            sigma_val = torch.std(noisy_ch_seq[:, c] - gt_ch_seq[:, c]).item()

            # 导出 16-bit 临时 TIFF (使用 %04d 定律)
            for f in range(7):
                cv2.imwrite(str(ch_dir / f"in_{f + 1:04d}.tif"), noisy_ch_seq[f, c].numpy().astype(np.uint16))

            # 执行算子 [cite: 14, 15]
            input_pattern = str(ch_dir / "in_%04d.tif").replace("\\", "/")
            cmd = [exe_path, "-i", input_pattern, "-f", "1", "-l", "7", "-sigma", f"{sigma_val:.2f}", "-add", "false"]

            # 预先清理，确保回收的绝对是当前结果
            for f_old in bin_dir.glob("deno_*.tif*"): os.remove(f_old)

            subprocess.run(cmd, cwd=str(bin_dir), capture_output=True, check=True)

            # --- B. 智能回收逻辑 (多模式兼容) ---
            ch_recovered = []
            for f_idx in range(1, 8):
                # 定义探测优先级：优先找你日志中的 001.tiff，再找标准的 0001.tif
                possible_names = [
                    f"deno_{f_idx:03d}.tiff",
                    f"deno_{f_idx:03d}.tif",
                    f"deno_{f_idx:04d}.tiff",
                    f"deno_{f_idx:04d}.tif"
                ]

                final_path = None
                for name in possible_names:
                    if (bin_dir / name).exists():
                        final_path = bin_dir / name
                        break

                if final_path:
                    # 读取 float32 无损数据
                    img_deno = cv2.imread(str(final_path), cv2.IMREAD_UNCHANGED)
                    ch_recovered.append(torch.from_numpy(img_deno))
                else:
                    raise FileNotFoundError(f"❌ 无法在 {bin_dir} 找到 VBM3D 产物，探测模式: {possible_names}")

            denoised_ch_results.append(torch.stack(ch_recovered))

        # --- C. 合并与 12-bit 落地 ---
        merged_input = torch.stack(denoised_ch_results, dim=1).unsqueeze(0)
        final_denoised_raw = merge_bayer_gbrg(merged_input)[0].numpy()

        for f in range(7):
            file_name = f"frame{f + 1}_clean_and_temporal_denoised.tiff"
            img_to_save = np.clip(final_denoised_raw[f], 0, 4095).astype(np.uint16)
            cv2.imwrite(str(target_dir / file_name), img_to_save)

        # 增益反馈报告
        gain = np.mean([calculate_psnr(final_denoised_raw[f], gt_seq[0, f].numpy()) -
                        calculate_psnr(noisy_seq[0, f].numpy(), gt_seq[0, f].numpy()) for f in range(7)])
        print(f"✨ Scene {scene_id} ISO {iso_val} 完成 | 增益: {gain:+.2f} dB")

    if workspace.exists(): shutil.rmtree(workspace)
    print(f"\n🎉 3DNR_vbm3d 数据集构建大功告成！")


def main() -> None:
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"

    torch.manual_seed(10)
    print(cv2.__version__)
    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1],
        iso_levels=[1600],
        num_frames=7,
        # sequence_mode = False
    )
    print(f"Total samples: {len(dataset)},Maximum is 55")

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"Number of batches: {len(loader)}")

    max_display_batches = 3
    # noise_std = 30000  # 12位数据上的噪声标准差

    for batch_idx, (original_noisy, gt) in enumerate(loader):
        # 如果 original_noisy 和 gt 已经是 tensor 且在 [0, 4095]，直接计算
        diff = original_noisy.float() - gt.float()

        # 2. 计算标准差 (Standard Deviation)
        # 这反映了噪声偏离真值的平均程度
        actual_sigma = torch.std(diff)
        # 忽略original_noisy，使用gt作为干净图像，手动添加噪声
        clean = gt
        # noise = torch.randn_like(clean) * noise_std
        noisy = original_noisy
        noisy = torch.clamp(noisy, 0, 4095)




        # BM3D 去噪
        sigma_norm = actual_sigma / 4095.0
        denoised_raw = denoise_raw_bm3d(noisy, sigma_psd=sigma_norm)

        # 去噪方法选择（可切换注释以对比）
        # 加上 .item() 提取数值后再 round
        denoised_raw = denoise_raw_bilateral(
            noisy,
            d=5,
            sigmaColor=round(actual_sigma.item()*2),
            sigmaSpace=50
        )
        # denoised_raw = denoise_raw_bilateral_direct(noisy, d=5, sigmaColor=50, sigmaSpace=50)
        # denoised_raw = denoise_raw_gbrg(noisy)

        psnr_noisy = calculate_psnr(noisy, clean)
        psnr_denoised = calculate_psnr(denoised_raw, clean)
        print(f"Batch {batch_idx}: Noisy PSNR = {psnr_noisy:.2f} dB, Denoised PSNR = {psnr_denoised:.2f} dB")
        # 在 main 函数开头导入或实例化 ISP
        isp = OpenCV_ISP2(show_preview=False)

        # 在显示部分，对于每个 batch，将 noisy, denoised_raw, gt 转为彩色
        # 注意：denoised_raw 也是 tensor，形状与 noisy 相同
        if batch_idx < max_display_batches:
            # 构造 (B, T, H, W) 的张量，这里我们只取 batch 内的第一个样本，但需要保持 batch 维度
            # 由于 OpenCV_ISP2 期望输入 (B, T, H, W)，我们可以直接传入 noisy[0:1] 等，但需要确保 T=7（从原始数据集中来）
            # 更简单：将 noisy[0] (shape (T, H, W)) 扩展为 (1, T, H, W)，然后传入 ISP，取结果的 [0,0]
            noisy_single = noisy[0:1]  # shape (1, T, H, W)
            denoised_single = denoised_raw[0:1]
            gt_single = gt[0:1]

            # 转换
            color_noisy = isp(noisy_single)[0, 0]  # (H, W, 3)
            color_denoised = isp(denoised_single)[0, 0]
            color_gt = isp(gt_single)[0, 0]

            # 显示
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(color_noisy)
            axes[0].set_title(f'Noisy (PSNR: {psnr_noisy:.2f})')
            axes[0].axis('off')

            axes[1].imshow(color_denoised)
            axes[1].set_title(f'Denoised (PSNR: {psnr_denoised:.2f})')
            axes[1].axis('off')

            axes[2].imshow(color_gt)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # 调用新增的包含 MD 权重的视觉增强版对比函数
    construct_3dnr_vbm3d_datasets()
    # noisy_path = "E:/CRVD_dataset/indoor_raw_noisy"
    # gt_path = "E:/CRVD_dataset/indoor_raw_gt"
    #
    # detect_dataset_structure(noisy_path)
    # detect_dataset_structure(gt_path)
"""
RAW视频去噪工具模块
提供基于OpenCV的多种去噪方法（快速NL均值、双边滤波），支持GBRG Bayer格式的RAW数据。
包含分通道处理、PSNR计算以及数据集测试的主函数。
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2
import bm3d
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm  # 用于显示进度条
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
    sigmaColor_norm = sigmaColor / 255.0  # 转换sigmaColor到[0,1]范围

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


# 导入你现有的工具函数
# 假设 split_bayer_gbrg, merge_bayer_gbrg, calculate_psnr 已在上下文定义

# ----------------------------------------------------------------------
# 1. 定义 ResNet 去噪模型
# ----------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class SimpleResNetDenoiser(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, num_blocks=8):
        super(SimpleResNetDenoiser, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[ResBlock(64) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 学习残差：Noise = Input - Clean -> Output = Input - Network_Output
        res = self.head(x)
        res = self.body(res)
        out = self.tail(res)
        return x - out  # 预测噪声并减去



# ----------------------------------------------------------------------
# 训练与监控函数
# ----------------------------------------------------------------------
def train():
    # --- 1. 基础配置与设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_std = 500.0
    max_val = 4095.0
    epochs = 100
    crop_size = 256  # 裁剪为 256x256 的小块，彻底解决 OOM
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 实例化数据集与加载器 ---
    # 确保路径正确，num_frames=1 足够做 2D 去噪
    dataset = CRVDDataset(
        noisy_root="E:/CRVD_dataset/indoor_raw_noisy",
        gt_root="E:/CRVD_dataset/indoor_raw_gt",
        scenes=[1, 2, 3, 4, 5,6,7,8,9,10,11],
        iso_levels=[1600, 3200,6400,12800,25600],
        num_frames=1
    )
    # Batch Size 设为 4-8 即可，裁剪后显存占用极低
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- 3. 模型初始化 ---
    model = SimpleResNetDenoiser(in_channels=4, out_channels=4,num_blocks=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f"开始训练 | 设备: {device} | 样本数: {len(dataset)}")
    print(f"预处理模式: 分通道双边滤波 -> CNN 残差修复")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for _, gt in pbar:
            # --- A. 随机裁剪 (Random Crop) ---
            # gt shape: (B, 1, H, W)
            b, f, h, w = gt.shape
            i = torch.randint(0, h - crop_size + 1, (1,)).item()
            j = torch.randint(0, w - crop_size + 1, (1,)).item()
            gt_patch = gt[:, :, i: i + crop_size, j: j + crop_size].to(device)

            # --- B. 数据准备与加噪 ---
            clean = gt_patch
            noise = torch.randn_like(clean) * noise_std
            noisy = torch.clamp(clean + noise, 0, max_val)

            # --- C. 分通道处理 (转为 4 通道) ---
            # split_bayer_gbrg 返回 (B, 1, 4, H/2, W/2)
            noisy_ch_5d = split_bayer_gbrg(noisy)
            clean_ch_4d = split_bayer_gbrg(clean).squeeze(1)

            # --- D. 修复维度不匹配的 ValueError ---
            # denoise_channels_bilateral 内部期望 5 维输入
            with torch.no_grad():
                # 先进行双边滤波预处理
                bf_denoised_5d = denoise_channels_bilateral(
                    noisy_ch_5d, d=5, sigmaColor=150, sigmaSpace=25, bit_depth=12
                )
                # 将处理后的结果压扁成 4 维 (B, 4, Hh, Wh) 喂给网络
                input_ch = bf_denoised_5d.squeeze(1).to(device)

            # --- E. 网络训练 ---
            optimizer.zero_grad()
            output_ch = model(input_ch)

            # 损失计算 (预测结果 vs 干净的 4 通道图)
            loss = criterion(output_ch, clean_ch_4d)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 每个 Epoch 打印平均损失
        avg_loss = epoch_loss / len(loader)

        # --- F. 推理 PSNR 差异 (每 10 个 Epoch 打印一次) ---
        if (epoch + 1) % 10 == 0:
            val_psnr = validate_simple(model, loader, device, noise_std, max_val, crop_size)
            print(f" >> Epoch {epoch + 1} 完成 | Avg Loss: {avg_loss:.4f} | 当前 PSNR: {val_psnr:.2f} dB")
            # 保存权重
            torch.save(model.state_dict(), os.path.join(save_dir, f"resnet_epoch_{epoch + 1}.pth"))

    # 最终保存
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet_final.pth"))
    print("训练圆满结束！")


def validate_simple(model, loader, device, noise_std, max_val, crop_size):
    """验证函数：计算双边滤波+CNN后的 PSNR"""
    model.eval()
    with torch.no_grad():
        # 取一个 batch
        _, gt = next(iter(loader))
        # 裁剪出中间块进行验证
        h, w = gt.shape[-2:]
        ch, cw = crop_size, crop_size
        gt_patch = gt[0:1, :, h // 2:h // 2 + ch, w // 2:w // 2 + cw].to(device)

        noise = torch.randn_like(gt_patch) * noise_std
        noisy = torch.clamp(gt_patch + noise, 0, max_val)

        # 预处理
        noisy_ch_5d = split_bayer_gbrg(noisy)
        bf_ch_5d = denoise_channels_bilateral(noisy_ch_5d, d=5, sigmaColor=150, sigmaSpace=25)
        input_ch = bf_ch_5d.squeeze(1).to(device)

        # 推理并还原
        denoised_ch = model(input_ch)
        denoised_raw = merge_bayer_gbrg(denoised_ch.unsqueeze(1))

        return calculate_psnr(denoised_raw, gt_patch, max_val=max_val)


def compare_bf_vs_cnn():
    # --- 1. 环境配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_std = 400.0
    max_val = 4095.0
    weight_path = "checkpoints/resnet_final.pth"

    # 实例化 ISP 用于彩色可视化
    isp = OpenCV_ISP2(show_preview=False)

    # --- 2. 加载训练好的模型 ---
    model = SimpleResNetDenoiser(in_channels=4, out_channels=4,num_blocks=64).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print(f"成功加载模型权重: {weight_path}")
    else:
        print("错误：找不到权重文件，请先训练模型！")
        return
    model.eval()

    # --- 3. 获取一张随机测试图 ---
    dataset = CRVDDataset(
        noisy_root="E:/CRVD_dataset/indoor_raw_noisy",
        gt_root="E:/CRVD_dataset/indoor_raw_gt",
        scenes=[10],  # 使用测试集场景
        num_frames=1,
        iso_levels=[1600]
    )
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    _, gt = next(iter(test_loader))

    # 准备高噪声数据
    clean = gt[:, 0:1, :, :].to(device)
    noise = torch.randn_like(clean) * noise_std
    noisy = torch.clamp(clean + noise, 0, max_val)

    # --- 4. 算法处理流程 ---
    with torch.no_grad():
        # A. 分通道预处理
        noisy_ch_5d = split_bayer_gbrg(noisy)

        # B. 纯双边滤波输出 (BF Only)
        bf_ch_5d = denoise_channels_bilateral(
            noisy_ch_5d, d=5, sigmaColor=150, sigmaSpace=25, bit_depth=12
        )
        raw_bf_only = merge_bayer_gbrg(bf_ch_5d)  # 仅 BF 的结果

        # C. 双边滤波 + CNN 精炼 (BF + CNN)
        input_ch = bf_ch_5d.squeeze(1).to(device)
        denoised_ch = model(input_ch)
        raw_cnn_refined = merge_bayer_gbrg(denoised_ch.unsqueeze(1))

    # --- 5. 计算指标 ---
    psnr_noisy = calculate_psnr(noisy, clean, max_val=max_val)
    psnr_bf = calculate_psnr(raw_bf_only, clean, max_val=max_val)
    psnr_cnn = calculate_psnr(raw_cnn_refined, clean, max_val=max_val)

    # --- 6. 视觉对比可视化 ---

    # 定义一个辅助函数来确保 Tensor 格式正确 (B, T, H, W)
    def prep_for_isp(t):
        # 如果 t 是 (1, 1, H, W) 或者是 (1, H, W)
        if t.ndim == 3:
            return t.unsqueeze(0).unsqueeze(1)  # 变 (1, 1, H, W)
        if t.ndim == 4 and t.shape[1] != 1:  # 如果第2维不是1
            return t.unsqueeze(1)  # 插入 T 维度
        return t

    # 在调用前进行转换
    # 注意：如果你的 ISP 强制要求 T=7，我们需要用 repeat 复制帧
    noisy_in = prep_for_isp(noisy)
    bf_in = prep_for_isp(raw_bf_only)
    cnn_in = prep_for_isp(raw_cnn_refined)
    gt_in = prep_for_isp(clean)

    # 转换所有 RAW 结果到 RGB 空间
    # 如果 ISP 报错是因为 T 不等于 7，这里使用 .repeat(1, 7, 1, 1)
    try:
        img_noisy = isp(noisy_in)[0, 0]
        img_bf = isp(bf_in)[0, 0]
        img_cnn = isp(cnn_in)[0, 0]
        img_gt = isp(gt_in)[0, 0]
    except ValueError:
        # 强制适配 T=7 的情况
        img_noisy = isp(noisy_in.repeat(1, 7, 1, 1))[0, 0]
        img_bf = isp(bf_in.repeat(1, 7, 1, 1))[0, 0]
        img_cnn = isp(cnn_in.repeat(1, 7, 1, 1))[0, 0]
        img_gt = isp(gt_in.repeat(1, 7, 1, 1))[0, 0]

    # 创建对比画布
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(img_noisy)
    axes[0].set_title(f"Noisy (Input)\nPSNR: {psnr_noisy:.2f}dB")
    axes[0].axis('off')

    axes[1].imshow(img_bf)
    axes[1].set_title(f"Bilateral Filter Only\nPSNR: {psnr_bf:.2f}dB")
    axes[1].axis('off')

    axes[2].imshow(img_cnn)
    axes[2].set_title(f"BF + CNN (Refined)\nPSNR: {psnr_cnn:.2f}dB")
    axes[2].axis('off')

    axes[3].imshow(img_gt)
    axes[3].set_title(f"Ground Truth\n(Reference)")
    axes[3].axis('off')

    plt.suptitle(f"Denoising Comparison (Noise Std: {noise_std})", fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"\n数值对比结果:")
    print(f"1. 原始噪声: {psnr_noisy:.2f} dB")
    print(f"2. 仅双边滤波: {psnr_bf:.2f} dB")
    print(f"3. BF + CNN 精炼: {psnr_cnn:.2f} dB (提升: {psnr_cnn - psnr_bf:.2f} dB)")


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

    for batch_idx, (noisy_seq, gt_seq) in enumerate(loader):
        # 2. 执行简单的两帧平均时域降噪
        # noisy_seq shape: (1, 7, H, W)
        denoised_seq = denoise_temporal_simple_average(noisy_seq)

        # 3. 逐帧对比显示 (0 到 6 帧)
        for f_idx in range(7):
            # 获取当前帧数据并转为彩色
            # 为了 ISP 处理，需要扩展维度到 (1, 1, H, W)
            color_gt = isp(gt_seq[:, f_idx:f_idx + 1])[0, 0]
            color_noisy = isp(noisy_seq[:, f_idx:f_idx + 1])[0, 0]
            color_denoised = isp(denoised_seq[:, f_idx:f_idx + 1])[0, 0]

            # 计算当前帧 PSNR
            psnr_n = calculate_psnr(noisy_seq[:, f_idx], gt_seq[:, f_idx])
            psnr_d = calculate_psnr(denoised_seq[:, f_idx], gt_seq[:, f_idx])

            # 可视化
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(color_gt)
            axes[0].set_title(f"Frame {f_idx}: Ground Truth")
            axes[0].axis('off')

            axes[1].imshow(color_noisy)
            axes[1].set_title(f"Noisy (PSNR: {psnr_n:.2f}dB)")
            axes[1].axis('off')

            axes[2].imshow(color_denoised)
            axes[2].set_title(f"Simple 3DNR (PSNR: {psnr_d:.2f}dB)")
            axes[2].axis('off')

            plt.suptitle(f"Sequence {batch_idx} - Two-Frame Average Test")
            plt.tight_layout()
            plt.show()  # 关闭当前窗口会自动弹出下一帧

            print(f"Seq {batch_idx} Frame {f_idx}: Noisy {psnr_n:.2f} -> Denoised {psnr_d:.2f}")

if __name__ == "__main__":
    # 执行推理测试的部分保持不变，可以直接调用
    # train()
    # compare_bf_vs_cnn()
    main_temporal_test()


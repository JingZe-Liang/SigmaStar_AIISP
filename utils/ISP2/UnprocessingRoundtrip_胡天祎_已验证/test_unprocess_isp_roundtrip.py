"""
测试 Unprocess 和 ISP Pipeline 的往复转换

警告：这两个流程 **不是完全互逆** 的！
主要原因：
1. Gamma 值不匹配 (unprocess: 2.2, ISP: 0.5)
2. Demosaicing 插值会引入误差
3. ISP 有额外的处理步骤（去噪、锐化等）
4. Unprocess 有 inverse_smoothstep，ISP 没有对应步骤

这个测试只是为了观察往复转换的质量损失。
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from third_party.unprocessing_torch.dataloader import unprocess


def bayer_to_raw_format(bayer_4ch: torch.Tensor) -> np.ndarray:
    """
    将 Bayer 4通道格式 (4, H/2, W/2) 转换为 RAW 单通道格式 (H, W)
    
    Args:
        bayer_4ch: shape (4, H/2, W/2), 通道顺序为 [R, Gr, Gb, B]
    
    Returns:
        raw: shape (H, W), Bayer pattern encoded in spatial positions
    """
    _, h_half, w_half = bayer_4ch.shape
    h, w = h_half * 2, w_half * 2
    
    # 创建输出 RAW 图像
    raw = np.zeros((h, w), dtype=np.float32)
    
    # RGGB pattern
    raw[0::2, 0::2] = bayer_4ch[0].numpy()  # R
    raw[0::2, 1::2] = bayer_4ch[1].numpy()  # Gr
    raw[1::2, 0::2] = bayer_4ch[2].numpy()  # Gb
    raw[1::2, 1::2] = bayer_4ch[3].numpy()  # B
    
    return raw


def simple_demosaic(raw: np.ndarray) -> np.ndarray:
    """
    简单的 Bayer demosaicing (最近邻插值)
    
    Args:
        raw: shape (H, W), RGGB Bayer pattern
    
    Returns:
        rgb: shape (H, W, 3)
    """
    h, w = raw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # 提取 Bayer channels
    r = raw[0::2, 0::2]   # R
    gr = raw[0::2, 1::2]  # G at red rows
    gb = raw[1::2, 0::2]  # G at blue rows
    b = raw[1::2, 1::2]   # B
    
    # R channel - 简单上采样
    rgb[0::2, 0::2, 0] = r
    rgb[0::2, 1::2, 0] = r
    rgb[1::2, 0::2, 0] = r
    rgb[1::2, 1::2, 0] = r
    
    # G channel - 平均 Gr 和 Gb
    rgb[0::2, 1::2, 1] = gr
    rgb[1::2, 0::2, 1] = gb
    # 对于 R 和 B 位置，使用最近邻
    rgb[0::2, 0::2, 1] = (gr + gb[:, :gb.shape[1]]) / 2 if gr.shape == gb.shape else gr
    rgb[1::2, 1::2, 1] = (gr[:gr.shape[0], :] + gb) / 2 if gr.shape == gb.shape else gb
    
    # B channel - 简单上采样
    rgb[1::2, 1::2, 2] = b
    rgb[1::2, 0::2, 2] = b
    rgb[0::2, 1::2, 2] = b
    rgb[0::2, 0::2, 2] = b
    
    return rgb


def apply_white_balance(rgb: np.ndarray, metadata: dict) -> np.ndarray:
    """应用白平衡 (使用 unprocess 的 metadata)"""
    rgb_gain = metadata['rgb_gain'].item()
    red_gain = metadata['red_gain'].item()
    blue_gain = metadata['blue_gain'].item()
    
    # 应用增益
    rgb_out = rgb.copy()
    rgb_out[..., 0] *= red_gain * rgb_gain
    rgb_out[..., 1] *= rgb_gain  # Green
    rgb_out[..., 2] *= blue_gain * rgb_gain
    
    return np.clip(rgb_out, 0, 1)


def apply_ccm(rgb: np.ndarray, cam2rgb: torch.Tensor) -> np.ndarray:
    """应用颜色校正矩阵"""
    cam2rgb_np = cam2rgb.numpy()
    h, w, c = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)
    rgb_corrected = rgb_flat @ cam2rgb_np.T
    return rgb_corrected.reshape(h, w, 3)


def gamma_correction(rgb: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Gamma 校正"""
    return np.clip(rgb, 0, 1) ** gamma


def test_roundtrip_conversion():
    """
    测试往复转换：sRGB -> RAW -> sRGB
    观察质量损失
    """
    print("=" * 80)
    print("测试 Unprocess → ISP 往复转换")
    print("=" * 80)
    
    # 1. 加载原始 sRGB 图像
    img_path = "images/uestc.jpg"
    img_pil = Image.open(img_path).resize((640, 360))  # 缩小以便可视化
    transform = transforms.ToTensor()
    img_srgb = transform(img_pil)
    
    print(f"\n[步骤 1] 原始 sRGB 图像")
    print(f"  - Shape: {img_srgb.shape}")
    print(f"  - Range: [{img_srgb.min():.4f}, {img_srgb.max():.4f}]")
    
    # 2. Unprocess: sRGB → RAW
    print(f"\n[步骤 2] Unprocess (sRGB → RAW)")
    raw_bayer_4ch, metadata = unprocess.unprocess(img_srgb)
    print(f"  - RAW Bayer 4ch shape: {raw_bayer_4ch.shape}")
    print(f"  - cam2rgb matrix:\n{metadata['cam2rgb']}")
    
    # 3. 转换为标准 RAW 格式 (H, W)
    raw_single_ch = bayer_to_raw_format(raw_bayer_4ch)
    print(f"\n[步骤 3] 转换为单通道 RAW")
    print(f"  - RAW shape: {raw_single_ch.shape}")
    print(f"  - Range: [{raw_single_ch.min():.4f}, {raw_single_ch.max():.4f}]")
    
    # 4. 简单 ISP Pipeline: RAW → RGB
    print(f"\n[步骤 4] ISP Pipeline (RAW → RGB)")
    
    # 4a. Demosaicing
    rgb_demosaic = simple_demosaic(raw_single_ch)
    print(f"  4a. Demosaicing: {rgb_demosaic.shape}")
    
    # 4b. White Balance
    rgb_wb = apply_white_balance(rgb_demosaic, metadata)
    print(f"  4b. White Balance applied")
    
    # 4c. Color Correction Matrix
    rgb_ccm = apply_ccm(rgb_wb, metadata['cam2rgb'])
    print(f"  4c. CCM applied")
    
    # 4d. Gamma Correction
    rgb_gamma = gamma_correction(np.clip(rgb_ccm, 0, 1), gamma=0.5)
    print(f"  4d. Gamma correction (^0.5)")
    
    # 5. 对比原始图像和往复后的图像
    print(f"\n[步骤 5] 质量评估")
    img_srgb_np = img_srgb.permute(1, 2, 0).numpy()
    
    # 计算 MSE 和 PSNR
    mse = np.mean((img_srgb_np - rgb_gamma) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    print(f"  - MSE: {mse:.6f}")
    print(f"  - PSNR: {psnr:.2f} dB")
    print(f"  - 原始范围: [{img_srgb_np.min():.4f}, {img_srgb_np.max():.4f}]")
    print(f"  - 往复范围: [{rgb_gamma.min():.4f}, {rgb_gamma.max():.4f}]")
    
    # 6. 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始 sRGB
    axes[0, 0].imshow(img_srgb_np)
    axes[0, 0].set_title("原始 sRGB")
    axes[0, 0].axis('off')
    
    # RAW (R 通道)
    axes[0, 1].imshow(raw_single_ch, cmap='gray')
    axes[0, 1].set_title("RAW (Bayer Pattern)")
    axes[0, 1].axis('off')
    
    # Demosaicing 后
    axes[0, 2].imshow(np.clip(rgb_demosaic, 0, 1))
    axes[0, 2].set_title("Demosaicing")
    axes[0, 2].axis('off')
    
    # White Balance 后
    axes[1, 0].imshow(np.clip(rgb_wb, 0, 1))
    axes[1, 0].set_title("After White Balance")
    axes[1, 0].axis('off')
    
    # 最终往复结果
    axes[1, 1].imshow(np.clip(rgb_gamma, 0, 1))
    axes[1, 1].set_title(f"往复后 (PSNR: {psnr:.1f}dB)")
    axes[1, 1].axis('off')
    
    # 差异图
    diff = np.abs(img_srgb_np - rgb_gamma)
    axes[1, 2].imshow(diff * 10)  # 放大10倍以便观察
    axes[1, 2].set_title("差异图 (x10)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/unprocess_isp_roundtrip.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: /tmp/unprocess_isp_roundtrip.png")
    
    print("\n" + "=" * 80)
    print("结论：")
    print("=" * 80)
    print("1. Unprocess 和 ISP Pipeline 不是完全互逆的")
    print("2. 主要误差来源：")
    print("   - Demosaicing 插值损失")
    print("   - Gamma 值不匹配 (2.2 vs 0.5)")
    print("   - 缺少 inverse_smoothstep 的逆操作")
    print(f"3. 往复后 PSNR = {psnr:.2f} dB (理想应该是 ∞)")
    print("=" * 80)


if __name__ == "__main__":
    test_roundtrip_conversion()

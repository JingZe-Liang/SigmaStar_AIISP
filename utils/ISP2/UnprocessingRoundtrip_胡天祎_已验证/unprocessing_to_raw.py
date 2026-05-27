import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from third_party.unprocessing_torch.dataloader import unprocess


def test_img_unprocess():
    """
    测试 unprocessing 流程：将 sRGB 图像转换为模拟的 RAW 数据
    
    Unprocessing 的主要步骤：
    1. 逆向全局色调映射 (inverse_smoothstep)
    2. 逆向 gamma 压缩 (gamma_expansion)
    3. 逆向颜色校正 (apply_ccm with rgb2cam)
    4. 逆向白平衡和亮度调整 (safe_invert_gains)
    5. 应用 Bayer 马赛克模式 (mosaic)
    """
    print("=" * 80)
    print("开始测试 Unprocessing 流程")
    print("=" * 80)
    
    # 1. 加载图像
    img_path = "images/uestc.jpg"
    img_pil = Image.open(img_path)
    print(f"\n[步骤 1] 加载原始图像")
    print(f"  - 图像路径: {img_path}")
    print(f"  - PIL 图像尺寸: {img_pil.size} (W x H)")
    print(f"  - PIL 图像模式: {img_pil.mode}")
    
    # 2. 转换为 Tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img_pil)
    print(f"\n[步骤 2] 转换为 Tensor")
    print(f"  - Tensor 形状: {img_tensor.shape} (C x H x W)")
    print(f"  - 数据类型: {img_tensor.dtype}")
    print(f"  - 数值范围: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    print(f"  - 均值: {img_tensor.mean():.4f}")
    print(f"  - 标准差: {img_tensor.std():.4f}")
    
    # 3. Unprocess - 将 sRGB 转换为 RAW
    print(f"\n[步骤 3] 执行 Unprocessing (sRGB -> RAW)")
    print(f"  Unprocessing 包含以下子步骤：")
    print(f"    a) 逆向全局色调映射 (inverse_smoothstep)")
    print(f"    b) 逆向 gamma 压缩 (gamma_expansion, gamma=2.2)")
    print(f"    c) 逆向颜色校正 (apply_ccm with random RGB->Camera CCM)")
    print(f"    d) 逆向白平衡和亮度 (safe_invert_gains)")
    print(f"    e) 应用 Bayer 马赛克 (mosaic: RGGB pattern)")
    
    unproc_img, metadata = unprocess.unprocess(img_tensor)
    
    print(f"\n[步骤 4] Unprocessing 结果")
    print(f"  - 输出形状: {unproc_img.shape} (4 x H/2 x W/2)")
    print(f"    * 4 个通道代表 Bayer pattern: [R, Gr, Gb, B]")
    print(f"    * 空间分辨率减半 (因为 Bayer mosaic)")
    print(f"  - 数据类型: {unproc_img.dtype}")
    print(f"  - 数值范围: [{unproc_img.min():.4f}, {unproc_img.max():.4f}]")
    print(f"  - 均值: {unproc_img.mean():.4f}")
    print(f"  - 标准差: {unproc_img.std():.4f}")
    
    # 4. 打印 metadata
    print(f"\n[步骤 5] Metadata (用于后续恢复 RGB)")
    print(f"  - cam2rgb 矩阵形状: {metadata['cam2rgb'].shape}")
    print(f"  - cam2rgb 矩阵:\n{metadata['cam2rgb']}")
    print(f"  - rgb_gain: {metadata['rgb_gain'].item():.4f} (整体亮度增益)")
    print(f"  - red_gain: {metadata['red_gain'].item():.4f} (红色通道白平衡)")
    print(f"  - blue_gain: {metadata['blue_gain'].item():.4f} (蓝色通道白平衡)")
    
    # 5. 添加噪声
    print(f"\n[步骤 6] 添加噪声 (模拟真实相机传感器噪声)")
    shot_noise, read_noise = unprocess.random_noise_levels()
    print(f"  - shot_noise (散粒噪声): {shot_noise.item():.6f}")
    print(f"    * 与图像强度成正比的噪声")
    print(f"  - read_noise (读取噪声): {read_noise.item():.6f}")
    print(f"    * 与图像强度无关的固定噪声")
    
    noisy_img = unprocess.add_noise(unproc_img, shot_noise, read_noise)
    print(f"\n  添加噪声后:")
    print(f"  - 噪声图像形状: {noisy_img.shape}")
    print(f"  - 数值范围: [{noisy_img.min():.4f}, {noisy_img.max():.4f}]")
    print(f"  - 均值: {noisy_img.mean():.4f}")
    print(f"  - 标准差: {noisy_img.std():.4f}")
    
    # 6. 计算方差
    variance = shot_noise * noisy_img + read_noise
    print(f"\n  噪声方差:")
    print(f"  - 方差形状: {variance.shape}")
    print(f"  - 方差范围: [{variance.min():.6f}, {variance.max():.6f}]")
    print(f"  - 方差均值: {variance.mean():.6f}")
    
    # 7. 每个 Bayer 通道的统计信息
    print(f"\n[步骤 7] Bayer 通道详细统计")
    channel_names = ['R (红色)', 'Gr (绿色-红行)', 'Gb (绿色-蓝行)', 'B (蓝色)']
    for i, name in enumerate(channel_names):
        channel = noisy_img[i]
        print(f"  通道 {i} - {name}:")
        print(f"    - 形状: {channel.shape}")
        print(f"    - 范围: [{channel.min():.4f}, {channel.max():.4f}]")
        print(f"    - 均值: {channel.mean():.4f}")
        print(f"    - 标准差: {channel.std():.4f}")
    
    print("\n" + "=" * 80)
    print("总结: Unprocessing 流程")
    print("=" * 80)
    print(f"输入: sRGB 图像 {img_tensor.shape} -> 输出: RAW Bayer 图像 {unproc_img.shape}")
    print(f"这个过程模拟了相机 ISP (Image Signal Processor) 的逆过程")
    print(f"可用于训练去噪模型，因为它提供了接近真实相机传感器的数据")
    print("=" * 80)

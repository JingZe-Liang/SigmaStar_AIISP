"""
ISP (Image Signal Processing) Module for Raw to sRGB Conversion

实现流程:
1. Pack GBRG Bayer pattern: (H, W) → (4, H/2, W/2)
2. Demosaic: (4, H/2, W/2) → (3, H, W)
3. White Balance: Gray World 算法
4. Color Correction Matrix (CCM): 相机 RGB → sRGB
5. Gamma Correction: 2.2 (sRGB 标准)

参数配置:
- Sensor: Sony IMX385 (STARVIS)
- Bayer Pattern: GBRG
- Gamma: 2.2
- CCM: 通用 sRGB 优化矩阵

注意事项:
- **输入数据应该已经完成去黑电平和归一化到 [0, 1]**
- 本模块不负责归一化与去除黑电平,只负责 ISP 转换流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISPModule(nn.Module):
    """
    ISP Module for converting raw Bayer images to sRGB

    输入: (B, T, H, W) - Bayer raw 序列,或 (B, H, W) - 单帧
          **假设输入已经归一化到 [0, 1] 范围**
    输出: (B, T, 3, H, W) - sRGB 序列,或 (B, 3, H, W) - 单帧

    参数:
        black_level: 黑电平(保留参数,但不使用)
        white_level: 白电平(保留参数,但不使用)
        gamma: 伽马值,默认 2.2
        wb_gains: 白平衡增益 [R, G, B],None 则使用 Gray World
        ccm: 3x3 色彩校正矩阵,None 则使用默认

    注意:
        - 输入数据应该已经完成去黑电平和归一化
        - ISP 模块只负责: Pack → Demosaic → WB → CCM → Gamma
    """

    def __init__(
            self,
            black_level: int = 240,
            white_level: int = 4095,
            gamma: float = 2.2,
            wb_gains: list[float] | None = None,
            ccm: list[list[float]] | None = None,
    ) -> None:
        super().__init__()

        self.black_level = black_level
        self.white_level = white_level
        self.gamma = gamma

        # 白平衡增益 (如果提供)
        if wb_gains is not None:
            self.register_buffer("wb_gains", torch.tensor(wb_gains, dtype=torch.float32))
        else:
            self.wb_gains = None

        # CCM 矩阵: 使用稳健的通用 sRGB 优化矩阵
        # 基于标准 D65 光源的典型 CCM
        if ccm is None:
            ccm = [
                [1.6, -0.4, -0.2],  # R 通道
                [-0.3, 1.5, -0.2],  # G 通道
                [-0.1, -0.5, 1.6],  # B 通道
            ]
        self.register_buffer("ccm", torch.tensor(ccm, dtype=torch.float32))

    def pack_gbrg_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Pack GBRG Bayer pattern into 4 channels

        GBRG pattern:
        G B G B ...
        R G R G ...

        Packing 顺序 (与 utils.py 一致):
        [0] R: raw[1::2, 0::2]
        [1] G1: raw[1::2, 1::2]
        [2] B: raw[0::2, 1::2]
        [3] G2: raw[0::2, 0::2]

        输入: (..., H, W) - 原始 raw 数据(未归一化)
        输出: (..., 4, H/2, W/2)

        注意: 不做归一化和去黑电平,假设输入已经预处理
        """
        raw = raw.float()

        # GBRG packing (不做归一化)
        r = raw[..., 1::2, 0::2]  # R
        g1 = raw[..., 1::2, 1::2]  # G1
        b = raw[..., 0::2, 1::2]  # B
        g2 = raw[..., 0::2, 0::2]  # G2

        packed = torch.stack([r, g1, b, g2], dim=-3)  # (..., 4, H/2, W/2)
        return packed

    def demosaic(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Demosaic: 4 channels (R, G1, B, G2) → 3 channels (R, G, B)

        使用双线性插值重建完整 RGB 图像

        输入: (..., 4, H/2, W/2)
        输出: (..., 3, H, W)
        """
        # 解包 RGBG
        r = packed[..., 0, :, :]  # (…, H/2, W/2)
        g1 = packed[..., 1, :, :]
        b = packed[..., 2, :, :]
        g2 = packed[..., 3, :, :]

        # 平均两个 G 通道
        g_avg = (g1 + g2) / 2.0  # (…, H/2, W/2)

        # 上采样到原始分辨率
        # 使用 bilinear 插值
        r_full = self._upsample_channel(r, packed.shape[-2] * 2, packed.shape[-1] * 2)
        g_full = self._upsample_channel(
            g_avg, packed.shape[-2] * 2, packed.shape[-1] * 2
        )
        b_full = self._upsample_channel(b, packed.shape[-2] * 2, packed.shape[-1] * 2)

        rgb = torch.stack([r_full, g_full, b_full], dim=-3)  # (..., 3, H, W)
        return rgb

    def _upsample_channel(
            self, channel: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """双线性插值上采样单通道"""
        # 需要 4D input for F.interpolate: (N, C, H, W)
        orig_shape = channel.shape
        channel = channel.reshape(-1, 1, orig_shape[-2], orig_shape[-1])

        upsampled = F.interpolate(
            channel, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        # 恢复原始 batch shape
        upsampled = upsampled.reshape(*orig_shape[:-2], target_h, target_w)
        return upsampled

    def white_balance(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        White Balance using Gray World algorithm

        如果提供了 wb_gains,则使用固定增益
        否则使用 Gray World: 假设场景平均为灰色

        输入: (..., 3, H, W)
        输出: (..., 3, H, W)
        """
        if self.wb_gains is not None:
            # 使用固定增益
            gains = self.wb_gains.view(1, 3, 1, 1)  # (1, 3, 1, 1)
            return rgb * gains
        else:
            # Gray World 算法
            # 计算每个通道的平均值
            r_mean = rgb[..., 0, :, :].mean(dim=(-2, -1), keepdim=True)
            g_mean = rgb[..., 1, :, :].mean(dim=(-2, -1), keepdim=True)
            b_mean = rgb[..., 2, :, :].mean(dim=(-2, -1), keepdim=True)

            # 计算增益 (以 G 为参考)
            r_gain = g_mean / (r_mean + 1e-8)
            b_gain = g_mean / (b_mean + 1e-8)

            # 应用增益
            rgb_wb = rgb.clone()
            rgb_wb[..., 0, :, :] = rgb[..., 0, :, :] * r_gain
            rgb_wb[..., 2, :, :] = rgb[..., 2, :, :] * b_gain

            return rgb_wb

    def apply_ccm(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Apply Color Correction Matrix

        RGB_out = CCM @ RGB_in

        输入: (..., 3, H, W)
        输出: (..., 3, H, W)
        """
        # 重塑为 (..., 3, H*W)
        orig_shape = rgb.shape
        rgb_flat = rgb.reshape(*orig_shape[:-2], 3, -1)  # (..., 3, H*W)

        # 矩阵乘法: (3, 3) @ (3, H*W) = (3, H*W)
        rgb_corrected = torch.matmul(
            self.ccm, rgb_flat
        )  # self.ccm: (3, 3), rgb_flat: (..., 3, N)

        # 恢复形状
        rgb_corrected = rgb_corrected.reshape(orig_shape)

        # Clamp to [0, 1]
        rgb_corrected = torch.clamp(rgb_corrected, 0.0, 1.0)

        return rgb_corrected

    def gamma_correction(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Gamma Correction: sRGB 标准

        输入: (..., 3, H, W) - Linear RGB [0, 1]
        输出: (..., 3, H, W) - sRGB [0, 1]
        """
        # sRGB gamma: y = x^(1/2.2) for x > 0
        # 避免 0 值的数值问题
        srgb = torch.pow(rgb + 1e-8, 1.0 / self.gamma)
        return srgb

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Complete ISP pipeline

        输入:
            - (B, T, H, W): 序列模式
            - (B, H, W): 单帧模式

        输出:
            - (B, T, 3, H, W): 序列模式
            - (B, 3, H, W): 单帧模式
        """
        is_sequence = raw.ndim == 4  # (B, T, H, W)

        if not is_sequence and raw.ndim != 3:
            raise ValueError(f"Input must be 3D (B, H, W) or 4D (B, T, H, W), got {raw.ndim}D")

        # Step 1: Pack GBRG
        packed = self.pack_gbrg_raw(raw)  # (..., 4, H/2, W/2)

        # Step 2: Demosaic
        rgb = self.demosaic(packed)  # (..., 3, H, W)

        # Step 3: White Balance
        rgb = self.white_balance(rgb)

        # Step 4: CCM
        rgb = self.apply_ccm(rgb)

        # Step 5: Gamma Correction
        srgb = self.gamma_correction(rgb)

        return srgb


def test_isp_module() -> None:
    """测试 ISP 模块"""
    print("=== 测试 ISP Module ===\n")

    # 创建 ISP 模块
    isp = ISPModule()

    # 测试 1: 单帧模式 (假设输入已归一化到 [0, 1])
    print("测试 1: 单帧模式")
    raw_single = torch.rand(2, 1080, 1920)  # (B, H, W) - [0, 1] 范围
    srgb_single = isp(raw_single)
    print(f"输入形状: {raw_single.shape}")
    print(f"输入范围: [{raw_single.min():.4f}, {raw_single.max():.4f}]")
    print(f"输出形状: {srgb_single.shape}")
    print(f"输出范围: [{srgb_single.min():.4f}, {srgb_single.max():.4f}]\n")

    # 测试 2: 序列模式
    print("测试 2: 序列模式")
    raw_seq = torch.rand(2, 7, 1080, 1920)  # (B, T, H, W) - [0, 1] 范围
    srgb_seq = isp(raw_seq)
    print(f"输入形状: {raw_seq.shape}")
    print(f"输入范围: [{raw_seq.min():.4f}, {raw_seq.max():.4f}]")
    print(f"输出形状: {srgb_seq.shape}")
    print(f"输出范围: [{srgb_seq.min():.4f}, {srgb_seq.max():.4f}]\n")

    # 测试 3: 自定义参数
    print("测试 3: 自定义白平衡增益")
    isp_custom = ISPModule(wb_gains=[1.2, 1.0, 1.5])
    srgb_custom = isp_custom(raw_single)
    print(f"输出形状: {srgb_custom.shape}")
    print(f"输出范围: [{srgb_custom.min():.4f}, {srgb_custom.max():.4f}]\n")

    print("✓ 所有测试通过!")
    print("\n注意: 输入数据应该已经归一化到 [0, 1] 范围")


if __name__ == "__main__":
    test_isp_module()
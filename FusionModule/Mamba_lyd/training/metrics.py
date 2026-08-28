from __future__ import annotations

"""Validation metrics for RAW fusion training.

本模块负责计算RAW图像融合训练的验证指标：
1. PSNR（峰值信噪比）：衡量重建质量的核心指标
2. SSIM（结构相似性）：衡量结构保持能力
3. SNR（信噪比）：衡量信号与噪声的比值
4. 运动区域PSNR：专门评估拖影/运动区域的去噪效果
5. 权重统计：监控融合权重的分布特性
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .tensor_utils import ensure_image3d, ensure_image4d


@dataclass
class AverageMeter:
    """平均值累加器，用于在线计算指标的平均值。
    
    采用总和+计数的方式避免数值精度问题，适合大规模数据累积。
    
    Attributes:
        total: 累计总和（加权累加，考虑每个值的样本数n）
        count: 累计样本总数
        
    Example:
        >>> meter = AverageMeter()
        >>> meter.update(0.5, n=10)  # 10个样本，每个值为0.5
        >>> meter.update(0.8, n=5)   # 5个样本，每个值为0.8
        >>> meter.avg  # (0.5*10 + 0.8*5) / 15 = 0.6
        0.6
    """
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        """更新累加器，添加新的观测值。
        
        Args:
            value: 单个观测值（如一个batch的指标值）
            n: 该值代表的样本数量（通常为batch_size）
            
        Note:
            使用加权累加：total += value × n，确保不同batch大小的正确平均
        """
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        """计算当前平均值。
        
        Returns:
            float: 累计平均值，如果count为0则返回0.0（避免除零错误）
        """
        return self.total / max(1, self.count)


class MetricAccumulator:
    """验证指标累加器，跨多个batch累积并计算最终指标。
    
    在验证阶段，对每个batch调用update()方法累积统计量，
    最后调用compute()方法计算所有指标的最终值。
    
    支持的指标分类:
    1. 全局PSNR指标: pred/dnr2/dnr3相对于clean的PSNR
    2. 运动区域PSNR: 专门评估高运动区域的去噪性能
    3. 结构相似性: SSIM分数
    4. 信噪比: SNR
    5. 权重统计: 融合权重的均值、标准差、Bayer平面差异
    
    Attributes:
        sse_pred/sse_dnr2/sse_dnr3: 预测/2DNR/3DNR的平方误差总和
        signal: clean图像的平方和（信号能量）
        count: 总像素数
        motion_sse_pred/motion_sse_dnr3: 运动区域的平方误差总和
        motion_count: 运动区域的像素数
        ssim: SSIM平均值累加器
        weight_mean/weight_std: 权重均值/标准差累加器
        plane_gap: W4模式下Bayer平面间差异累加器
    """
    def __init__(self) -> None:
        """初始化所有累加器状态为零。"""
        self.sse_pred = 0.0          # 预测值的平方误差总和
        self.sse_dnr2 = 0.0          # 2DNR的平方误差总和
        self.sse_dnr3 = 0.0          # 3DNR的平方误差总和
        self.signal = 0.0            # clean信号的平方和
        self.count = 0               # 总像素计数
        self.motion_sse_pred = 0.0   # 运动区域预测的平方误差总和
        self.motion_sse_dnr3 = 0.0   # 运动区域3DNR的平方误差总和
        self.motion_count = 0        # 运动区域像素计数
        self.ssim = AverageMeter()   # SSIM累加器
        self.weight_mean = AverageMeter()    # 权重均值累加器
        self.weight_std = AverageMeter()     # 权重标准差累加器
        self.plane_gap = AverageMeter()      # Bayer平面差异累加器

    @torch.no_grad()
    def update(self, batch: dict[str, Any], output: Any) -> None:
        """处理单个batch的预测结果，累积各项指标统计量。
        
        Args:
            batch: 数据batch字典，包含以下键:
                - clean: 真实标签图像 [B, H, W] 或 [B, 1, H, W]
                - dnr2: 2DNR去噪结果 [B, H, W] 或 [B, 1, H, W]
                - dnr3: 3DNR去噪结果 [B, H, W] 或 [B, 1, H, W]
                - motion_prior: 运动先验特征 [B, 4, H/2, W/2]
            output: 模型输出对象，包含以下属性:
                - prediction: 融合预测结果 [B, H, W] 或 [B, 1, H, W]
                - weight: 全分辨率融合权重 [B, 1, 2H, 2W]
                - packed_weight: 打包分辨率权重 [B, 1, H, W](W1) 或 [B, 4, H, W](W4)
                
        Note:
            张量形状转换流程:
            1. ensure_image3d将[B,1,H,W]转为[B,H,W]统一格式
            2. 计算全局SSE时展平所有空间维度
            3. build_motion_mask从[B,4,H/2,W/2]生成[B,H,W]二值掩码
            4. SSIM计算时将[B,H,W]转回[B,1,H,W]进行卷积操作
            
            所有张量都调用.detach()断开梯度图，节省内存
        """
        # 步骤1: 标准化张量形状为[B, H, W]格式（3D图像张量）
        # 输入可能是[B, H, W]或[B, 1, H, W]，统一转换为[B, H, W]
        pred = ensure_image3d(output.prediction).detach()      # [B, H, W]
        clean = ensure_image3d(batch["clean"]).detach()         # [B, H, W]
        dnr2 = ensure_image3d(batch["dnr2"]).detach()           # [B, H, W]
        dnr3 = ensure_image3d(batch["dnr3"]).detach()           # [B, H, W]

        # 步骤2: 累积全局平方误差（用于PSNR计算）
        # squared_error_sum内部计算: sum((pred[i] - clean[i])^2) 对所有像素求和
        self.sse_pred += squared_error_sum(pred, clean)        # 标量累加
        self.sse_dnr2 += squared_error_sum(dnr2, clean)        # 标量累加
        self.sse_dnr3 += squared_error_sum(dnr3, clean)        # 标量累加
        
        # 步骤3: 累积信号能量（用于SNR计算）
        # clean.float() ** 2: [B, H, W] → [B, H, W] (逐元素平方)
        # .sum(): [B, H, W] → 标量 (所有像素求和)
        self.signal += float((clean.float() ** 2).sum().item())
        
        # 步骤4: 累积像素总数（用于MSE = SSE / count）
        # clean.numel()返回张量中元素总数 = B × H × W
        self.count += int(clean.numel())

        # 步骤5: 构建运动区域掩码并累积运动区域指标
        # build_motion_mask: [B, 4, H/2, W/2] → [B, H, W] (二值掩码)
        motion_mask = build_motion_mask(batch["motion_prior"], clean.shape[-2:])
        if motion_mask.any():  # 仅在存在运动区域时计算
            # masked_squared_error_sum: 仅对mask=True的像素计算平方误差
            self.motion_sse_pred += masked_squared_error_sum(pred, clean, motion_mask)
            self.motion_sse_dnr3 += masked_squared_error_sum(dnr3, clean, motion_mask)
            # motion_mask.sum()计算运动区域的像素总数
            self.motion_count += int(motion_mask.sum().item())

        # 步骤6: 累积SSIM分数（批量计算）
        # ssim_score: ([B, H, W], [B, H, W]) → 标量float (整个batch的平均SSIM)
        # pred.shape[0]是batch size B，作为样本数传入AverageMeter
        self.ssim.update(ssim_score(pred, clean), pred.shape[0])
        
        # 步骤7: 累积权重统计信息
        # output.weight: [B, 1, 2H, 2W]，计算整个batch的均值和标准差
        self.weight_mean.update(float(output.weight.detach().mean().item()), pred.shape[0])
        self.weight_std.update(float(output.weight.detach().std().item()), pred.shape[0])
        
        # 步骤8: 如果是W4模式，累积Bayer平面间的差异
        # output.packed_weight: [B, 4, H, W] (W4模式下有4个Bayer通道)
        if output.packed_weight.shape[1] == 4:
            # mean(dim=(0, 2, 3)): [B, 4, H, W] → [4] (每个Bayer通道的全局均值)
            plane_means = output.packed_weight.detach().mean(dim=(0, 2, 3))
            # plane_means.max() - plane_means.min(): [4] → 标量 (最大差异)
            self.plane_gap.update(float((plane_means.max() - plane_means.min()).item()), pred.shape[0])

    def compute(self) -> dict[str, float]:
        """计算并返回所有累积的最终指标值。
        
        基于累积的平方误差总和、像素计数等统计量，计算各种PSNR、SNR、SSIM指标。
        
        Returns:
            dict: 包含以下指标的字典:
                - psnr: 预测结果的PSNR (dB)
                - psnr_dnr2: 2DNR的PSNR (dB)
                - psnr_dnr3: 3DNR的PSNR (dB)
                - psnr_gain_dnr2: 预测相对2DNR的PSNR提升 (dB)
                - psnr_gain_dnr3: 预测相对3DNR的PSNR提升 (dB)
                - snr: 预测结果的信噪比 (dB)
                - ssim: 结构相似性 (0~1)
                - motion_psnr: 运动区域预测的PSNR (dB)
                - motion_psnr_dnr3: 运动区域3DNR的PSNR (dB)
                - motion_psnr_gain_dnr3: 运动区域预测相对3DNR的PSNR提升 (dB)
                - weight_mean: 融合权重的平均值 (0~1)
                - weight_std: 融合权重的标准差
                - plane_mean_gap: W4模式下Bayer平面均值的最大差异
                
        Note:
            PSNR计算公式: PSNR = -10 × log10(MSE) = -10 × log10(SSE / count)
            SNR计算公式: SNR = 10 × log10(signal / noise) = 10 × log10(sum(x²) / sum(e²))
            PSNR gain > 0表示融合结果优于对应的DN方法
        """
        # 计算全局PSNR指标
        pred_psnr = psnr_from_sse(self.sse_pred, self.count)       # 预测PSNR
        dnr2_psnr = psnr_from_sse(self.sse_dnr2, self.count)       # 2DNR PSNR
        dnr3_psnr = psnr_from_sse(self.sse_dnr3, self.count)       # 3DNR PSNR
        
        # 计算运动区域PSNR指标
        motion_psnr = psnr_from_sse(self.motion_sse_pred, self.motion_count)           # 运动区预测PSNR
        motion_dnr3_psnr = psnr_from_sse(self.motion_sse_dnr3, self.motion_count)      # 运动区3DNR PSNR
        
        return {
            "psnr": pred_psnr,                                    # 核心指标：融合结果质量
            "psnr_dnr2": dnr2_psnr,                               # 基线1：2DNR质量
            "psnr_dnr3": dnr3_psnr,                               # 基线2：3DNR质量
            "psnr_gain_dnr2": pred_psnr - dnr2_psnr,              # 相对2DNR的提升
            "psnr_gain_dnr3": pred_psnr - dnr3_psnr,              # 相对3DNR的提升
            "snr": snr_from_sums(self.signal, self.sse_pred),     # 信噪比
            "ssim": self.ssim.avg,                                # 结构相似性
            "motion_psnr": motion_psnr,                           # 运动区域质量
            "motion_psnr_dnr3": motion_dnr3_psnr,                 # 运动区域3DNR质量
            "motion_psnr_gain_dnr3": motion_psnr - motion_dnr3_psnr,  # 运动区域提升
            "weight_mean": self.weight_mean.avg,                  # 权重分布中心
            "weight_std": self.weight_std.avg,                    # 权重分布离散度
            "plane_mean_gap": self.plane_gap.avg,                 # Bayer平面一致性
        }


def build_motion_mask(motion_prior: Tensor, target_hw: tuple[int, int]) -> Tensor:
    """基于运动先验构建二值运动区域掩码。
    
    通过阈值分割识别高运动区域（前20%的高运动像素），用于专门评估
    拖影/快速运动场景下的去噪性能。
    
    Args:
        motion_prior: 运动先验特征 [B, 4, H/2, W/2]，来自|curr4-prev4|
        target_hw: 目标掩码的空间尺寸 (H, W)，通常与clean图像一致
        
    Returns:
        mask: 二值掩码 [B, H, W]，True表示运动区域，False表示静止区域
        
    Note:
        处理流程:
        1. 通道平均: [B, 4, H/2, W/2] → [B, H/2, W/2] (4个Bayer通道取平均)
        2. 计算80%分位数阈值: [B, H/2, W/2] → [B, 1, 1] (每帧独立阈值)
        3. 阈值分割: [B, H/2, W/2] > [B, 1, 1] → [B, H/2, W/2] (布尔掩码)
        4. 2×上采样: [B, H/2, W/2] → [B, H, W] (恢复到原始分辨率)
        5. 裁剪到目标尺寸: [B, H, W] → [B, target_H, target_W] (防止尺寸不匹配)
        
        关键设计:
        - 使用80%分位数而非固定阈值，自适应不同场景的运动强度
        - 每帧独立计算阈值，避免batch内场景差异的影响
        - repeat_interleave实现最近邻上采样，保持掩码的二值特性
        
    Example:
        >>> motion_prior = torch.rand(2, 4, 128, 128)  # batch=2, 4通道, 128x128
        >>> mask = build_motion_mask(motion_prior, (256, 256))
        >>> mask.shape
        torch.Size([2, 256, 256])
        >>> mask.dtype
        torch.bool
    """
    # 步骤1: 对4个Bayer通道求平均，得到单通道运动强度图
    # motion_prior: [B, 4, H/2, W/2] → motion: [B, H/2, W/2]
    motion = motion_prior.mean(dim=1)
    
    # 步骤2: 计算每帧的80%分位数作为运动阈值
    # motion.flatten(1): [B, H/2, W/2] → [B, (H/2)*(W/2)] (展平空间维度)
    # torch.quantile(..., 0.80, dim=1): [B, (H/2)*(W/2)] → [B] (每帧一个阈值)
    # .view(-1, 1, 1): [B] → [B, 1, 1] (广播兼容形状)
    threshold = torch.quantile(motion.flatten(1), 0.80, dim=1).view(-1, 1, 1)
    
    # 步骤3: 阈值分割，标记高于阈值的像素为运动区域
    # motion: [B, H/2, W/2], threshold: [B, 1, 1]
    # 广播比较: [B, H/2, W/2] > [B, 1, 1] → mask: [B, H/2, W/2] (布尔张量)
    mask = motion > threshold
    
    # 步骤4: 2×上采样恢复到原始分辨率
    # 高度方向2×: [B, H/2, W/2] → [B, H, W/2]
    # 宽度方向2×: [B, H, W/2] → [B, H, W]
    mask = mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    
    # 步骤5: 裁剪到目标尺寸（防止因crop导致的尺寸不匹配）
    # mask: [B, H, W] → mask[..., :target_H, :target_W]: [B, target_H, target_W]
    return mask[..., : target_hw[0], : target_hw[1]]


def ssim_score(pred: Tensor, target: Tensor) -> float:
    """计算结构相似性指数(SSIM)。
    
    SSIM综合考虑亮度、对比度和结构三个维度的相似性，比PSNR更符合人眼感知。
    使用11×11高斯窗口进行局部统计计算。
    
    Args:
        pred: 预测图像 [B, H, W] 或 [B, 1, H, W]
        target: 真实图像 [B, H, W] 或 [B, 1, H, W]
        
    Returns:
        float: 整个batch的平均SSIM分数，范围[0, 1]，越接近1表示越相似
        
    Note:
        SSIM计算公式:
        SSIM(x, y) = [(2μₓμᵧ + C₁)(2σₓᵧ + C₂)] / [(μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)]
        
        其中:
        - μₓ, μᵧ: 局部均值（亮度）
        - σₓ², σᵧ²: 局部方差（对比度）
        - σₓᵧ: 局部协方差（结构）
        - C₁ = (K₁L)², C₂ = (K₂L)²: 稳定常数（K₁=0.01, K₂=0.03, L=1）
        
        处理流程:
        1. 确保4D格式: [B, H, W] → [B, 1, H, W]
        2. 计算局部均值: 11×11平均池化 [B, 1, H, W] → [B, 1, H, W]
        3. 计算局部方差: E[X²] - (E[X])²
        4. 计算局部协方差: E[XY] - E[X]E[Y]
        5. 代入SSIM公式计算逐像素分数
        6. 全局平均得到标量SSIM
        
    Example:
        >>> pred = torch.rand(4, 256, 256)
        >>> target = torch.rand(4, 256, 256)
        >>> ssim = ssim_score(pred, target)
        >>> print(f"{ssim:.4f}")  # 如 0.8523
    """
    # 步骤1: 标准化为4D格式 [B, 1, H, W] 并转为float32
    pred4 = ensure_image4d(pred).float()      # [B, H, W] → [B, 1, H, W]
    target4 = ensure_image4d(target).float()  # [B, H, W] → [B, 1, H, W]
    
    # 步骤2: 设置11×11窗口参数（SSIM标准配置）
    kernel_size = 11
    padding = kernel_size // 2  # padding=5，保持输出尺寸不变
    
    # 步骤3: 计算局部均值（亮度分量）
    # 使用11×11平均池化，stride=1保持空间分辨率
    # mu_x: [B, 1, H, W] → [B, 1, H, W] (每个像素是其11×11邻域的平均值)
    mu_x = F.avg_pool2d(pred4, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target4, kernel_size, stride=1, padding=padding)
    
    # 步骤4: 计算局部方差（对比度分量）
    # Var(X) = E[X²] - (E[X])²
    # sigma_x: [B, 1, H, W] (每个像素位置的局部方差)
    sigma_x = F.avg_pool2d(pred4 * pred4, kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target4 * target4, kernel_size, stride=1, padding=padding) - mu_y * mu_y
    
    # 步骤5: 计算局部协方差（结构分量）
    # Cov(X,Y) = E[XY] - E[X]E[Y]
    # sigma_xy: [B, 1, H, W] (pred和target的局部协方差)
    sigma_xy = F.avg_pool2d(pred4 * target4, kernel_size, stride=1, padding=padding) - mu_x * mu_y
    
    # 步骤6: 设置稳定常数（避免除零）
    # C₁ = (K₁ × L)² = (0.01 × 1)² = 0.0001 (亮度稳定性)
    # C₂ = (K₂ × L)² = (0.03 × 1)² = 0.0009 (对比度稳定性)
    c1 = 0.01**2
    c2 = 0.03**2
    
    # 步骤7: 计算SSIM分数
    # 分子: (2μₓμᵧ + C₁)(2σₓᵧ + C₂) - 亮度和结构的相似性
    # 分母: (μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂) - 归一化因子
    # score: [B, 1, H, W] (逐像素SSIM分数)
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12  # 1e-12防止除零
    )
    
    # 步骤8: 全局平均得到标量SSIM
    # score.mean(): [B, 1, H, W] → 标量 (整个batch的平均SSIM)
    # .item(): 提取Python float值
    return float(score.mean().item())


def squared_error_sum(pred: Tensor, target: Tensor) -> float:
    """计算两个张量之间的平方误差总和(SSE)。
    
    SSE = Σ(predᵢ - targetᵢ)²，对所有元素求和。
    这是PSNR计算的基础统计量。
    
    Args:
        pred: 预测张量，任意形状（通常是[B, H, W]）
        target: 真实张量，与pred形状相同
        
    Returns:
        float: 平方误差总和（标量）
        
    Note:
        计算流程:
        1. pred.float() - target.float(): 逐元素相减 [B, H, W] → [B, H, W]
        2. ** 2: 逐元素平方 [B, H, W] → [B, H, W]
        3. .sum(): 所有元素求和 [B, H, W] → 标量
        4. .item(): 提取Python float值
        
        转为float32是为了避免低精度（如float16）导致的数值误差
    """
    return float(((pred.float() - target.float()) ** 2).sum().item())


def masked_squared_error_sum(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    """计算掩码区域内的平方误差总和。
    
    仅对mask=True的像素位置计算平方误差，用于评估特定区域（如运动区域）的性能。
    
    Args:
        pred: 预测张量 [B, H, W]
        target: 真实张量 [B, H, W]
        mask: 二值掩码 [B, H, W]，布尔类型，True表示需要计算的像素
        
    Returns:
        float: 掩码区域内所有像素的平方误差总和（标量）
        
    Note:
        计算流程:
        1. pred.float() - target.float(): [B, H, W] → [B, H, W] (逐元素相减)
        2. ** 2: [B, H, W] → [B, H, W] (逐元素平方)
        3. [mask]: 布尔索引，提取mask=True的元素 → [N] (N为True的数量)
        4. .sum(): [N] → 标量 (求和)
        5. .item(): 提取Python float值
        
        示例:
        如果mask中有1000个True值，则只计算这1000个位置的误差并求和
    """
    return float((((pred.float() - target.float()) ** 2)[mask]).sum().item())


def psnr_from_sse(sse: float, count: int) -> float:
    """从平方误差总和(SSE)和像素数计算PSNR。
    
    PSNR = -10 × log₁₀(MSE) = -10 × log₁₀(SSE / count)
    
    Args:
        sse: 平方误差总和 Σ(predᵢ - targetᵢ)²
        count: 像素总数
        
    Returns:
        float: PSNR值（单位：dB），越高表示质量越好
        - 典型范围: 20~50 dB
        - > 40 dB: 优秀质量
        - 30~40 dB: 良好质量
        - < 30 dB: 较差质量
        
    Note:
        特殊情况处理:
        - count <= 0: 返回NaN（无效输入）
        - MSE < 1e-12: 钳制为1e-12，避免log(0) = -∞
        
        PSNR的物理意义:
        - 假设信号范围为[0, 1]，PSNR表示信号峰值(1.0)与噪声功率(MSE)的比值
        - 每增加6 dB，相当于MSE降低4倍（或RMSE降低2倍）
        
    Example:
        >>> psnr_from_sse(sse=0.01, count=10000)  # MSE = 0.000001
        60.0  # 非常高的PSNR
        >>> psnr_from_sse(sse=100, count=10000)   # MSE = 0.01
        20.0  # 较低的PSNR
    """
    if count <= 0:
        return float("nan")  # 无效输入，返回NaN
    
    # 计算均方误差 MSE = SSE / N
    # 钳制最小值为1e-12，避免log(0)导致-∞
    mse = max(sse / count, 1e-12)
    
    # PSNR = -10 × log₁₀(MSE)
    # 当MSE=1e-12时，PSNR = -10 × (-12) = 120 dB（理论上限）
    # 当MSE=0.01时，PSNR = -10 × (-2) = 20 dB
    return -10.0 * math.log10(mse)


def snr_from_sums(signal: float, noise: float) -> float:
    """从信号能量和噪声能量计算信噪比(SNR)。
    
    SNR = 10 × log₁₀(signal / noise) = 10 × log₁₀(Σx² / Σe²)
    
    Args:
        signal: 信号能量 Σx²（clean图像的平方和）
        noise: 噪声能量 Σe²（预测误差的平方和，即SSE）
        
    Returns:
        float: SNR值（单位：dB），越高表示信噪比越好
        
    Note:
        与PSNR的区别:
        - PSNR: 基于MSE，假设信号范围为[0, 1]，参考点是峰值1.0
        - SNR: 基于实际信号能量，参考点是信号本身的功率
        
        特殊情况处理:
        - signal或noise < 1e-12: 钳制为1e-12，避免除零或log(0)
        
        SNR的物理意义:
        - SNR > 0 dB: 信号功率大于噪声功率（正常情况）
        - SNR = 0 dB: 信号功率等于噪声功率
        - SNR < 0 dB: 噪声功率大于信号功率（严重失真）
        
    Example:
        >>> snr_from_sums(signal=1000, noise=10)  # 信号是噪声的100倍
        20.0  # 10 × log₁₀(100) = 20 dB
        >>> snr_from_sums(signal=100, noise=100)  # 信号等于噪声
        0.0   # 10 × log₁₀(1) = 0 dB
    """
    # SNR = 10 × log₁₀(signal / noise)
    # 钳制最小值为1e-12，避免除零或log(0)
    return 10.0 * math.log10(max(signal, 1e-12) / max(noise, 1e-12))

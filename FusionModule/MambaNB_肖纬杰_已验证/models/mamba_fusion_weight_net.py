from __future__ import annotations

"""Top-level MambaFusionWeightNet-Lite model for RAW 2DNR/3DNR fusion."""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..raw_utils import (
    DEFAULT_RAW_RANGE,
    RawRange,
    bayer_unpack,
    get_cfa_positions,
    prepare_noisy_pair_features,
)
from .layers import ActivationType, build_activation
from .stems import PackedRawStemEncoder, StemConfig, StemFeatures
from .stmamba_config import STMambaLiteConfig
from .stmamba_stack import STMambaLiteStack, STMambaLiteStackOutput


WeightMode = Literal["w1", "w4"]


def _validate_feature_pair(
    current_feature: Tensor,
    motion_feature: Tensor,
    channels: int,
) -> None:
    """验证当前帧特征和运动先验特征的形状一致性。
    
    Args:
        current_feature: 当前帧特征图 [B, C, H, W]
        motion_feature: 运动先验特征图 [B, C, H, W]
        channels: 期望的通道数
    """
    for name, feature_map in {
        "current_feature": current_feature,
        "motion_feature": motion_feature,
    }.items():
        if feature_map.ndim != 4:
            raise ValueError(f"{name} must have shape [B, C, H, W], got {feature_map.shape}")
        if feature_map.shape[1] != channels:
            raise ValueError(f"{name} channel mismatch: expected {channels}, got {feature_map.shape[1]}")
    if current_feature.shape != motion_feature.shape:
        raise ValueError(
            "current_feature and motion_feature must share the same shape, "
            f"got {current_feature.shape} vs {motion_feature.shape}"
        )


@dataclass(frozen=True)
class MambaFusionWeightNetLiteConfig:
    """完整轻量级RAW融合权重网络的配置类。
    
    该配置定义了从stem编码器到backbone主干网络，再到权重预测头的完整网络结构参数。
    """

    stem_config: StemConfig = field(default_factory=StemConfig)  # Stem编码器配置（包含输入输出通道等）
    backbone_config: STMambaLiteConfig = field(default_factory=STMambaLiteConfig)  # ST-Mamba主干网络配置
    weight_mode: WeightMode = "w4"  # 权重模式："w1"(单通道共享权重) 或 "w4"(四通道Bayer独立权重)
    cfa_pattern: str = "GBRG"  # Bayer色彩滤镜阵列模式（如GRBG、GBRG等）
    raw_range: RawRange = DEFAULT_RAW_RANGE  # RAW数据的动态范围配置（black_level, white_level）
    refine_activation: ActivationType = "silu"  # 精炼头使用的激活函数类型
    weight_bias_init: float = 0.0  # 权重预测层偏置的初始值

    def __post_init__(self) -> None:
        if self.stem_config.stem_channels != self.backbone_config.channels:
            raise ValueError(
                "stem_config.stem_channels must match backbone_config.channels, "
                f"got {self.stem_config.stem_channels} vs {self.backbone_config.channels}"
            )
        if self.weight_mode not in {"w1", "w4"}:
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")
        get_cfa_positions(self.cfa_pattern)


@dataclass(frozen=True)
class MambaFusionWeightNetLiteOutput:
    """MambaFusionWeightNetLite的结构化输出容器。
    
    Attributes:
        prediction: 融合后的预测结果 [B, 1, H, W] 或 None（当未提供dnr2/dnr3时）
        weight: 全分辨率融合权重图 [B, 1, H, W]，用于2DNR/3DNR加权融合
        packed_weight: 打包分辨率的原始权重 [B, 1, H/2, W/2](W1模式) 或 [B, 4, H/2, W/2](W4模式)
        current_feature: 从时序堆栈中选择的当前帧特征 [B, C, H, W]
        refined_feature: 经运动先验精炼后的当前帧特征 [B, C, H, W]
        stem_features: Stem编码器的中间输出（包含时序堆栈和运动特征）
        stack_output: ST-Mamba主干网络的输出（包含时序堆栈等）
    """

    prediction: Tensor | None
    weight: Tensor
    packed_weight: Tensor
    current_feature: Tensor
    refined_feature: Tensor
    stem_features: StemFeatures
    stack_output: STMambaLiteStackOutput


class CurrentFrameRefineHead(nn.Module):
    """使用运动先验特征对选定的当前帧特征进行精炼。
    
    网络结构：
    1. 拼接当前帧特征和运动先验特征 → 1x1卷积降维 → 激活函数
    2. 3x3深度可分离卷积提取空间信息 → 激活函数  
    3. 1x1卷积输出精炼后的特征（保持通道数不变）
    
    输入输出形状：
        - 输入: current_feature [B, C, H, W], motion_feature [B, C, H, W]
        - 输出: refined_feature [B, C, H, W]
    """

    def __init__(self, channels: int, activation: ActivationType = "silu") -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        # 将双通道特征拼接后降维回原通道数: [B, 2C, H, W] → [B, C, H, W]
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        # 深度可分离卷积: 逐通道空间滤波 [B, C, H, W] → [B, C, H, W]
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        # 最终投影层: 保持通道数不变 [B, C, H, W] → [B, C, H, W]
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.activation = build_activation(activation)

    def forward(self, current_feature: Tensor, motion_feature: Tensor) -> Tensor:
        """前向传播：融合当前帧特征与运动先验特征。
        
        Args:
            current_feature: 当前帧特征 [B, C, H, W]
            motion_feature: 运动先验特征 [B, C, H, W]
            
        Returns:
            refined_feature: 精炼后的特征 [B, C, H, W]
        """
        _validate_feature_pair(current_feature, motion_feature, self.channels)

        # 步骤1: 沿通道维度拼接两个特征 → [B, 2C, H, W]
        feature = torch.cat([current_feature, motion_feature], dim=1)
        # 步骤2: 1x1卷积降维 + 激活 → [B, C, H, W]
        feature = self.activation(self.reduce(feature))
        # 步骤3: 3x3深度卷积提取空间上下文 + 激活 → [B, C, H, W]
        feature = self.activation(self.depthwise(feature))
        # 步骤4: 1x1卷积输出最终精炼特征 → [B, C, H, W]
        return self.out_proj(feature)


class WeightHead(nn.Module):
    """在打包分辨率下预测融合权重（支持W1或W4格式）。
    
    W1模式: 输出单通道权重，所有Bayer位置共享同一权重值
    W4模式: 输出四通道权重，每个Bayer位置（R, Gr, Gb, B）有独立权重值
    
    网络结构：
    1. 1x1卷积将特征映射到权重通道数（1或4）
    2. Sigmoid激活函数将权重约束到[0, 1]区间
    
    输入输出形状：
        - 输入: feature_map [B, C, H, W]
        - 输出: packed_weight [B, 1, H, W](W1) 或 [B, 4, H, W](W4)
    """

    def __init__(
        self,
        channels: int,
        weight_mode: WeightMode = "w1",
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if weight_mode not in {"w1", "w4"}:
            raise ValueError(f"Unsupported weight_mode: {weight_mode}")

        self.channels = channels
        self.weight_mode = weight_mode
        # 根据权重模式确定输出通道数: W1→1通道, W4→4通道
        out_channels = 1 if weight_mode == "w1" else 4
        # 1x1卷积: [B, C, H, W] → [B, out_channels, H, W]
        self.proj = nn.Conv2d(channels, out_channels, kernel_size=1, bias=True)
        # 初始化: 权重设为0，偏置设为bias_init，使初始输出接近sigmoid(bias_init)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, bias_init)

    def forward(self, feature_map: Tensor) -> Tensor:
        """前向传播：从特征图预测融合权重。
        
        Args:
            feature_map: 输入特征图 [B, C, H, W]
            
        Returns:
            packed_weight: 打包分辨率的权重 [B, 1, H, W](W1) 或 [B, 4, H, W](W4)，值域[0, 1]
        """
        if feature_map.ndim != 4:
            raise ValueError(f"feature_map must have shape [B, C, H, W], got {feature_map.shape}")
        if feature_map.shape[1] != self.channels:
            raise ValueError(
                f"feature_map channel mismatch: expected {self.channels}, got {feature_map.shape[1]}"
            )
        # 1x1卷积投影 → Sigmoid激活: [B, C, H, W] → [B, out_channels, H, W]
        return torch.sigmoid(self.proj(feature_map))


class MambaFusionWeightNetLite(nn.Module):
    """完整的轻量级网络，用于预测2DNR/3DNR融合权重图。
    
    网络架构流程：
    1. Stem编码器: 将RAW数据转换为特征表示，构建时序堆栈和运动先验特征
    2. ST-Mamba主干: 处理时序堆栈特征，利用状态空间模型进行时序建模
    3. 当前帧选择: 从时序堆栈中提取当前帧（t=1）的特征
    4. 特征精炼: 使用运动先验特征对当前帧特征进行精炼
    5. 权重预测: 从精炼特征预测融合权重（W1或W4模式）
    6. 权重解包: 将打包分辨率的权重上采样到全分辨率
    7. 图像融合: 使用权重对2DNR和3DNR结果进行加权融合
    
    关键设计：
    - 显式的当前帧选择发生在ST-Mamba堆栈之后：z = stack_output.temporal_stack[:, :, 1]
    - 支持两种权重模式：W1（单通道共享）和W4（Bayer独立）
    """

    def __init__(self, config: MambaFusionWeightNetLiteConfig | None = None) -> None:
        """初始化网络各组件。
        
        Args:
            config: 网络配置对象，如果为None则使用默认配置
        """
        super().__init__()
        self.config = config or MambaFusionWeightNetLiteConfig()
        channels = self.config.backbone_config.channels  # 获取主干网络的通道数（与stem输出通道一致）

        # Stem编码器: 将packed RAW输入转换为特征表示 [B, 4, H, W] x 3 -> temporal_stack[B, C, T=2, H, W] + motion_feature[B, C, H, W]
        self.stem_encoder = PackedRawStemEncoder(self.config.stem_config)
        # ST-Mamba主干网络: 时序建模 [B, C, T=2, H, W] → [B, C, T=2, H, W]
        self.backbone = STMambaLiteStack(config=self.config.backbone_config)
        # 当前帧精炼头: 融合运动先验 [B, C, H, W] + [B, C, H, W] → [B, C, H, W]
        self.refine_head = CurrentFrameRefineHead(
            channels=channels,
            activation=self.config.refine_activation,
        )
        # 权重预测头: 特征到权重映射 [B, C, H, W] -> [B, 1, H, W] 或 [B, 4, H, W]
        self.weight_head = WeightHead(
            channels=channels,
            weight_mode=self.config.weight_mode,
            bias_init=self.config.weight_bias_init,
        )

    def forward(
        self,
        noisy_pair: Tensor,
        dnr2: Tensor | None = None,
        dnr3: Tensor | None = None,
    ) -> MambaFusionWeightNetLiteOutput:
        """标准前向传播接口，接受原始RAW数据对。
        
        Args:
            noisy_pair: 噪声图像对 [B, 2, H, W]，第0帧为前一帧，第1帧为当前帧
            dnr2: 2DNR去噪结果 [B, H, W] 或 [B, 1, H, W]，可选
            dnr3: 3DNR去噪结果 [B, H, W] 或 [B, 1, H, W]，可选
            
        Returns:
            MambaFusionWeightNetLiteOutput: 包含权重图、精炼特征和融合结果的输出对象
        """
        if not isinstance(noisy_pair, Tensor):
            raise TypeError(f"noisy_pair must be a torch.Tensor, got {type(noisy_pair)!r}")

        # 准备噪声对特征：解析RAW数据并构建运动先验
        # 输入: noisy_pair [B, 2, H, W]
        # 输出: prev_packed[B, 4, H/2, W/2], curr_packed[B, 4, H/2, W/2], motion_prior[B, 4, H/2, W/2]
        raw_features = prepare_noisy_pair_features(
            noisy_pair,
            raw_range=self.config.raw_range,
            cfa_pattern=self.config.cfa_pattern,
        )
        # 调用打包格式的前向传播接口
        return self.forward_packed(
            prev4=raw_features.prev_packed,
            curr4=raw_features.curr_packed,
            motion_prior=raw_features.motion_prior,
            dnr2=dnr2,
            dnr3=dnr3,
        )

    def forward_packed(
        self,
        prev4: Tensor,
        curr4: Tensor,
        motion_prior: Tensor,
        dnr2: Tensor | None = None,
        dnr3: Tensor | None = None,
    ) -> MambaFusionWeightNetLiteOutput:
        """打包格式的前向传播，直接接受已打包的RAW数据和运动先验。
        
        Args:
            prev4: 前一帧打包RAW数据 [B, 4, H/2, W/2]
            curr4: 当前帧打包RAW数据 [B, 4, H/2, W/2]
            motion_prior: 运动先验特征 [B, 4, H/2, W/2]
            dnr2: 2DNR去噪结果，可选
            dnr3: 3DNR去噪结果，可选
            
        Returns:
            MambaFusionWeightNetLiteOutput: 完整的网络输出对象
        """
        # 阶段1: Stem编码 - 将RAW数据转换为特征表示
        # 输入: prev4[B, 4, H/2, W/2], curr4[B, 4, H/2, W/2], motion_prior[B, 4, H/2, W/2]
        # 输出: temporal_stack[B, C, T=2, H, W], motion_feature[B, C, H, W]
        stem_features = self.stem_encoder(prev4, curr4, motion_prior)
        
        # 阶段2: ST-Mamba主干网络 - 时序特征建模
        # 输入: temporal_stack[B, C, T=2, H, W], motion_feature[B, C, H, W]
        # 输出: temporal_stack[B, C, T=2, H, W] (经过时序交互增强)
        stack_output = self.backbone.forward_stack(
            stem_features.temporal_stack,
            stem_features.motion_feature,
        )

        # 阶段3: 从时序堆栈中选择当前帧特征 (t=1)
        # 输入: temporal_stack[B, C, T=2, H, W]
        # 输出: current_feature[B, C, H, W]
        current_feature = self.select_current_frame_feature(stack_output)
        
        # 阶段4: 使用运动先验精炼当前帧特征
        # 输入: current_feature[B, C, H, W], motion_feature[B, C, H, W]
        # 输出: refined_feature[B, C, H, W]
        refined_feature = self.refine_head(current_feature, stem_features.motion_feature)
        
        # 阶段5: 预测打包分辨率的融合权重
        # 输入: refined_feature[B, C, H, W]
        # 输出: packed_weight[B, 1, H, W](W1) 或 [B, 4, H, W](W4)
        packed_weight = self.weight_head(refined_feature)
        
        # 阶段6: 将打包权重解包/广播到全分辨率
        # 输入: packed_weight[B, 1, H, W] 或 [B, 4, H, W]
        # 输出: weight[B, 1, 2H, 2W] (恢复到原始RAW分辨率)
        weight = self.unpack_or_broadcast_weight(packed_weight)
        
        # 阶段7: 使用预测的权重融合2DNR和3DNR结果
        # 输入: dnr2, dnr3, weight[B, 1, 2H, 2W]
        # 输出: prediction[B, 1, 2H, 2W] 或 None
        prediction = self.fuse_dnr(dnr2=dnr2, dnr3=dnr3, weight=weight)

        return MambaFusionWeightNetLiteOutput(
            prediction=prediction,
            weight=weight,
            packed_weight=packed_weight,
            current_feature=current_feature,
            refined_feature=refined_feature,
            stem_features=stem_features,
            stack_output=stack_output,
        )

    @staticmethod
    def select_current_frame_feature(stack_output: STMambaLiteStackOutput) -> Tensor:
        """从时序堆栈中选择当前帧（t=1）的特征。
        
        根据设计文档，时序堆栈的第0维是前一帧（prev），第1维是当前帧（curr）。
        
        Args:
            stack_output: ST-Mamba主干网络的输出对象，包含temporal_stack
            
        Returns:
            current_feature: 当前帧特征 [B, C, H, W]
        """
        if stack_output.temporal_stack.ndim != 5:
            raise ValueError(
                "stack_output.temporal_stack must have shape [B, C, T=2, H, W], "
                f"got {stack_output.temporal_stack.shape}"
            )
        if stack_output.temporal_stack.shape[2] != 2:
            raise ValueError(
                f"stack_output.temporal_stack must have T=2, got {stack_output.temporal_stack.shape[2]}"
            )
        # 索引t=1获取当前帧: [B, C, T=2, H, W] → [B, C, H, W]
        return stack_output.temporal_stack[:, :, 1]

    def unpack_or_broadcast_weight(self, packed_weight: Tensor) -> Tensor:
        """将打包分辨率的W1/W4权重转换为全分辨率 [B, 1, H, W]。
        
        W1模式: 使用repeat_interleave进行2×上采样（每个权重值复制到2×2区域）
        W4模式: 使用Bayer解包将4通道权重展开为单通道全分辨率权重图
        
        Args:
            packed_weight: 打包权重 [B, 1, H, W](W1) 或 [B, 4, H, W](W4)
            
        Returns:
            weight: 全分辨率权重 [B, 1, 2H, 2W]
        """
        if packed_weight.ndim != 4:
            raise ValueError(f"packed_weight must have shape [B, C, H, W], got {packed_weight.shape}")

        if self.config.weight_mode == "w1":
            # W1模式: 单通道权重，通过重复插值上采样2倍
            if packed_weight.shape[1] != 1:
                raise ValueError(f"W1 packed_weight must have one channel, got {packed_weight.shape[1]}")
            # 高度方向2×上采样: [B, 1, H, W] → [B, 1, 2H, W]
            # 宽度方向2×上采样: [B, 1, 2H, W] → [B, 1, 2H, 2W]
            return packed_weight.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

        # W4模式: 四通道Bayer权重，需要按照Bayer模式解包
        if packed_weight.shape[1] != 4:
            raise ValueError(f"W4 packed_weight must have four channels, got {packed_weight.shape[1]}")
        # Bayer解包: [B, 4, H, W] → [B, 2H, 2W]，然后增加通道维度 → [B, 1, 2H, 2W]
        return bayer_unpack(packed_weight, cfa_pattern=self.config.cfa_pattern).unsqueeze(1)

    @staticmethod
    def fuse_dnr(dnr2: Tensor | None, dnr3: Tensor | None, weight: Tensor) -> Tensor | None:
        """应用融合公式: pred = W × 3DNR + (1 - W) × 2DNR。
        
        当同时提供2DNR和3DNR结果时，使用预测的权重图进行自适应融合。
        权重越接近1，越倾向于3DNR结果；权重越接近0，越倾向于2DNR结果。
        
        Args:
            dnr2: 2DNR去噪结果 [B, H, W] 或 [B, 1, H, W]
            dnr3: 3DNR去噪结果 [B, H, W] 或 [B, 1, H, W]
            weight: 融合权重 [B, 1, H, W]，值域[0, 1]
            
        Returns:
            prediction: 融合后的结果 [B, H, W] 或 [B, 1, H, W]，与输入dnr形状一致；若dnr均为None则返回None
        """
        if dnr2 is None and dnr3 is None:
            return None
        if dnr2 is None or dnr3 is None:
            raise ValueError("dnr2 and dnr3 must be provided together")
        if dnr2.shape != dnr3.shape:
            raise ValueError(f"dnr2 and dnr3 must share the same shape, got {dnr2.shape} vs {dnr3.shape}")
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError(f"weight must have shape [B, 1, H, W], got {weight.shape}")

        # 情况1: dnr为3D张量 [B, H, W]
        if dnr2.ndim == 3:
            if dnr2.shape[0] != weight.shape[0] or dnr2.shape[-2:] != weight.shape[-2:]:
                raise ValueError(f"dnr tensors must match weight batch/spatial shape, got {dnr2.shape} vs {weight.shape}")
            # 去掉weight的通道维度以匹配3D张量: [B, 1, H, W] → [B, H, W]
            weight_2d = weight[:, 0]
            # 逐元素加权融合: [B, H, W] × [B, H, W] + [B, H, W] × [B, H, W] → [B, H, W]
            return weight_2d * dnr3 + (1.0 - weight_2d) * dnr2

        # 情况2: dnr为4D张量 [B, 1, H, W]
        if dnr2.ndim == 4:
            if dnr2.shape[0] != weight.shape[0] or dnr2.shape[-2:] != weight.shape[-2:]:
                raise ValueError(f"dnr tensors must match weight batch/spatial shape, got {dnr2.shape} vs {weight.shape}")
            if dnr2.shape[1] != 1:
                raise ValueError(f"dnr tensors must be single-channel RAW maps, got {dnr2.shape}")
            # 直接逐元素加权融合: [B, 1, H, W] × [B, 1, H, W] + ... → [B, 1, H, W]
            return weight * dnr3 + (1.0 - weight) * dnr2

        raise ValueError(f"dnr tensors must have shape [B, H, W] or [B, 1, H, W], got {dnr2.shape}")

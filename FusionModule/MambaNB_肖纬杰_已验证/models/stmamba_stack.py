from __future__ import annotations

"""Stack wrapper for chaining ST-Mamba-Lite blocks after packed RAW stems."""

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch import Tensor

from .stmamba_block import STMambaLiteBlock, _validate_feature_triplet
from .stmamba_config import STMambaLiteConfig


@dataclass(frozen=True)
class STMambaLiteStackOutput:
    """Structured output of a stacked ST-Mamba-Lite backbone stage.
    
    Attributes:
        prev_feature: 前一帧特征图, shape [B, C, H, W]
        curr_feature: 当前帧特征图, shape [B, C, H, W]
        temporal_stack: 时间堆叠特征, shape [B, C, T=2, H, W]
    """

    prev_feature: Tensor
    curr_feature: Tensor
    temporal_stack: Tensor


class STMambaLiteStack(nn.Module):
    """Chain multiple `STMambaLiteBlock` instances at packed RAW resolution.
    
    该模块将多个ST-Mamba-Lite块串联起来,在打包的RAW分辨率下处理时序特征。
    输入为前后两帧的特征图和运动特征,输出为增强后的时序特征。
    """

    def __init__(
        self,
        config: STMambaLiteConfig | None = None,
        channels: int | None = None,
        num_blocks: int | None = None,
    ) -> None:
        """初始化ST-Mamba-Lite堆叠模块。
        
        Args:
            config: 配置对象,包含通道数、块数量等参数
            channels: 特征通道数,默认为24。若提供则覆盖config中的设置
            num_blocks: ST-Mamba块的数量,默认为2。若提供则覆盖config中的设置
        """
        super().__init__()
        if config is None:
            # 使用默认配置: 24通道, 2个块
            config = STMambaLiteConfig(
                channels=channels or 24,
                num_blocks=num_blocks or 2,
            )
        else:
            # 如果提供了config但有额外参数,则更新config
            overrides: dict[str, int] = {}
            if channels is not None:
                overrides["channels"] = channels
            if num_blocks is not None:
                overrides["num_blocks"] = num_blocks
            if overrides:
                config = replace(config, **overrides)

        self.config = config
        # 创建指定数量的ST-Mamba-Lite块,串行连接
        self.blocks = nn.ModuleList(
            [STMambaLiteBlock(config=config) for _ in range(config.num_blocks)]
        )

    def forward(
        self,
        prev_feature: Tensor,
        curr_feature: Tensor,
        motion_feature: Tensor,
    ) -> STMambaLiteStackOutput:
        """前向传播,通过所有ST-Mamba-Lite块处理时序特征。
        
        Args:
            prev_feature: 前一帧特征图, shape [B, C, H, W]
                         B=batch size, C=通道数, H=高度, W=宽度
            curr_feature: 当前帧特征图, shape [B, C, H, W]
            motion_feature: 运动特征图, shape [B, C, H, W]
            
        Returns:
            STMambaLiteStackOutput: 包含增强后的时序特征
                - prev_feature: 增强后的前一帧特征, shape [B, C, H, W]
                - curr_feature: 增强后的当前帧特征, shape [B, C, H, W]
                - temporal_stack: 时间堆叠张量, shape [B, C, T=2, H, W]
        """
        # 验证输入特征的维度一致性(通道数必须匹配配置)
        _validate_feature_triplet(prev_feature, curr_feature, motion_feature, self.config.channels)

        # 初始化运行特征变量,用于在块之间传递
        running_prev_feature = prev_feature  # shape: [B, C, H, W]
        running_curr_feature = curr_feature  # shape: [B, C, H, W]

        # 依次通过每个ST-Mamba-Lite块进行处理
        for block in self.blocks:
            running_prev_feature, running_curr_feature = block(
                running_prev_feature,
                running_curr_feature,
                motion_feature,
            )

        # 将前后两帧特征沿时间维度堆叠,形成时序张量
        # 输入: running_prev_feature [B, C, H, W], running_curr_feature [B, C, H, W]
        # 输出: temporal_stack [B, C, T=2, H, W]
        temporal_stack = torch.stack([running_prev_feature, running_curr_feature], dim=2)
        
        return STMambaLiteStackOutput(
            prev_feature=running_prev_feature,      # shape: [B, C, H, W]
            curr_feature=running_curr_feature,      # shape: [B, C, H, W]
            temporal_stack=temporal_stack,          # shape: [B, C, T=2, H, W]
        )

    def forward_stack(self, temporal_stack: Tensor, motion_feature: Tensor) -> STMambaLiteStackOutput:
        """使用前向传播的便捷方法,直接接受时间堆叠张量作为输入。
        
        Args:
            temporal_stack: 时间堆叠特征张量, shape [B, C, T=2, H, W]
                           其中T=2表示包含前后两帧
            motion_feature: 运动特征图, shape [B, C, H, W]
            
        Returns:
            STMambaLiteStackOutput: 与forward()相同的输出结构
            
        Raises:
            ValueError: 当输入张量维度不为5或时间维度不为2时抛出异常
        """
        # 验证输入必须是5维张量 [B, C, T, H, W]
        if temporal_stack.ndim != 5:
            raise ValueError(
                f"temporal_stack must have shape [B, C, T=2, H, W], got {temporal_stack.shape}"
            )
        # 验证时间维度必须为2(前后两帧)
        if temporal_stack.shape[2] != 2:
            raise ValueError(f"temporal_stack must have T=2, got T={temporal_stack.shape[2]}")

        # 从时间堆叠张量中分离出前后两帧特征
        # temporal_stack shape: [B, C, T=2, H, W]
        # prev_feature shape: [B, C, H, W] (取第0帧)
        # curr_feature shape: [B, C, H, W] (取第1帧)
        prev_feature = temporal_stack[:, :, 0]
        curr_feature = temporal_stack[:, :, 1]
        
        # 调用标准forward方法进行处理
        return self.forward(prev_feature, curr_feature, motion_feature)

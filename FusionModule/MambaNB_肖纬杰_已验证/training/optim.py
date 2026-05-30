from __future__ import annotations

"""Model and optimizer construction for AI fusion training.

本模块负责构建AI Fusion训练所需的模型和优化器组件：
1. 根据命令行参数构建MambaFusionWeightNetLite模型
2. 构建带权重衰减分组的优化器参数组（区分需要/不需要正则化的参数）
3. 构建带warmup的余弦退火学习率调度器
"""

import argparse
import math
from typing import Any

from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from MyNet.ai_fusion.models import (
    MambaFusionWeightNetLite,
    MambaFusionWeightNetLiteConfig,
    STMambaLiteConfig,
    StemConfig,
)


def build_model(args: argparse.Namespace) -> MambaFusionWeightNetLite:
    """根据命令行参数构建AI Fusion模型。
    
    按照配置层次依次构建Stem编码器、ST-Mamba主干网络和完整的融合网络。
    
    Args:
        args: 命令行参数命名空间，包含以下关键字段:
            - channels: 特征通道数（默认24），控制网络宽度
            - num_blocks: ST-Mamba块数量，控制网络深度
            - mamba_state_dim: Mamba状态空间维度（默认8）
            - mamba_expand: Mamba扩展倍数（默认2）
            - mamba_scan_backend: Mamba扫描后端（'auto'/'reference'/'mamba_ssm'）
            - weight_mode: 权重模式（'w1'单通道共享 / 'w4'Bayer独立）
            - cfa_pattern: CFA拜耳模式（如'GBRG'）
            - weight_bias_init: 权重预测头的偏置初始化值
            
    Returns:
        MambaFusionWeightNetLite: 构建好的完整模型实例
        
    Note:
        模型整体架构和数据流:
        1. Stem编码: prev4[B,4,H/2,W/2] + curr4[B,4,H/2,W/2] + motion_prior[B,4,H/2,W/2]
                   → temporal_stack[B,C,T=2,H,W] + motion_feature[B,C,H,W]
        2. ST-Mamba主干: temporal_stack[B,C,T=2,H,W] → temporal_stack[B,C,T=2,H,W] (时序增强)
        3. 当前帧选择: temporal_stack[B,C,T=2,H,W] → current_feature[B,C,H,W] (取t=1)
        4. 特征精炼: current_feature[B,C,H,W] + motion_feature[B,C,H,W] → refined[B,C,H,W]
        5. 权重预测: refined[B,C,H,W] → packed_weight[B,1,H,W](W1) 或 [B,4,H,W](W4)
        6. 权重解包: packed_weight → weight[B,1,2H,2W] (恢复到原始RAW分辨率)
        7. 图像融合: weight × dnr3 + (1-weight) × dnr2 → prediction[B,1,2H,2W]
        
        其中C=args.channels，典型值为24或48
    """
    # 第1步：构建Stem编码器配置
    # Stem负责将packed RAW输入 [B, 4, H/2, W/2] 转换为特征表示 [B, C, H, W]
    stem_config = StemConfig(stem_channels=args.channels)
    
    # 第2步：构建ST-Mamba主干网络配置
    # ST-Mamba对时序堆栈 [B, C, T=2, H, W] 进行时序建模和空间特征提取
    backbone_config = STMambaLiteConfig(
        channels=args.channels,              # 特征通道数C
        num_blocks=args.num_blocks,          # Mamba块数量（网络深度）
        mamba_state_dim=args.mamba_state_dim,   # 状态空间维度D（默认8）
        mamba_expand=args.mamba_expand,         # 隐藏层扩展倍数E（默认2，内部维度=C*E）
        mamba_scan_backend=args.mamba_scan_backend,  # 扫描算法后端选择
    )
    
    # 第3步：构建完整的MambaFusionWeightNetLite配置
    model_config = MambaFusionWeightNetLiteConfig(
        stem_config=stem_config,             # Stem编码器配置
        backbone_config=backbone_config,     # ST-Mamba主干配置
        weight_mode=args.weight_mode,        # 权重模式：'w1'或'w4'
        cfa_pattern=args.cfa_pattern,        # CFA拜耳模式（如'GBRG'）
        weight_bias_init=args.weight_bias_init,  # 权重头偏置初始化（控制初始融合倾向）
    )
    
    # 第4步：实例化模型
    return MambaFusionWeightNetLite(model_config)


def build_parameter_groups(model: nn.Module, weight_decay: float, lr: float) -> list[dict[str, Any]]:
    """构建带权重衰减分组的优化器参数组。
    
    将模型参数分为两组：
    1. 需要权重衰减的参数：权重矩阵、卷积核等高维参数
    2. 不需要权重衰减的参数：偏置项、LayerNorm/BatchNorm参数、Mamba特定参数
    
    这种分组策略可以避免对偏置和归一化层过度正则化，提升训练稳定性。
    
    Args:
        model: PyTorch模型实例，将遍历其所有可训练参数
        weight_decay: 权重衰减系数（L2正则化强度），典型值0.01~0.1
        lr: 基础学习率，应用于所有参数组
        
    Returns:
        list[dict]: 两个参数组字典列表，可直接传给优化器构造函数
        [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr}
        ]
        
    Note:
        不需要权重衰减的参数判断规则（满足任一条件即加入no_decay组）:
        1. param.ndim < 2: 维度小于2的参数（如1D偏置向量）
        2. 参数名包含以下关键词:
           - 'bias': 偏置项
           - 'norm': LayerNorm/BatchNorm等归一化层的缩放和平移参数
           - 'A_log': Mamba状态矩阵A的对数参数
           - 'D': Mamba skip connection参数
           - 'direction_logits': 方向 logits 参数（如果存在）
        
        典型使用示例:
        ```python
        param_groups = build_parameter_groups(model, weight_decay=0.05, lr=1e-3)
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        ```
    """
    decay: list[nn.Parameter] = []      # 需要权重衰减的参数列表
    no_decay: list[nn.Parameter] = []   # 不需要权重衰减的参数列表
    
    # 定义不需要权重衰减的参数名称关键词
    no_decay_tokens = ("bias", "norm", "A_log", "D", "direction_logits")
    
    # 遍历模型所有命名参数
    for name, param in model.named_parameters():
        # 跳过冻结的参数（requires_grad=False）
        if not param.requires_grad:
            continue
            
        # 判断是否属于不需要权重衰减的参数
        # 条件1: 参数维度<2（如偏置向量[param_dim]）
        # 条件2: 参数名包含特定关键词（bias/norm/A_log/D等）
        if param.ndim < 2 or any(token in name for token in no_decay_tokens):
            no_decay.append(param)  # 加入无衰减组
        else:
            decay.append(param)     # 加入有衰减组（如卷积核[out,in,kh,kw]、全连接权重等）
            
    # 返回两个参数组，供优化器使用
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},      # 高维参数应用L2正则
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},            # 偏置/BN等不应用L2正则
    ]


def build_scheduler(
    optimizer: Any,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """构建带线性warmup的余弦退火学习率调度器。
    
    学习率变化曲线分为两个阶段：
    1. Warmup阶段（0 ~ warmup_steps）: 学习率从min_ratio线性增长到1.0
    2. 余弦退火阶段（warmup_steps ~ total_steps）: 学习率从1.0按余弦曲线下降到min_ratio
    
    Args:
        optimizer: PyTorch优化器实例，调度器将调整其学习率
        total_steps: 总训练步数（决定余弦退火的周期长度）
        warmup_steps: warmup步数（在此步数内学习率线性增长）
        min_lr: 最小学习率（余弦退火的终点值）
        base_lr: 基础学习率（warmup结束时的峰值学习率）
        
    Returns:
        LambdaLR: PyTorch学习率调度器，在每个step调用scheduler.step()时自动更新学习率
        
    Note:
        学习率计算公式:
        
        Warmup阶段 (step < warmup_steps):
            lr_factor = max(min_ratio, (step + 1) / warmup_steps)
            实际学习率 = base_lr × lr_factor
            
        余弦退火阶段 (step >= warmup_steps):
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine = 0.5 × (1 + cos(π × progress))
            lr_factor = min_ratio + (1 - min_ratio) × cosine
            实际学习率 = base_lr × lr_factor
        
        关键特性:
        - min_ratio = min_lr / base_lr，通常在0.01~0.1之间
        - warmup避免训练初期梯度爆炸，让模型平稳启动
        - 余弦退火在训练后期逐渐降低学习率，帮助模型收敛到更优解
        - step+1确保第一步的学习率不为0（从1/warmup_steps开始）
        
        典型使用示例:
        ```python
        total_steps = len(train_loader) * num_epochs
        warmup_steps = total_steps // 10  # 10%的步数用于warmup
        scheduler = build_scheduler(optimizer, total_steps, warmup_steps, 
                                    min_lr=1e-6, base_lr=1e-3)
        
        for epoch in range(num_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()  # 每步更新学习率
        ```
        
        学习率曲线示意:
        ```
        lr
        |
        |        /\
        |       /  \
        |      /    \_____
        |     /           \
        |____/             \___
        |________________________ step
             warmup   cosine decay
        ```
    """
    # 计算最小学习率相对于基础学习率的比值（通常在0.01~0.1）
    min_ratio = min_lr / base_lr
    
    # 确保total_steps至少为1，避免除零错误
    total_steps = max(1, total_steps)
    
    # 限制warmup_steps在合理范围内 [0, total_steps]
    warmup_steps = min(max(0, warmup_steps), total_steps)

    def lr_lambda(step: int) -> float:
        """学习率缩放因子计算函数，由LambdaLR在每个step调用。
        
        Args:
            step: 当前全局训练步数（从0开始）
            
        Returns:
            float: 学习率缩放因子，实际学习率 = base_lr × 返回值
        """
        # 阶段1: Warmup - 线性增长学习率
        if warmup_steps > 0 and step < warmup_steps:
            # 计算线性增长因子: (step+1) / warmup_steps
            # step=0时为1/warmup_steps，step=warmup_steps-1时接近1.0
            # 使用max确保不低于min_ratio（防止warmup_steps很大时初始lr过小）
            return max(min_ratio, float(step + 1) / warmup_steps)
        
        # 阶段2: 余弦退火 - 平滑下降学习率
        # 计算当前在余弦阶段的进度 [0, 1]
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        
        # 计算余弦因子: 从1.0平滑下降到0.0
        # progress=0时cos(0)=1 → cosine=1.0
        # progress=1时cos(π)=-1 → cosine=0.0
        # progress>1时限制为1.0，cosine保持0.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        
        # 插值计算最终缩放因子: 从1.0下降到min_ratio
        # progress=0时: min_ratio + (1-min_ratio) × 1.0 = 1.0
        # progress=1时: min_ratio + (1-min_ratio) × 0.0 = min_ratio
        return min_ratio + (1.0 - min_ratio) * cosine

    # 创建LambdaLR调度器，传入自定义的学习率计算函数
    return LambdaLR(optimizer, lr_lambda)

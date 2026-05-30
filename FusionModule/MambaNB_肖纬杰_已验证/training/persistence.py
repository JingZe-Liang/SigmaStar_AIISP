from __future__ import annotations

"""Checkpoint and lightweight JSON logging helpers.

本模块负责训练过程中的持久化操作：
1. 模型检查点的保存与加载（包含模型权重、优化器状态、调度器状态等）
2. 轻量级JSON日志记录（训练指标、验证指标等）
3. 训练配置的序列化存储
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .data import SplitStats


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_score: float,
    args: argparse.Namespace,
    train_stats: SplitStats,
    val_stats: SplitStats,
) -> None:
    """保存完整的训练检查点到指定路径。
    
    检查点包含恢复训练所需的全部状态信息，支持断点续训。
    
    Args:
        path: 检查点保存路径（.pth或.pt文件）
        model: PyTorch模型实例，其state_dict将被保存
        optimizer: 优化器实例，保存其内部状态（动量、学习率等）
        scheduler: 学习率调度器实例，保存当前调度状态
        scaler: 梯度缩放器（用于混合精度训练），保存缩放因子历史
        epoch: 当前训练轮次编号
        global_step: 全局训练步数（跨epoch累计）
        best_score: 历史最佳验证分数（用于判断是否保存最优模型）
        args: 命令行参数命名空间，记录训练配置
        train_stats: 训练集统计信息（场景数、分片数等）
        val_stats: 验证集统计信息
        
    Note:
        保存的模型state_dict结构取决于模型定义，对于AI Fusion模型通常包含:
        - backbone层权重: 卷积核形状如 [out_channels, in_channels, kernel_h, kernel_w]
        - BN层统计量: running_mean/var 形状如 [num_features]
        - 其他可学习参数: 根据网络架构而定
        
        优化器state_dict包含:
        - param_groups: 参数组配置（学习率、权重衰减等）
        - state: 每个参数的优化器状态（如Adam的momentum缓冲区）
    """
    # 确保父目录存在，避免保存时因目录不存在而失败
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建检查点字典，包含所有需要持久化的状态信息
    torch.save(
        {
            "model": model.state_dict(),          # 模型参数字典 {param_name: Tensor}
            "optimizer": optimizer.state_dict(),  # 优化器状态（动量、步数等）
            "scheduler": scheduler.state_dict(),  # 调度器状态（当前epoch、学习率等）
            "scaler": scaler.state_dict(),        # 梯度缩放器状态（用于AMP混合精度训练）
            "epoch": epoch,                       # 当前训练轮次
            "global_step": global_step,           # 全局训练步数
            "best_score": best_score,             # 历史最佳验证分数
            "args": vars_for_json(args),          # 训练配置参数（Path对象转为字符串）
            "train_stats": asdict(train_stats),   # 训练集统计信息字典
            "val_stats": asdict(val_stats),       # 验证集统计信息字典
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    """从检查点文件加载训练状态并恢复到指定设备。
    
    用于断点续训或推理时加载预训练权重。
    
    Args:
        path: 检查点文件路径
        model: 目标模型实例，将加载保存的权重参数
        optimizer: 目标优化器实例，将恢复优化器状态
        scheduler: 目标调度器实例，将恢复调度状态
        scaler: 目标梯度缩放器，将恢复缩放历史（如果检查点中包含）
        device: 加载到的目标设备（cpu/cuda:0等）
        
    Returns:
        tuple包含:
        - epoch: 恢复的训练轮次编号，从该轮次继续训练
        - global_step: 恢复的全局步数，用于日志记录和调度器计算
        - best_score: 历史最佳验证分数，用于后续模型选择判断
        
    Note:
        - 使用map_location确保张量正确加载到目标设备，避免CUDA内存问题
        - scaler状态是可选的，旧检查点可能不包含此字段
        - 加载后模型参数形状必须与检查点中保存的形状完全一致，否则会报错
        
        典型使用场景:
        ```python
        start_epoch, start_step, best = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device
        )
        for epoch in range(start_epoch, total_epochs):
            # 从断点处继续训练
        ```
    """
    # 加载检查点字典到指定设备（避免全部加载到GPU再转移的内存浪费）
    checkpoint = torch.load(path, map_location=device)
    
    # 恢复模型权重参数（逐层匹配param_name -> Tensor）
    model.load_state_dict(checkpoint["model"])
    
    # 恢复优化器内部状态（动量缓冲区、步数计数器等）
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # 恢复学习率调度器状态（当前epoch、累积步数等）
    scheduler.load_state_dict(checkpoint["scheduler"])
    
    # 恢复梯度缩放器状态（如果检查点中包含），用于混合精度训练连续性
    scaler_state = checkpoint.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
        
    # 返回训练进度信息，供训练循环使用以决定从哪里继续
    return (
        int(checkpoint.get("epoch", 0)),                    # 默认从第0轮开始（如果检查点无此字段）
        int(checkpoint.get("global_step", 0)),              # 默认从第0步开始
        float(checkpoint.get("best_score", -float("inf"))), # 默认最佳分数为负无穷（任何分数都会更新）
    )


def load_model_weights(path: Path, model: nn.Module, device: torch.device) -> None:
    """Load only model weights from a full training checkpoint or raw state_dict."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def format_log(prefix: str, record: dict[str, Any]) -> str:
    """格式化日志记录为可读字符串。
    
    将字典形式的训练/验证指标转换为统一的日志格式，便于控制台输出和日志文件记录。
    
    Args:
        prefix: 日志前缀（如"Epoch 5 Train"、"Epoch 5 Val"等）
        record: 指标字典，键为指标名称，值为数值（如{'loss': 0.123, 'psnr': 35.6}）
        
    Returns:
        str: 格式化后的日志字符串，格式为 "prefix | key1=value1 | key2=value2 | ..."
        
    Example:
        >>> format_log("Epoch 1 Train", {"loss": 0.1234, "psnr": 35.6789})
        'Epoch 1 Train | loss=0.1234 | psnr=35.6789'
        
    Note:
        - 浮点数保留最多6位有效数字（:.6g格式），避免过长小数影响可读性
        - 非浮点数直接转换为字符串表示
        - 各字段用 " | " 分隔，便于日志解析工具提取
    """
    parts = [f"{prefix}"]  # 初始化日志部分列表，以prefix开头
    
    # 遍历指标字典，逐个格式化键值对
    for key, value in record.items():
        if isinstance(value, float):
            # 浮点数使用科学计数法或定点表示（最多6位有效数字）
            parts.append(f"{key}={value:.6g}")
        else:
            # 整数、字符串等其他类型直接转换
            parts.append(f"{key}={value}")
            
    # 用 " | " 连接所有部分形成最终日志字符串
    return " | ".join(parts)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """追加单条记录到JSONL文件（每行一个JSON对象）。
    
    JSONL格式适合流式写入训练日志，便于后续分析和可视化。
    
    Args:
        path: JSONL文件路径（通常为logs/train.jsonl或logs/val.jsonl）
        record: 要记录的字典数据（如{"epoch": 5, "loss": 0.123, "step": 1000}）
        
    Note:
        - 每条记录独占一行，便于按行读取和处理
        - 使用ensure_ascii=False保证中文等非ASCII字符正常显示
        - 自动创建父目录，避免因目录不存在而写入失败
        
    Example:
        文件内容示例:
        {"epoch": 1, "loss": 0.456, "psnr": 28.5}
        {"epoch": 2, "loss": 0.234, "psnr": 32.1}
        ...
    """
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 以追加模式打开文件，写入单行JSON记录（末尾加换行符）
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写入完整的JSON文件（带缩进格式化）。
    
    用于保存结构化配置或最终结果，如训练总结、最佳模型指标等。
    
    Args:
        path: JSON文件保存路径（如results/summary.json）
        data: 要保存的字典数据（可嵌套，会自动序列化为JSON）
        
    Note:
        - 使用indent=2进行美化输出，便于人工阅读和版本控制对比
        - 覆盖模式写入（"w"），每次调用会替换文件全部内容
        - 适用于保存不频繁更新的结构性数据，而非流式日志
        
    Example:
        生成的JSON文件格式:
        {
          "best_epoch": 50,
          "best_psnr": 38.5,
          "config": {...}
        }
    """
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 以写入模式打开文件，保存格式化的JSON数据（2空格缩进）
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def vars_for_json(args: argparse.Namespace) -> dict[str, Any]:
    """将argparse命名空间转换为可JSON序列化的字典。
    
    解决Path对象等不可JSON序列化类型的转换问题。
    
    Args:
        args: argparse解析后的命令行参数命名空间
        
    Returns:
        dict: 可JSON序列化的参数字典，其中Path对象已转为字符串
        
    Note:
        - vars(args)将Namespace转换为普通字典 {param_name: value}
        - Path对象无法直接被json.dumps序列化，需转为字符串路径
        - 其他类型（int、float、str、bool等）保持原样不变
        
    Example:
        输入: Namespace(data_dir=Path('/path/to/data'), batch_size=32)
        输出: {'data_dir': '/path/to/data', 'batch_size': 32}
    """
    # 遍历参数字典，将Path对象转换为字符串，其他类型保持不变
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

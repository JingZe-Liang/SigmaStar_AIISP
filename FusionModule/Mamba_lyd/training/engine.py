from __future__ import annotations

"""Training, validation, and dry-run loops.

本模块实现RAW图像融合训练的三大核心循环：
1. train_one_epoch: 单轮训练循环，包含梯度累积和混合精度训练
2. validate: 验证循环，计算PSNR、SSIM等指标
3. run_dry_pass: 干运行测试，验证模型前向/后向传播是否正常

主要特性:
- 支持梯度累积（grad_accum_steps）以模拟更大的batch size
- 使用AMP（自动混合精度）加速训练并减少显存占用
- 梯度裁剪防止梯度爆炸
- 学习率调度器动态调整学习率
- 详细的训练日志输出（损失、学习率、权重统计等）
"""

import time
import json
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .losses import LossConfig, compute_loss
from .metrics import AverageMeter, MetricAccumulator
from .persistence import format_log
from .tensor_utils import autocast_context, move_batch_to_device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    loss_config: LossConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    epoch: int,
    global_step: int,
    total_steps: int,
    grad_accum_steps: int,
    grad_clip_norm: float,
    log_every: int,
) -> tuple[dict[str, float], int]:
    """执行一轮完整的训练循环。
    
    该函数遍历数据加载器中的所有batch，执行前向传播、损失计算、反向传播和参数更新。
    支持梯度累积和混合精度训练，以提高训练效率和稳定性。
    
    Args:
        model: 待训练的神经网络模型
        loader: 数据加载器，提供训练batch
        optimizer: 优化器（如AdamW）
        scheduler: 学习率调度器
        scaler: 梯度缩放器，用于混合精度训练
        loss_config: 损失函数配置对象
        device: 计算设备（CPU或GPU）
        amp_dtype: 混合精度数据类型（torch.float16或torch.bfloat16），None表示禁用AMP
        epoch: 当前训练轮次（从0开始）
        global_step: 全局步数计数器
        total_steps: 总训练步数上限
        grad_accum_steps: 梯度累积步数（每N个batch更新一次参数）
        grad_clip_norm: 梯度裁剪的L2范数阈值
        log_every: 日志输出频率（每N步输出一次）
        
    Returns:
        tuple: (平均损失字典, 更新后的global_step)
            - 平均损失字典包含: loss, loss_rec, loss_tv, loss_plane, loss_oracle, 
              loss_diversity, grad_norm的平均值
            - global_step: 更新后的全局步数
            
    Note:
        训练流程:
        1. 设置模型为训练模式 model.train()
        2. 初始化各损失项的平均值累加器
        3. 对每个batch:
           a. 将数据移至指定设备
           b. 在AMP上下文中执行前向传播
           c. 计算损失并按梯度累积步数缩放
           d. 反向传播累积梯度
           e. 达到累积步数后：梯度裁剪 → 参数更新 → 学习率调整 → 清零梯度
           f. 记录损失统计和权重分布
        4. 返回平均损失和更新后的global_step
        
        张量形状说明（典型情况）:
        - batch['prev4']: [B, C, H, W] 前一帧的4通道Bayer图像
        - batch['curr4']: [B, C, H, W] 当前帧的4通道Bayer图像
        - batch['motion_prior']: [B, 1, H//2, W//2] 运动先验特征（下采样）
        - batch['dnr2']: [B, C, H, W] 2DNR去噪结果
        - batch['dnr3']: [B, C, H, W] 3DNR去噪结果
        - output.prediction: [B, C, H, W] 融合预测结果
        - output.weight: [B, 1, H, W] 全分辨率融合权重
        - output.packed_weight: [B, K, H, W] 打包分辨率权重（K=1或4）
    """
    # 步骤1: 设置模型为训练模式（启用dropout、batchnorm等训练特定行为）
    model.train()
    
    # 步骤2: 初始化各损失项的平均值累加器
    # AverageMeter内部维护total和count，通过avg属性计算平均值
    meters = {
        key: AverageMeter()
        for key in (
            "loss",         # 总损失
            "loss_rec",     # 重建损失（Charbonnier损失）
            "loss_tv",      # 总变分损失（边缘感知的平滑约束）
            "loss_plane",   # 平面一致性损失（鼓励4帧权重均匀）
            "loss_oracle",  # Oracle权重损失（监督学习理想权重）
            "loss_diversity",  # 多样性损失（避免权重极端化）
            "grad_norm",    # 梯度L2范数（监控梯度爆炸）
        )
    }
    
    # 步骤3: 清零所有参数的梯度，set_to_none=True可节省内存
    optimizer.zero_grad(set_to_none=True)
    
    # 步骤4: 记录当前step的起始时间，用于计算steps_per_sec
    step_started = time.perf_counter()

    # 步骤5: 遍历数据加载器中的所有batch
    for batch_idx, batch in enumerate(loader):
        # 步骤5.1: 将batch中的所有张量移动到指定设备（GPU/CPU）
        # non_blocking=True允许异步传输，提升性能
        # batch是字典，键包括: 'prev4', 'curr4', 'motion_prior', 'dnr2', 'dnr3', 'clean'等
        batch = move_batch_to_device(batch, device)
        
        # 步骤5.2: 在混合精度上下文中执行前向传播
        # 如果amp_dtype不为None，则使用指定的半精度类型（fp16/bf16）
        # 否则使用默认的fp32精度
        with autocast_context(device, amp_dtype):
            # 模型前向传播：输入多帧信息，输出融合结果和权重
            # 输入张量形状:
            #   prev4: [B, C, H, W] 前一帧（4通道Bayer格式）
            #   curr4: [B, C, H, W] 当前帧（4通道Bayer格式）
            #   motion_prior: [B, 1, H//2, W//2] 运动先验（下采样2倍）
            #   dnr2: [B, C, H, W] 2DNR去噪结果
            #   dnr3: [B, C, H, W] 3DNR去噪结果
            # 输出对象属性:
            #   prediction: [B, C, H, W] 融合后的预测图像
            #   weight: [B, 1, H, W] 全分辨率融合权重（单通道）
            #   packed_weight: [B, K, H, W] 打包权重（K=1为W1模式，K=4为W4模式）
            output = model.forward_packed(
                prev4=batch["prev4"],
                curr4=batch["curr4"],
                motion_prior=batch["motion_prior"],
                dnr2=batch["dnr2"],
                dnr3=batch["dnr3"],
            )
            
            # 计算总损失和各分项损失
            # 输入:
            #   batch: 包含'clean', 'motion_prior', 'curr4', 'dnr2', 'dnr3'等键
            #   output: 模型输出对象，包含'prediction', 'weight', 'packed_weight'
            #   loss_config: 损失配置，包含各损失项的权重系数
            # 返回:
            #   loss: 标量张量 [] 总损失（加权和）
            #   loss_parts: 字典，包含各分项损失的浮点数值
            #     {'loss': float, 'loss_rec': float, 'loss_tv': float, 
            #      'loss_plane': float, 'loss_oracle': float, 'loss_diversity': float}
            loss, loss_parts = compute_loss(batch, output, loss_config)
            
            # 按梯度累积步数缩放损失
            # 这样累积grad_accum_steps个batch的梯度后，等效于使用大batch训练
            # scaled_loss: 标量张量 []
            scaled_loss = loss / grad_accum_steps

        # 步骤5.3: 反向传播（在AMP上下文中，scaler会自动处理梯度缩放）
        # scaler.scale()先将loss乘以缩放因子，防止fp16下溢
        # .backward()计算梯度并累积到参数的.grad属性中
        scaler.scale(scaled_loss).backward()
        
        # 步骤5.4: 判断是否应该执行参数更新
        # 条件1: 当前batch索引+1是grad_accum_steps的倍数（常规更新点）
        # 条件2: 当前batch是最后一个batch（确保最后一批梯度也被应用）
        should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader)

        # 步骤5.5: 如果达到更新条件，执行参数更新流程
        if should_step:
            # 步骤5.5.1: 反缩放梯度（unscale）
            # 将梯度除以缩放因子，恢复到原始尺度，以便进行梯度裁剪
            scaler.unscale_(optimizer)
            
            # 步骤5.5.2: 梯度裁剪
            # 计算所有参数梯度的L2范数，如果超过grad_clip_norm则按比例缩放
            # 这防止梯度爆炸，稳定训练过程
            # grad_norm: float 标量，表示裁剪前的梯度范数
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm).item())
            
            # 步骤5.5.3: 执行参数更新
            # scaler.step()会检查梯度是否有Inf/NaN，只有正常时才调用optimizer.step()
            scaler.step(optimizer)
            
            # 步骤5.5.4: 更新scaler的缩放因子
            # 如果上一步检测到溢出，scaler会自动减小缩放因子
            scaler.update()
            
            # 步骤5.5.5: 更新学习率
            # 根据scheduler的策略（如cosine退火）调整下一轮的学习率
            scheduler.step()
            
            # 步骤5.5.6: 清零梯度，准备下一轮累积
            # set_to_none=True比zero_()更快且节省内存
            optimizer.zero_grad(set_to_none=True)
            
            # 步骤5.5.7: 递增全局步数计数器
            global_step += 1

            # 步骤5.6: 记录当前batch的损失统计
            # 获取当前batch的样本数（batch size）
            # batch["clean"]: [B, C, H, W] 或 [B, H, W]
            # shape[0] = B (batch size)
            batch_size = int(batch["clean"].shape[0])
            
            # 更新各损失项的平均值累加器
            # loss_parts是字典，键为损失名称，值为该batch的损失值（float）
            for key, value in loss_parts.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(value, batch_size)
            
            # 更新梯度范数的累加器
            meters["grad_norm"].update(grad_norm, batch_size)

            # 步骤5.7: 定期输出训练日志
            # 条件: global_step是log_every的倍数，或者是第1步（便于早期调试）
            if global_step % log_every == 0 or global_step == 1:
                # 计算从上一个日志点到现在的耗时（秒）
                elapsed = max(1e-6, time.perf_counter() - step_started)
                
                # 构建日志记录字典
                record = {
                    "epoch": epoch + 1,  # 轮次（从1开始显示）
                    "step": global_step,  # 全局步数
                    "lr": optimizer.param_groups[0]["lr"],  # 当前学习率
                    # 每秒处理的步数（第一步特殊处理）
                    "steps_per_sec": log_every / elapsed if global_step > 1 else 1.0 / elapsed,
                    # 权重统计信息（用于监控权重分布是否正常）
                    "w_mean": float(output.weight.detach().mean().item()),  # 权重均值
                    "w_min": float(output.weight.detach().min().item()),    # 权重最小值
                    "w_max": float(output.weight.detach().max().item()),    # 权重最大值
                    # 展开各损失项的平均值
                    **{key: meter.avg for key, meter in meters.items()},
                }
                
                # 格式化并打印日志（JSON格式）
                print(format_log("train", record))
                
                # 重置计时器，为下一个日志周期做准备
                step_started = time.perf_counter()

            # 步骤5.8: 检查是否达到总步数上限
            # 如果达到，提前退出训练循环
            if global_step >= total_steps:
                break

    # 步骤6: 返回本轮训练的平均损失和更新后的global_step
    # {key: meter.avg} 提取每个累加器的平均值，形成最终的结果字典
    return {key: meter.avg for key, meter in meters.items()}, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """执行验证循环，计算各项评估指标。
    
    在验证阶段，模型不进行梯度计算和参数更新，仅评估其在验证集上的性能。
    使用@torch.no_grad()装饰器禁用梯度计算，节省显存并加速推理。
    
    Args:
        model: 待评估的神经网络模型
        loader: 验证数据加载器
        device: 计算设备（CPU或GPU）
        amp_dtype: 混合精度数据类型，None表示禁用AMP
        
    Returns:
        dict: 包含以下验证指标的字典:
            - psnr: 预测结果的峰值信噪比 (dB)
            - psnr_dnr2: 2DNR的PSNR (dB)
            - psnr_dnr3: 3DNR的PSNR (dB)
            - psnr_gain_dnr2: 预测相对2DNR的PSNR提升 (dB)
            - psnr_gain_dnr3: 预测相对3DNR的PSNR提升 (dB)
            - snr: 预测结果的信噪比 (dB)
            - ssim: 结构相似性指数 (0~1)
            - motion_psnr: 运动区域预测的PSNR (dB)
            - motion_psnr_dnr3: 运动区域3DNR的PSNR (dB)
            - motion_psnr_gain_dnr3: 运动区域预测相对3DNR的PSNR提升 (dB)
            - weight_mean: 融合权重的平均值 (0~1)
            - weight_std: 融合权重的标准差
            - plane_mean_gap: W4模式下Bayer平面均值的最大差异
            
    Note:
        验证流程:
        1. 设置模型为评估模式 model.eval()（禁用dropout等训练特定行为）
        2. 初始化MetricAccumulator累加器
        3. 对每个验证batch:
           a. 将数据移至指定设备
           b. 在AMP上下文中执行前向传播（无梯度）
           c. 更新指标累加器（PSNR、SSIM、SNR等）
        4. 计算并返回所有指标的最终值
        
        张量形状说明:
        - 输入batch与train_one_epoch相同
        - output.prediction: [B, C, H, W] 融合预测
        - output.weight: [B, 1, H, W] 融合权重
        - output.packed_weight: [B, K, H, W] 打包权重
        
        关键区别:
        - 不计算损失，不调用backward()
        - 不更新参数，仅评估性能
        - 使用no_grad上下文节省显存
    """
    # 步骤1: 设置模型为评估模式
    # 这会禁用dropout、冻结batchnorm的统计量等
    model.eval()
    
    # 步骤2: 初始化指标累加器
    # MetricAccumulator内部维护多个AverageMeter和累积统计量
    # 包括: sse_pred, sse_dnr2, sse_dnr3, signal, count, motion_sse等
    metrics = MetricAccumulator()
    
    # 步骤3: 遍历验证集中的所有batch
    for batch in loader:
        # 步骤3.1: 将batch数据移至指定设备
        batch = move_batch_to_device(batch, device)
        
        # 步骤3.2: 在AMP上下文中执行前向传播（无梯度计算）
        # @torch.no_grad()装饰器已禁用梯度追踪，进一步节省显存
        with autocast_context(device, amp_dtype):
            # 模型前向传播：与训练时相同
            # 输入:
            #   prev4: [B, C, H, W]
            #   curr4: [B, C, H, W]
            #   motion_prior: [B, 1, H//2, W//2]
            #   dnr2: [B, C, H, W]
            #   dnr3: [B, C, H, W]
            # 输出:
            #   prediction: [B, C, H, W]
            #   weight: [B, 1, H, W]
            #   packed_weight: [B, K, H, W]
            output = model.forward_packed(
                prev4=batch["prev4"],
                curr4=batch["curr4"],
                motion_prior=batch["motion_prior"],
                dnr2=batch["dnr2"],
                dnr3=batch["dnr3"],
            )
        
        # 步骤3.3: 更新指标累加器
        # 内部计算:
        #   1. 全局SSE（平方误差总和）用于PSNR
        #   2. 信号能量用于SNR
        #   3. 运动区域SSE用于motion_psnr
        #   4. SSIM分数（基于11×11窗口）
        #   5. 权重统计（均值、标准差、Bayer平面差异）
        # 输入:
        #   batch: 包含'clean', 'dnr2', 'dnr3', 'motion_prior'等键
        #   output: 包含'prediction', 'weight', 'packed_weight'属性
        metrics.update(batch, output)
    
    # 步骤4: 计算并返回所有指标的最终值
    # 内部基于累积的SSE、count等统计量计算PSNR、SNR等
    # 返回字典包含13个指标（见Returns部分）
    return metrics.compute()


def run_dry_pass(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_config: LossConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    grad_clip_norm: float,
) -> None:
    """执行干运行测试，验证模型的前向/后向传播是否正常。
    
    干运行（dry run）是在正式训练前进行的单次测试，用于：
    1. 验证模型架构是否正确（无维度不匹配错误）
    2. 验证损失计算是否正常（无NaN/Inf）
    3. 验证梯度流动是否正常（梯度裁剪有效）
    4. 输出关键张量的形状和初始损失值，便于调试
    
    Args:
        model: 待测试的神经网络模型
        loader: 数据加载器（仅取第一个batch）
        optimizer: 优化器（用于梯度unscale操作）
        scaler: 梯度缩放器（用于混合精度训练）
        loss_config: 损失函数配置对象
        device: 计算设备（CPU或GPU）
        amp_dtype: 混合精度数据类型，None表示禁用AMP
        grad_clip_norm: 梯度裁剪的L2范数阈值
        
    Returns:
        None（直接打印JSON格式的测试结果）
        
    Note:
        干运行流程:
        1. 设置模型为训练模式
        2. 从loader中取出第一个batch
        3. 执行前向传播（AMP上下文）
        4. 计算损失
        5. 执行反向传播
        6. 梯度unscale和裁剪
        7. 打印测试结果（JSON格式）
        
        输出内容包括:
        - dry_run: True（标识这是干运行）
        - loss: 总损失值
        - grad_norm: 梯度L2范数
        - prediction_shape: 预测张量的形状
        - weight_shape: 权重张量的形状
        - packed_weight_shape: 打包权重张量的形状
        - 各分项损失值（loss_rec, loss_tv等）
        
        张量形状示例（假设B=2, C=4, H=256, W=256, K=4）:
        - prediction_shape: [2, 4, 256, 256]
        - weight_shape: [2, 1, 256, 256]
        - packed_weight_shape: [2, 4, 256, 256]
        
        典型用途:
        - 训练开始前快速验证配置是否正确
        - 修改模型架构后检查维度兼容性
        - 调试新的损失函数或正则化项
    """
    # 步骤1: 设置模型为训练模式
    # 虽然是测试，但需要训练模式以启用某些层（如dropout）的正常行为
    model.train()
    
    # 步骤2: 从数据加载器中取出第一个batch
    # iter(loader)创建迭代器，next()获取第一个元素
    # 这个batch用于测试，不会进行完整的训练循环
    batch = move_batch_to_device(next(iter(loader)), device)
    
    # 步骤3: 在AMP上下文中执行前向传播和损失计算
    with autocast_context(device, amp_dtype):
        # 步骤3.1: 模型前向传播
        # 输入张量形状（示例）:
        #   prev4: [B, C, H, W] = [2, 4, 256, 256]
        #   curr4: [B, C, H, W] = [2, 4, 256, 256]
        #   motion_prior: [B, 1, H//2, W//2] = [2, 1, 128, 128]
        #   dnr2: [B, C, H, W] = [2, 4, 256, 256]
        #   dnr3: [B, C, H, W] = [2, 4, 256, 256]
        # 输出对象属性:
        #   prediction: [B, C, H, W] = [2, 4, 256, 256]
        #   weight: [B, 1, H, W] = [2, 1, 256, 256]
        #   packed_weight: [B, K, H, W] = [2, 4, 256, 256] (W4模式)
        output = model.forward_packed(
            prev4=batch["prev4"],
            curr4=batch["curr4"],
            motion_prior=batch["motion_prior"],
            dnr2=batch["dnr2"],
            dnr3=batch["dnr3"],
        )
        
        # 步骤3.2: 计算损失
        # 输入:
        #   batch: 包含'clean', 'motion_prior', 'curr4', 'dnr2', 'dnr3'
        #   output: 包含'prediction', 'weight', 'packed_weight'
        #   loss_config: 损失配置
        # 返回:
        #   loss: 标量张量 [] 总损失
        #   parts: 字典，包含各分项损失的浮点数值
        loss, parts = compute_loss(batch, output, loss_config)
    
    # 步骤4: 反向传播
    # scaler.scale(loss)将损失乘以缩放因子（防止fp16下溢）
    # .backward()计算梯度并累积到参数的.grad属性中
    scaler.scale(loss).backward()
    
    # 步骤5: 梯度unscale
    # 将梯度除以缩放因子，恢复到原始尺度
    # 这一步必须在梯度裁剪之前执行
    scaler.unscale_(optimizer)
    
    # 步骤6: 梯度裁剪并获取梯度范数
    # clip_grad_norm_计算所有参数梯度的L2范数
    # 如果超过grad_clip_norm，则按比例缩放所有梯度
    # grad_norm: float 标量，表示裁剪前的梯度范数
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm).item())
    
    # 步骤7: 打印测试结果（JSON格式）
    # 使用json.dumps格式化输出，ensure_ascii=False支持中文，indent=2美化输出
    print(
        json.dumps(
            {
                "dry_run": True,  # 标识这是干运行测试
                "loss": float(loss.detach().item()),  # 总损失值（标量）
                "grad_norm": grad_norm,  # 梯度L2范数
                # 关键张量的形状（用于验证维度正确性）
                "prediction_shape": list(output.prediction.shape),  # [B, C, H, W]
                "weight_shape": list(output.weight.shape),          # [B, 1, H, W]
                "packed_weight_shape": list(output.packed_weight.shape),  # [B, K, H, W]
                # 展开各分项损失值
                **parts,  # {'loss_rec': float, 'loss_tv': float, ...}
            },
            ensure_ascii=False,  # 允许非ASCII字符（如中文）
            indent=2,  # JSON缩进2空格，便于阅读
        )
    )

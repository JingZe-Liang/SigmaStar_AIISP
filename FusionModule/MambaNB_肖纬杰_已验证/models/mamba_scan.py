from __future__ import annotations

"""Reference Mamba selective-scan kernels used by ST-Mamba-Lite."""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # Optional fused CUDA/Triton kernel from mamba-ssm.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan_fn
except Exception:  # pragma: no cover - optional dependency may fail to load CUDA extensions
    _mamba_selective_scan_fn = None


ScanBackend = Literal["auto", "reference", "mamba_ssm"]


def _validate_sequence(name: str, sequence: Tensor, channels: int | None = None) -> None:
    """验证输入序列的维度是否符合要求。
    
    Args:
        name: 张量的名称（用于错误提示）
        sequence: 待验证的张量，期望形状为 [B, L, C]
                  B: batch size（批次大小）
                  L: sequence length（序列长度）
                  C: channels（通道数）
        channels: 期望的通道数，如果提供则进行验证
    """
    if sequence.ndim != 3:
        raise ValueError(f"{name} must have shape [B, L, C], got {sequence.shape}")
    if channels is not None and sequence.shape[-1] != channels:
        raise ValueError(f"{name} channel mismatch: expected {channels}, got {sequence.shape[-1]}")


def selective_scan_reference(
    input_sequence: Tensor,
    delta: Tensor,
    state_transition: Tensor,
    input_matrix: Tensor,
    readout_matrix: Tensor,
    skip: Tensor,
) -> Tensor:
    """参考实现：Mamba 选择性扫描算法（遵循原始 Mamba 参数化）。
    
    该函数实现了状态空间模型（SSM）的核心扫描操作，通过逐时间步递推计算隐藏状态和输出。
    
    Args:
        input_sequence: 输入序列，形状 [B, L, C]
                       B: batch size
                       L: sequence length
                       C: inner_channels（内部通道数）
        delta: 时间步长参数，形状 [B, L, C]，控制每个时间步的状态更新速度
        state_transition: 状态转移矩阵 A，形状 [C, N]
                         C: inner_channels
                         N: state_dim（状态维度）
        input_matrix: 输入投影矩阵 B，形状 [B, L, N]
        readout_matrix: 输出投影矩阵 C，形状 [B, L, N]
        skip: 跳跃连接系数 D，形状 [C]
    
    Returns:
        输出序列，形状 [B, L, C]
    """

    _validate_sequence("input_sequence", input_sequence)
    _validate_sequence("delta", delta, input_sequence.shape[-1])

    if input_matrix.ndim != 3:
        raise ValueError(f"input_matrix must have shape [B, L, N], got {input_matrix.shape}")
    if readout_matrix.ndim != 3:
        raise ValueError(f"readout_matrix must have shape [B, L, N], got {readout_matrix.shape}")

    batch_size, sequence_length, inner_channels = input_sequence.shape
    if input_matrix.shape[:2] != (batch_size, sequence_length):
        raise ValueError(
            "input_matrix must share the same batch/length as input_sequence, "
            f"got {input_matrix.shape} vs {input_sequence.shape}"
        )
    if readout_matrix.shape[:2] != (batch_size, sequence_length):
        raise ValueError(
            "readout_matrix must share the same batch/length as input_sequence, "
            f"got {readout_matrix.shape} vs {input_sequence.shape}"
        )
    if state_transition.ndim != 2 or state_transition.shape[0] != inner_channels:
        raise ValueError(
            "state_transition must have shape [C, N] with C matching input_sequence, "
            f"got {state_transition.shape} vs channels={inner_channels}"
        )
    if skip.ndim != 1 or skip.shape[0] != inner_channels:
        raise ValueError(
            f"skip must have shape [C] with C={inner_channels}, got {skip.shape}"
        )

    # 将数据转换为计算精度（float16/bfloat16 转为 float32 以保证数值稳定性）
    compute_dtype = torch.float32 if input_sequence.dtype in {torch.float16, torch.bfloat16} else input_sequence.dtype
    u = input_sequence.to(compute_dtype)          # [B, L, C]
    dt = delta.to(compute_dtype)                   # [B, L, C]
    a = state_transition.to(device=input_sequence.device, dtype=compute_dtype)  # [C, N]
    b = input_matrix.to(compute_dtype)             # [B, L, N]
    c = readout_matrix.to(compute_dtype)           # [B, L, N]
    d = skip.to(device=input_sequence.device, dtype=compute_dtype)              # [C]

    state_dim = a.shape[1]

    if sequence_length == 1:
        u0 = u[:, 0]
        dt0 = dt[:, 0]
        b0 = b[:, 0]
        c0 = c[:, 0]
        state0 = dt0.unsqueeze(-1) * u0.unsqueeze(-1) * b0.unsqueeze(1)
        y0 = (state0 * c0.unsqueeze(1)).sum(dim=-1) + d * u0
        return y0.unsqueeze(1).to(dtype=input_sequence.dtype)

    if sequence_length == 2:
        u0 = u[:, 0]
        u1 = u[:, 1]
        dt0 = dt[:, 0]
        dt1 = dt[:, 1]
        b0 = b[:, 0]
        b1 = b[:, 1]
        c0 = c[:, 0]
        c1 = c[:, 1]

        state0 = dt0.unsqueeze(-1) * u0.unsqueeze(-1) * b0.unsqueeze(1)
        y0 = (state0 * c0.unsqueeze(1)).sum(dim=-1) + d * u0

        delta_a1 = torch.exp(dt1.unsqueeze(-1) * a.unsqueeze(0))
        state1 = delta_a1 * state0 + dt1.unsqueeze(-1) * u1.unsqueeze(-1) * b1.unsqueeze(1)
        y1 = (state1 * c1.unsqueeze(1)).sum(dim=-1) + d * u1

        return torch.stack([y0, y1], dim=1).to(dtype=input_sequence.dtype)
    # 初始化隐藏状态为零张量
    state = torch.zeros(
        batch_size,
        inner_channels,
        state_dim,
        device=input_sequence.device,
        dtype=compute_dtype,
    )  # [B, C, N] - 隐藏状态：每个样本、每个通道都有一个 N 维的状态向量
    outputs: list[Tensor] = []

    # 逐时间步进行状态空间模型的递推计算
    for step in range(sequence_length):
        # 提取当前时间步的参数，形状：
        dt_step = dt[:, step]      # [B, C] - 当前时间步的时间步长
        u_step = u[:, step]        # [B, C] - 当前时间步的输入
        b_step = b[:, step]        # [B, N] - 当前时间步的输入投影矩阵
        c_step = c[:, step]        # [B, N] - 当前时间步的输出投影矩阵

        # 计算离散化的状态转移矩阵: exp(dt * A)
        # einsum("bc,cn->bcn"): 对每个样本和每个通道，将 dt[b,c] 与 A[c,n] 相乘后取指数
        # 结果形状: [B, C, N]
        delta_a = torch.exp(torch.einsum("bc,cn->bcn", dt_step, a))
        
        # 计算输入对状态的贡献: dt * B * u
        # einsum("bc,bn,bc->bcn"): 对每个样本、通道和状态维度，计算 dt[b,c] * B[b,n] * u[b,c]
        # 结果形状: [B, C, N]
        delta_b_u = torch.einsum("bc,bn,bc->bcn", dt_step, b_step, u_step)
        
        # 状态更新方程: h_t = exp(dt*A) * h_{t-1} + dt*B*u_t
        # 这是离散化后的状态空间模型核心公式
        state = delta_a * state + delta_b_u  # [B, C, N]
        
        # 计算当前时间步的输出: y_t = C * h_t + D * u_t
        # einsum("bcn,bn->bc"): 对每个样本和通道，将状态 state[b,c,n] 与输出矩阵 C[b,n] 做内积
        # 结果形状: [B, C]
        y_step = torch.einsum("bcn,bn->bc", state, c_step) + d * u_step
        outputs.append(y_step)

    # 将所有时间步的输出堆叠起来，并转换回原始数据类型
    # 输出形状: [B, L, C]
    return torch.stack(outputs, dim=1).to(dtype=input_sequence.dtype)


def selective_scan_mamba_ssm(
    input_sequence: Tensor,
    delta: Tensor,
    state_transition: Tensor,
    input_matrix: Tensor,
    readout_matrix: Tensor,
    skip: Tensor,
) -> Tensor:
    """Run selective scan through the optional mamba-ssm fused kernel."""
    if _mamba_selective_scan_fn is None:
        raise RuntimeError("mamba_ssm selective_scan_fn is not available")
    if not input_sequence.is_cuda:
        raise RuntimeError("mamba_ssm selective_scan_fn requires CUDA tensors")

    _validate_sequence("input_sequence", input_sequence)
    _validate_sequence("delta", delta, input_sequence.shape[-1])

    u = input_sequence.transpose(1, 2).contiguous()
    dt = delta.transpose(1, 2).contiguous()
    b = input_matrix.transpose(1, 2).contiguous()
    c = readout_matrix.transpose(1, 2).contiguous()
    a = state_transition.to(device=input_sequence.device)
    d = skip.to(device=input_sequence.device)

    output = _mamba_selective_scan_fn(
        u,
        dt,
        a,
        b,
        c,
        D=d,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.transpose(1, 2).contiguous().to(dtype=input_sequence.dtype)


class MambaSelectiveScan1D(nn.Module):
    """完整的 Mamba 一维选择性扫描模块，处理 [B, L, C] 形状的序列数据。
    
    该模块实现了 Mamba 架构的核心组件，包括：
    1. 输入投影和卷积预处理
    2. 动态参数生成（delta, B, C）
    3. 选择性扫描（状态空间模型）
    4. 门控机制和输出投影
    
    整体数据流:
        输入 [B, L, C] -> 投影 -> 卷积 -> SSM扫描 -> 门控 -> 输出 [B, L, C]
    """

    def __init__(
        self,
        channels: int,
        state_dim: int = 8,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_kernel: int = 3,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        scan_backend: ScanBackend = "auto",
    ) -> None:
        """初始化 Mamba 选择性扫描模块。
        
        Args:
            channels: 输入/输出通道数 C
            state_dim: 状态维度 N（隐藏状态的维度），默认 8
            expand: 扩展倍数，内部通道数 = channels * expand，默认 2
            dt_rank: 时间步长参数的低秩维度，默认为 ceil(channels/16)
            conv_kernel: 因果卷积的核大小，默认 3
            dt_min: 时间步长的最小值（用于初始化），默认 1e-3
            dt_max: 时间步长的最大值（用于初始化），默认 1e-1
        """
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if expand <= 0:
            raise ValueError(f"expand must be positive, got {expand}")
        if conv_kernel <= 0:
            raise ValueError(f"conv_kernel must be positive, got {conv_kernel}")
        if dt_min <= 0 or dt_max <= 0 or dt_min >= dt_max:
            raise ValueError(
                f"dt_min/dt_max must satisfy 0 < dt_min < dt_max, got {dt_min}, {dt_max}"
            )
        if scan_backend not in {"auto", "reference", "mamba_ssm"}:
            raise ValueError(f"Unsupported scan_backend: {scan_backend}")

        self.channels = channels                              # 输入通道数 C
        self.state_dim = state_dim                            # 状态维度 N
        self.expand = expand                                  # 扩展倍数
        self.inner_channels = channels * expand               # 内部通道数 C_inner = C * expand
        self.dt_rank = dt_rank or math.ceil(channels / 16)   # 时间步长的低秩维度
        self.conv_kernel = conv_kernel                        # 卷积核大小

        self.conv_kernel = conv_kernel
        self.scan_backend = scan_backend
        if self.dt_rank <= 0:
            raise ValueError(f"dt_rank must be positive, got {self.dt_rank}")

        # 输入投影层：将输入通道扩展到 2 倍内部通道（一份用于 SSM，一份用于门控）
        # 输入: [B, L, C] -> 输出: [B, L, 2*C_inner]
        self.in_proj = nn.Linear(channels, self.inner_channels * 2, bias=True)
        
        # 一维因果卷积：在 SSM 之前对输入进行局部特征提取
        # 使用分组卷积（groups=inner_channels），每个通道独立卷积
        # 输入: [B, C_inner, L] -> 输出: [B, C_inner, L+kernel-1]（后续会裁剪）
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=conv_kernel,
            groups=self.inner_channels,  # 深度可分离卷积
            padding=conv_kernel - 1,     # 保证因果性（只依赖当前和过去的信息）
            bias=True,
        )
        
        # 参数投影层：从输入特征动态生成 SSM 的参数 (delta, B, C)
        # 输入: [B, L, C_inner] -> 输出: [B, L, dt_rank + N + N]
        # 输出被分割为三部分：
        #   - dt_params: [B, L, dt_rank] 用于生成时间步长 delta
        #   - input_matrix: [B, L, N] 即 B 矩阵（输入投影）
        #   - readout_matrix: [B, L, N] 即 C 矩阵（输出投影）
        self.param_proj = nn.Linear(
            self.inner_channels,
            self.dt_rank + state_dim * 2,
            bias=False,
        )
        
        # 时间步长投影：将低秩表示映射到完整通道维度
        # 输入: [B, L, dt_rank] -> 输出: [B, L, C_inner]
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_channels, bias=True)
        
        # 状态转移矩阵 A 的对数参数（可学习）
        # 形状: [C_inner, N]，初始化为 log(1), log(2), ..., log(N) 的重复
        # 实际使用时 A = -exp(A_log)，保证 A 的元素为负值（稳定系统）
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).repeat(
                self.inner_channels,
                1,
            )
        )  # [C_inner, N]
        
        # 跳跃连接系数 D（类似于残差连接）
        # 形状: [C_inner]，初始化为全 1
        self.D = nn.Parameter(torch.ones(self.inner_channels, dtype=torch.float32))
        
        # 输出投影层：将内部通道映射回原始通道数
        # 输入: [B, L, C_inner] -> 输出: [B, L, C]
        self.out_proj = nn.Linear(self.inner_channels, channels, bias=True)

        self._reset_dt_parameters(dt_min=dt_min, dt_max=dt_max)

    def forward(self, sequence: Tensor) -> Tensor:
        """前向传播。
        
        Args:
            sequence: 输入序列，形状 [B, L, C]
        
        Returns:
            输出序列，形状 [B, L, C]
        """
        _validate_sequence("sequence", sequence, self.channels)
        sequence_length = sequence.shape[1]  # L

        # 步骤 1: 输入投影，分成两部分（SSM 分支和门控分支）
        # 输入: [B, L, C] -> 输出: [B, L, 2*C_inner] -> 分割为两个 [B, L, C_inner]
        input_projection, gate_projection = self.in_proj(sequence).chunk(2, dim=-1)
        # input_projection: [B, L, C_inner] - 用于 SSM 处理
        # gate_projection: [B, L, C_inner] - 用于门控机制
        
        # 步骤 2: 一维因果卷积（需要转置为 [B, C, L] 格式）
        # 转置: [B, L, C_inner] -> [B, C_inner, L]
        input_projection = input_projection.transpose(1, 2)
        # 卷积: [B, C_inner, L] -> [B, C_inner, L+kernel-1]
        input_projection = self.conv1d(input_projection)[..., :sequence_length]
        # 裁剪回原长度: [B, C_inner, L]
        # 激活并转置回: [B, C_inner, L] -> [B, L, C_inner]
        input_projection = F.silu(input_projection.transpose(1, 2))

        # 步骤 3: 动态生成 SSM 参数（delta, B, C）
        # 输入: [B, L, C_inner] -> 输出: [B, L, dt_rank + N + N]
        dt_params, input_matrix, readout_matrix = self.param_proj(input_projection).split(
            [self.dt_rank, self.state_dim, self.state_dim],
            dim=-1,
        )
        # dt_params: [B, L, dt_rank] - 时间步长的低秩表示
        # input_matrix: [B, L, N] - B 矩阵（输入到状态的投影）
        # readout_matrix: [B, L, N] - C 矩阵（状态到输出的投影）
        
        # 步骤 4: 计算时间步长 delta（通过 softplus 保证正值）
        # 输入: [B, L, dt_rank] -> 输出: [B, L, C_inner]
        delta = F.softplus(self.dt_proj(dt_params))  # [B, L, C_inner]
        
        # 步骤 5: 计算状态转移矩阵 A = -exp(A_log)
        # A_log: [C_inner, N] -> A: [C_inner, N]（元素均为负值，保证系统稳定）
        state_transition = -torch.exp(self.A_log)  # [C_inner, N]

        # 步骤 6: 执行选择性扫描（SSM 核心计算）
        # 输入:
        #   input_sequence: [B, L, C_inner]
        #   delta: [B, L, C_inner]
        #   state_transition: [C_inner, N]
        #   input_matrix: [B, L, N]
        #   readout_matrix: [B, L, N]
        #   skip: [C_inner]
        # 输出: [B, L, C_inner]
        output = self._selective_scan(
            input_sequence=input_projection,
            delta=delta,
            state_transition=state_transition,
            input_matrix=input_matrix,
            readout_matrix=readout_matrix,
            skip=self.D,
        )
        
        # 步骤 7: 应用门控机制（逐元素乘法）
        # output: [B, L, C_inner] * gate_projection: [B, L, C_inner] -> [B, L, C_inner]
        output = output * F.silu(gate_projection)
        
        # 步骤 8: 输出投影，映射回原始通道数
        # 输入: [B, L, C_inner] -> 输出: [B, L, C]
        return self.out_proj(output)

    def _selective_scan(
        self,
        input_sequence: Tensor,
        delta: Tensor,
        state_transition: Tensor,
        input_matrix: Tensor,
        readout_matrix: Tensor,
        skip: Tensor,
    ) -> Tensor:
        if self.scan_backend == "reference":
            return selective_scan_reference(
                input_sequence=input_sequence,
                delta=delta,
                state_transition=state_transition,
                input_matrix=input_matrix,
                readout_matrix=readout_matrix,
                skip=skip,
            )

        can_use_mamba_ssm = _mamba_selective_scan_fn is not None and input_sequence.is_cuda
        if self.scan_backend == "mamba_ssm" and not can_use_mamba_ssm:
            raise RuntimeError("scan_backend='mamba_ssm' requires mamba-ssm and CUDA tensors")

        if can_use_mamba_ssm:
            try:
                return selective_scan_mamba_ssm(
                    input_sequence=input_sequence,
                    delta=delta,
                    state_transition=state_transition,
                    input_matrix=input_matrix,
                    readout_matrix=readout_matrix,
                    skip=skip,
                )
            except Exception:
                if self.scan_backend == "mamba_ssm":
                    raise

        return selective_scan_reference(
            input_sequence=input_sequence,
            delta=delta,
            state_transition=state_transition,
            input_matrix=input_matrix,
            readout_matrix=readout_matrix,
            skip=skip,
        )

    def _reset_dt_parameters(self, dt_min: float, dt_max: float) -> None:
        """初始化时间步长相关的参数。
        
        采用特殊的初始化策略，使得初始的时间步长均匀分布在 [dt_min, dt_max] 范围内。
        
        Args:
            dt_min: 时间步长最小值
            dt_max: 时间步长最大值
        """
        # 权重初始化：使用小的随机值（Xavier 风格的缩放）
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # 偏置初始化：使得初始的 delta 值在 [dt_min, dt_max] 之间均匀分布
        # 由于 delta = softplus(dt_proj_bias)，需要反向计算 bias 的初始值
        # 1. 首先在 [dt_min, dt_max] 之间均匀采样 delta 值
        dt = torch.exp(
            torch.rand(self.inner_channels) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )  # [C_inner]，值域: [dt_min, dt_max]
        
        # 2. 计算 softplus 的反函数：inv_softplus(x) = x + log(-expm1(-x))
        #    这样 softplus(inv_softplus(x)) ≈ x
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        
        # 3. 将计算得到的偏置值复制到模型参数中
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

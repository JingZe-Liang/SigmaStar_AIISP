import torch


def back_norm(output_mid: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """
    1. 输入：Output_Mid ：Tensor（BT ,  H , W）;bit_depth --- 单位bit
    2. 输出：Output_Mid_Norm: Tensor (BT , H ,W)
    3. 逻辑：依据bit _ depth进行反归一化
    """
    max_val = (2**bit_depth) - 1
    return (output_mid * max_val).round().clamp(0, max_val)

if __name__ == "__main__":
    # 模拟归一化数据
    # 模拟归一化数据
    output_mid = torch.rand(32, 256, 256)  # BT=32, 值域 [0, 1]

    # 处理前
    print(f"处理前 - min: {output_mid.min().item():.4f}, max: {output_mid.max().item():.4f}")

    # 10-bit 反归一化
    output_10bit = back_norm(output_mid, bit_depth=10)
    print(f"处理后 (10-bit) - min: {output_10bit.min().item():.1f}, max: {output_10bit.max().item():.1f}")
    print(f"dtype: {output_10bit.dtype}\n")

    # 12-bit 反归一化
    output_12bit = back_norm(output_mid, bit_depth=12)
    print(f"处理前 - min: {output_mid.min().item():.4f}, max: {output_mid.max().item():.4f}")
    print(f"处理后 (12-bit) - min: {output_12bit.min().item():.1f}, max: {output_12bit.max().item():.1f}")


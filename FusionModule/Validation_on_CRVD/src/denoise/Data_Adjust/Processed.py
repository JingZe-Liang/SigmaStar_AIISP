import torch

from Liang.src.denoise.Data_Adjust.Before_Processing import befor_processing


def processed(output_mid_norm: torch.Tensor, t: int) -> torch.Tensor:
    """
    1. 输入 ： Output_ Mid_Norm：Tensor（BT , H , W）; T）
    2. 输出 ： Output: Tensor (B , T ,H , W)
    3. 要求 输出按行优先折叠会得到输入
    """
    bt, h, w = output_mid_norm.shape
    b = bt // t
    return output_mid_norm.reshape(b, t, h, w)




if __name__ == "__main__":
    # 模拟数据
    output_mid_norm = torch.rand(32, 256, 256)  # BT=32
    T = 8

    # 处理前
    print(f"处理前 - shape: {output_mid_norm.shape}")  # torch.Size([32, 256, 256])

    # 恢复维度
    output = processed(output_mid_norm, t=T)
    print(f"处理后 - shape: {output.shape}")  # torch.Size([4, 8, 256, 256])

    # 验证行优先顺序：完整往返测试
    raw_original = torch.randn(4, 8, 256, 256)  # B=4, T=8
    raw1, T_saved = befor_processing(raw_original)
    raw_recovered = processed(raw1, t=T_saved)

    print(f"\n往返一致性: {torch.allclose(raw_original, raw_recovered)}")  # True
    print(f"最大误差: {(raw_original - raw_recovered).abs().max().item():.10f}")  # 0.0

    # 验证索引对应关系
    print(f"\nraw_original[0, 1] == raw_recovered[0, 1]: {torch.equal(raw_original[0, 1], raw_recovered[0, 1])}")
    print(f"raw_original[1, 0] == raw_recovered[1, 0]: {torch.equal(raw_original[1, 0], raw_recovered[1, 0])}")

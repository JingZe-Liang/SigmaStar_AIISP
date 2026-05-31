import torch


def norm(raw0: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """
    1. 输入：RAW0 ：Tensor （BT , H , W） ; bit_depth---单位bit
    2. 输出 ：RAW1 ； Tensor(BT , H , W),
    3. 逻辑 ：依据bit_depth 归一化到了0-1
    """
    max_val = (2**bit_depth) - 1  # 最大像素值
    return raw0.float() / max_val

if __name__ == "__main__":
    # 10-bit RAW 数据
    raw0 = torch.randint(0, 2047, (32, 256, 256))  # BT=32, 范围 [0, 1023]
    raw1 = norm(raw0, bit_depth=10)

    print(f"min: {raw1.min().item():.4f}, max: {raw1.max().item():.4f}")  # 约 0.0000, 1.0000
    print(f"dtype: {raw1.dtype}")  # dtype: torch.float32

    # 12-bit RAW 数据
    raw0_12bit = torch.randint(0, 4096, (32, 256, 256))
    raw1_12bit = norm(raw0_12bit, bit_depth=12)
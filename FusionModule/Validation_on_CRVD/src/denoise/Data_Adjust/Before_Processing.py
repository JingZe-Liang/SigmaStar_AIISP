import torch

def befor_processing(raw: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    1. 输入：RAW  :  Tensor（B，T，H，W）
    2. 输出：RAW0：Tensor （BT，H，W）; T
    3， 不进行归一化
    """
    b, t, h, w = raw.shape
    raw1 = raw.reshape(b * t, h, w)
    return raw1, t

if __name__ == "__main__":
    raw = torch.randn(4, 8, 256, 256)  # B=4, T=8
    raw1, T = befor_processing(raw)

    print(raw1.shape)  # torch.Size([32, 256, 256])
    print(T)  # 8

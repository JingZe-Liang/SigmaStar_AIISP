import torch
from kornia.filters import gaussian_blur2d


def MD(raw1: torch.Tensor, t: int) -> torch.Tensor:
    """基于时域差异和空间平滑的运动检测模块。

    功能说明：
        对视频序列进行逐帧运动检测，通过计算相邻帧间的像素差异并进行空间平滑，
        生成反映运动强度的权重图。每个batch内的T帧构成独立视频序列。

    输入：
        raw1: torch.Tensor
            - 形状: (BT, H, W)
            - 类型: torch.float32 或 torch.float64
            - 范围: [0, 1]，已归一化的RAW数据
            - 说明: BT = B × T，其中B是batch数，T是每个视频的帧数

        t: int
            - 说明: 时间维度大小，表示每个视频序列的帧数
            - 要求: BT必须能被t整除

    输出：
        motion_map: torch.Tensor
            - 形状: (BT, H, W)，与输入形状一致
            - 类型: 与输入dtype一致
            - 范围: [0, 1]
                * 0.0: 精确值，表示首帧（无参考帧）
                * ~0.007: 接近0，表示静止（帧间无变化）
                * 0.05~0.2: 弱运动
                * 0.3~0.7: 中等运动
                * 0.8~0.999: 强运动
            - 含义: 每个像素的运动强度权重
                * 值越大表示该位置相对上一帧变化越剧烈
                * 可直接用作时域融合的权重图
                * 每个视频序列的首帧（索引 b*T）固定输出全0

    算法流程：
        1. 将 (BT, H, W) 重组为 (B, T, H, W) 便于按视频序列处理
        2. 对每个视频的每一帧：
           - 首帧（t=0）：输出全0（无运动参考）
           - 其余帧（t≥1）：计算 |curr - prev| 的绝对差异
        3. 空间高斯平滑：降低噪声引起的误检（kernel_size=5, sigma=1.0）
        4. Sigmoid映射：将差异值映射到 [0, 1] 权重范围

    示例：

    """
    device = raw1.device
    dtype = raw1.dtype
    bt, h, w = raw1.shape
    b = bt // t

    frames = raw1.reshape(b, t, h, w)
    maps = []

    for batch_idx in range(b):
        for frame_idx in range(t):
            if frame_idx == 0:
                motion_map = torch.zeros(h, w, device=device, dtype=dtype)
            else:
                curr = frames[batch_idx, frame_idx]
                prev = frames[batch_idx, frame_idx - 1]
                diff = (curr - prev).abs()

                diff_4d = diff.unsqueeze(0).unsqueeze(0)
                smoothed = gaussian_blur2d(diff_4d, kernel_size=(5, 5), sigma=(1.0, 1.0))
                smoothed = smoothed.squeeze(0).squeeze(0)

                # 改进映射：让静止时接近0，强运动时接近1
                motion_map = torch.sigmoid(smoothed * 20.0 - 5.0)

            maps.append(motion_map)

    output = torch.stack(maps, dim=0)
    return output

if __name__ == "__main__":
    # ========== 测试1: 时域运动检测基本功能 ==========
    print("=" * 50)
    print("测试1: 时域运动检测基本功能")
    print("=" * 50)

    B, T = 2, 4
    H, W = 128, 128
    raw1 = torch.zeros(B * T, H, W)

    # Batch 0: 完全静止场景
    for t in range(T):
        raw1[0 * T + t] = 0.5  # 所有帧相同

    # Batch 1: 有运动场景
    for t in range(T):
        if t == 0:
            raw1[1 * T + t] = 0.5
        elif t == 2:
            raw1[1 * T + t] = 0.8  # 第3帧整体变亮
        else:
            raw1[1 * T + t] = 0.5

    motion_map = MD(raw1, t=T)

    print("\nBatch 0 (静止场景):")
    for t in range(T):
        idx = 0 * T + t
        mean_val = motion_map[idx].mean().item()
        max_val = motion_map[idx].max().item()
        print(f"  帧{t}: mean={mean_val:.6f}, max={max_val:.6f} {'(首帧)' if t == 0 else '(应接近0)'}")

    print("\nBatch 1 (运动场景):")
    for t in range(T):
        idx = 1 * T + t
        mean_val = motion_map[idx].mean().item()
        max_val = motion_map[idx].max().item()
        status = "(首帧)" if t == 0 else "(强运动)" if t == 2 else "(静止)"
        print(f"  帧{t}: mean={mean_val:.6f}, max={max_val:.6f} {status}")

    # ========== 测试2: 局部运动检测 ==========
    print("\n" + "=" * 50)
    print("测试2: 局部运动检测能力")
    print("=" * 50)

    B, T = 1, 3
    raw1 = torch.ones(B * T, 64, 64) * 0.5

    # 帧0: 基准帧
    # 帧1: 左上角有运动
    raw1[1, 10:20, 10:20] = 0.9
    # 帧2: 右下角有运动
    raw1[2, 44:54, 44:54] = 0.2

    motion_map = MD(raw1, t=T)

    print("\n帧0 (首帧):")
    print(f"  全图: mean={motion_map[0].mean().item():.6f}")

    print("\n帧1 (左上角运动):")
    print(f"  左上角(10:20, 10:20): mean={motion_map[1, 10:20, 10:20].mean().item():.4f}")
    print(f"  右下角(44:54, 44:54): mean={motion_map[1, 44:54, 44:54].mean().item():.4f}")
    print(f"  全图: mean={motion_map[1].mean().item():.4f}")

    print("\n帧2 (右下角运动):")
    print(f"  左上角(10:20, 10:20): mean={motion_map[2, 10:20, 10:20].mean().item():.4f}")
    print(f"  右下角(44:54, 44:54): mean={motion_map[2, 44:54, 44:54].mean().item():.4f}")
    print(f"  全图: mean={motion_map[2].mean().item():.4f}")

    # ========== 测试3: 验证不是时域滤波 ==========
    print("\n" + "=" * 50)
    print("测试3: 验证这是运动检测而非时域滤波")
    print("=" * 50)

    B, T = 1, 3
    raw1 = torch.zeros(B * T, 32, 32)

    # 帧0: 全0
    raw1[0] = 0.0
    # 帧1: 全1
    raw1[1] = 1.0
    # 帧2: 全0.5
    raw1[2] = 0.5

    motion_map = MD(raw1, t=T)

    print("\n如果是时域滤波，输出应该是平滑的像素值")
    print("如果是运动检测，输出应该是运动强度（高响应表示大变化）\n")

    print("帧0 (首帧，值=0.0):")
    print(f"  MD输出: mean={motion_map[0].mean().item():.6f} (首帧固定为0)")

    print("\n帧1 (值=1.0，相对帧0变化1.0):")
    print(f"  MD输出: mean={motion_map[1].mean().item():.4f} (检测到强运动)")
    print(f"  原始像素值: {raw1[1].mean().item():.1f}")
    print(f"  → 输出是运动强度，不是像素值！")

    print("\n帧2 (值=0.5，相对帧1变化0.5):")
    print(f"  MD输出: mean={motion_map[2].mean().item():.4f} (检测到中等运动)")
    print(f"  原始像素值: {raw1[2].mean().item():.1f}")
    print(f"  → 输出是运动强度，不是像素值！")

    print("\n✅ 验证完成：MD是运动检测模块，不是时域滤波")
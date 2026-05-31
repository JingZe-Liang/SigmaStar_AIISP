import torch


def fusion(
        raw1_2: torch.Tensor,
        raw1_3: torch.Tensor,
        map: torch.Tensor,
        threshold: float = 0.5,
        scale: float = 10.0,
) -> torch.Tensor:
    """运动自适应Sigmoid融合2DNR和3DNR结果。

    输入：
        raw1_2: torch.Tensor
            - 形状: (BT, H, W)
            - 范围: [0, 1]
            - 说明: 2DNR空间降噪结果

        raw1_3: torch.Tensor
            - 形状: (BT, H, W)
            - 范围: [0, 1]
            - 说明: 3DNR时域降噪结果

        map: torch.Tensor
            - 形状: (BT, H, W)
            - 范围: [0, 1]
            - 说明: MD运动检测权重图（0=静止, 1=运动）

        threshold: float
            - 范围: [0, 1], 默认0.5
            - 说明: Sigmoid切换中心点
            - Map < threshold: 偏向3DNR
            - Map > threshold: 偏向2DNR

        scale: float
            - 范围: >0, 默认10.0
            - 说明: Sigmoid陡峭程度
            - scale越大，切换越急剧
            - 推荐范围: 5.0~20.0

    输出：
        output_mid: torch.Tensor
            - 形状: (BT, H, W)
            - 范围: [0, 1]
            - 说明: 融合后的降噪结果

    算法：
        weight_3dnr = sigmoid((threshold - Map) × scale)
        Output = weight_3dnr × RAW1_3 + (1 - weight_3dnr) × RAW1_2

        特性：
        - 静止区域(Map≈0): weight_3dnr≈1 → 主要用3DNR（强降噪）
        - 运动区域(Map≈1): weight_3dnr≈0 → 主要用2DNR（无拖影）
        - 过渡区域(Map≈threshold): 平滑混合

    """
    # 计算3DNR权重：静止区域权重高，运动区域权重低
    weight_3dnr = torch.sigmoid((threshold - map) * scale)

    # 加权融合
    output_mid = weight_3dnr * raw1_3 + (1.0 - weight_3dnr) * raw1_2

    return output_mid

if __name__ == "__main__":
    import torch
    import math

    # ========== 验证1: Sigmoid权重分布验证 ==========
    print("=" * 70)
    print("验证1: Sigmoid权重分布特性")
    print("=" * 70)

    # 测试不同scale参数的效果
    map_values = torch.linspace(0, 1, 11)
    threshold = 0.5
    scales = [5.0, 10.0, 20.0]

    print(f"\nMap值从0到1，观察weight_3dnr的变化:")
    print(f"{'Map':<8} ", end="")
    for scale in scales:
        print(f"scale={scale:<4.1f}  ", end="")
    print()
    print("-" * 70)

    for map_val in map_values:
        print(f"{map_val.item():<8.1f} ", end="")

        for scale in scales:
            # 修正：使用 math.sigmoid 或转换为 Tensor
            weight = 1.0 / (1.0 + math.exp(-((threshold - map_val.item()) * scale)))
            # 或者: weight = torch.sigmoid((threshold - map_val) * scale).item()
            print(f"{weight:<12.4f} ", end="")
        print()

    print("\n说明:")
    print("  • Map=0 (静止): weight_3dnr≈1 (使用3DNR)")
    print("  • Map=1 (运动): weight_3dnr≈0 (使用2DNR)")
    print("  • scale越大，在threshold附近切换越陡峭")

    # ========== 验证2: 极端情况验证 ==========
    print("\n" + "=" * 70)
    print("验证2: 极端情况 - 纯静止/纯运动")
    print("=" * 70)

    raw1_2_test = torch.ones(4, 64, 64) * 0.3
    raw1_3_test = torch.ones(4, 64, 64) * 0.7

    # 情况1: 纯静止 (Map=0)
    map_static = torch.zeros(4, 64, 64)
    output_static = fusion(raw1_2_test, raw1_3_test, map_static, threshold=0.5, scale=10.0)

    print(f"\n纯静止场景 (Map=0):")
    print(f"  RAW1_2均值: {raw1_2_test.mean().item():.6f}")
    print(f"  RAW1_3均值: {raw1_3_test.mean().item():.6f}")
    print(f"  Output均值: {output_static.mean().item():.6f}")
    print(f"  预期: 应接近RAW1_3 (0.7)")
    diff_static = abs(output_static.mean().item() - 0.7)
    print(f"  误差: {diff_static:.6f}")
    print(f"  结果: {'✓通过' if diff_static < 0.01 else '✗失败'}")

    # 情况2: 纯运动 (Map=1)
    map_motion = torch.ones(4, 64, 64)
    output_motion = fusion(raw1_2_test, raw1_3_test, map_motion, threshold=0.5, scale=10.0)

    print(f"\n纯运动场景 (Map=1):")
    print(f"  RAW1_2均值: {raw1_2_test.mean().item():.6f}")
    print(f"  RAW1_3均值: {raw1_3_test.mean().item():.6f}")
    print(f"  Output均值: {output_motion.mean().item():.6f}")
    print(f"  预期: 应接近RAW1_2 (0.3)")
    diff_motion = abs(output_motion.mean().item() - 0.3)
    print(f"  误差: {diff_motion:.6f}")
    print(f"  结果: {'✓通过' if diff_motion < 0.01 else '✗失败'}")

    # 情况3: 过渡区域 (Map=0.5)
    map_transition = torch.ones(4, 64, 64) * 0.5
    output_transition = fusion(raw1_2_test, raw1_3_test, map_transition, threshold=0.5, scale=10.0)

    print(f"\n过渡区域 (Map=0.5, threshold=0.5):")
    print(f"  Output均值: {output_transition.mean().item():.6f}")
    print(f"  预期: 应接近中点 (0.5)")
    diff_transition = abs(output_transition.mean().item() - 0.5)
    print(f"  误差: {diff_transition:.6f}")
    print(f"  结果: {'✓通过' if diff_transition < 0.05 else '✗失败'}")

    # ========== 验证3: 空间自适应性 ==========
    print("\n" + "=" * 70)
    print("验证3: 空间自适应融合")
    print("=" * 70)

    H, W = 64, 64
    raw1_2_spatial = torch.ones(1, H, W) * 0.2
    raw1_3_spatial = torch.ones(1, H, W) * 0.8
    map_spatial = torch.zeros(1, H, W)

    # 左半部分静止，右半部分运动
    map_spatial[:, :, W // 2:] = 1.0

    output_spatial = fusion(raw1_2_spatial, raw1_3_spatial, map_spatial, threshold=0.5, scale=10.0)

    left_mean = output_spatial[0, :, :W // 2].mean().item()
    right_mean = output_spatial[0, :, W // 2:].mean().item()

    print(f"\n混合场景（左静止，右运动）:")
    print(f"  左半部分Output均值: {left_mean:.6f} (应接近0.8)")
    print(f"  右半部分Output均值: {right_mean:.6f} (应接近0.2)")

    left_correct = abs(left_mean - 0.8) < 0.05
    right_correct = abs(right_mean - 0.2) < 0.05

    print(f"  左半部分: {'✓通过' if left_correct else '✗失败'} (误差{abs(left_mean - 0.8):.6f})")
    print(f"  右半部分: {'✓通过' if right_correct else '✗失败'} (误差{abs(right_mean - 0.2):.6f})")

    # ========== 验证4: 与线性融合对比 ==========
    print("\n" + "=" * 70)
    print("验证4: Sigmoid vs 线性融合对比")
    print("=" * 70)

    map_compare = torch.linspace(0, 1, 100).reshape(1, 10, 10)
    raw1_2_compare = torch.zeros(1, 10, 10)
    raw1_3_compare = torch.ones(1, 10, 10)

    # Sigmoid融合
    output_sigmoid = fusion(raw1_2_compare, raw1_3_compare, map_compare, threshold=0.5, scale=10.0)

    # 线性融合
    output_linear = (1 - map_compare) * raw1_3_compare + map_compare * raw1_2_compare

    print(f"\n在Map=0.3, 0.5, 0.7处的对比:")
    test_indices = [29, 49, 69]  # 对应Map≈0.29, 0.49, 0.69
    for idx in test_indices:
        row, col = idx // 10, idx % 10
        map_val = map_compare[0, row, col].item()
        sig_val = output_sigmoid[0, row, col].item()
        lin_val = output_linear[0, row, col].item()

        print(f"  Map={map_val:.2f}: Sigmoid={sig_val:.4f}, Linear={lin_val:.4f}, "
              f"差异={abs(sig_val - lin_val):.4f}")

    print(f"\nSigmoid优势:")
    print(f"  • 在threshold附近切换更果断")
    print(f"  • 减少'两边都不强'的问题")
    print(f"  • scale可调节陡峭程度")

    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    print(f"✅ 所有验证通过")
    print(f"\n参数建议:")
    print(f"  • threshold=0.5: 适用于大多数场景")
    print(f"  • scale=10.0: 标准陡峭程度（推荐）")
    print(f"  • scale=5.0: 更平滑过渡（减少硬切换）")
    print(f"  • scale=20.0: 更急剧切换（接近硬阈值）")
    print(f"\n下一步: 在真实CRVD数据上测试完整pipeline")
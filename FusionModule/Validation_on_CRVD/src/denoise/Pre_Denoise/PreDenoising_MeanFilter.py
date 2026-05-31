import torch
from kornia.filters import box_blur


def PreDenoising(raw0_1: torch.Tensor, t: int) -> torch.Tensor:
    """

    输入：raw0_1 (BT, H, W), 范围[0,1], 已归一化
          t: 时间维度大小
    输出：raw1 (BT, H, W), 范围[0,1], 平滑后数据

    注意：仅适用于高噪声场景(σ≥0.05)，低噪声会损失细节
          建议先在真实数据上验证效果再使用
    """
    # 添加通道维度: (BT, H, W) -> (BT, 1, H, W)
    raw_4d = raw0_1.unsqueeze(1)

    # 均值滤波（kernel_size=3，最快速）
    denoised = box_blur(raw_4d, kernel_size=(3, 3))

    # 移除通道维度: (BT, 1, H, W) -> (BT, H, W)
    raw1 = denoised.squeeze(1)

    return raw1

if __name__ == "__main__":
    print("=" * 70)
    print("均值滤波在不同噪声水平下的效果验证")
    print("=" * 70)

    # 测试多个噪声强度
    noise_levels = [0.01, 0.03, 0.05, 0.45, 0.4, 0.8, 0.7, 0.6, 0.5]

    results = []

    for noise_std in noise_levels:
        # 生成干净数据
        raw_clean = torch.rand(32, 256, 256)

        # 添加高斯噪声
        noise = torch.randn_like(raw_clean) * noise_std
        raw_noisy = (raw_clean + noise).clamp(0, 1)

        # 均值滤波降噪
        raw_denoised = PreDenoising(raw_noisy, t=8)

        # 计算误差
        error_noisy = (raw_clean - raw_noisy).abs().mean().item()
        error_denoised = (raw_clean - raw_denoised).abs().mean().item()
        improvement = error_noisy - error_denoised
        improvement_percent = (improvement / error_noisy * 100) if error_noisy > 0 else 0

        # 计算PSNR
        mse_noisy = ((raw_clean - raw_noisy) ** 2).mean().item()
        mse_denoised = ((raw_clean - raw_denoised) ** 2).mean().item()
        psnr_noisy = 10 * torch.log10(torch.tensor(1.0 / mse_noisy)).item()
        psnr_denoised = 10 * torch.log10(torch.tensor(1.0 / mse_denoised)).item()
        psnr_gain = psnr_denoised - psnr_noisy

        results.append({
            'noise_std': noise_std,
            'error_noisy': error_noisy,
            'error_denoised': error_denoised,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'psnr_gain': psnr_gain
        })

        # 实时输出
        status = "✓有效" if improvement > 0 else "✗无效"
        print(f"\n噪声σ={noise_std:.2f}:")
        print(f"  降噪前误差: {error_noisy:.6f}  |  PSNR: {psnr_noisy:.2f} dB")
        print(f"  降噪后误差: {error_denoised:.6f}  |  PSNR: {psnr_denoised:.2f} dB")
        print(
            f"  改善程度: {improvement:+.6f} ({improvement_percent:+.1f}%)  |  PSNR增益: {psnr_gain:+.2f} dB  {status}")

    # ========== 总结分析 ==========
    print("\n" + "=" * 70)
    print("总结：均值滤波有效性分析")
    print("=" * 70)

    # 找出转折点
    effective_threshold = None
    for r in results:
        if r['improvement'] > 0 and effective_threshold is None:
            effective_threshold = r['noise_std']

    print(f"\n关键发现:")
    print(f"  • 有效噪声阈值: σ ≥ {effective_threshold:.2f}")
    print(f"  • 最佳改善场景: 高噪声区域\n")

    # 详细统计表
    print("详细统计表:")
    print("-" * 70)
    print(f"{'噪声σ':<10} {'降噪前误差':<15} {'降噪后误差':<15} {'改善率':<12} {'建议':<10}")
    print("-" * 70)

    for r in results:
        recommendation = "✓使用" if r['improvement'] > 0 else "✗跳过"
        print(f"{r['noise_std']:<10.2f} {r['error_noisy']:<15.6f} {r['error_denoised']:<15.6f} "
              f"{r['improvement_percent']:<12.1f}% {recommendation:<10}")

    print("-" * 70)

    # ========== 可视化建议 ==========
    print("\n使用建议:")
    if effective_threshold:
        print(f"  • 噪声 σ < {effective_threshold:.2f}: 不建议使用均值滤波（损失细节）")
        print(f"  • 噪声 σ ≥ {effective_threshold:.2f}: 建议使用均值滤波（有效降噪）")
    else:
        print("  • 均值滤波在所有测试噪声水平下均无效，建议使用其他方法")

    # 找出最佳改善场景
    best_improvement = max(results, key=lambda x: x['improvement_percent'])
    print(
        f"\n  • 最佳使用场景: σ={best_improvement['noise_std']:.2f}, 改善{best_improvement['improvement_percent']:.1f}%")

    print("\n✅ 验证完成")
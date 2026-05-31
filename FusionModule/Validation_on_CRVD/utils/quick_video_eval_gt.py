import torch
import torch.nn.functional as F
import numpy as np


def quick_evaluate(sRGB_noisy, sRGB_gt, sRGB_denoise):
    """快速评估降噪效果（自动处理numpy/torch）。

    输入：
        sRGB_noisy: (B, T, H, W, 3) numpy或tensor
        sRGB_gt: (B, T, H, W, 3) numpy或tensor
        sRGB_denoise: (B, T, H, W, 3) numpy或tensor
    """
    # 自动转换为 tensor
    if isinstance(sRGB_noisy, np.ndarray):
        sRGB_noisy = torch.from_numpy(sRGB_noisy).float()
    if isinstance(sRGB_gt, np.ndarray):
        sRGB_gt = torch.from_numpy(sRGB_gt).float()
    if isinstance(sRGB_denoise, np.ndarray):
        sRGB_denoise = torch.from_numpy(sRGB_denoise).float()

    B, T, H, W, C = sRGB_noisy.shape

    # 重塑为 (BT, C, H, W)
    noisy = sRGB_noisy.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    gt = sRGB_gt.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    denoise = sRGB_denoise.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

    print("=" * 70)
    print("降噪效果快速评估")
    print("=" * 70)

    # ========== 1. PSNR计算 ==========
    def calc_psnr(img1, img2):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    psnr_noisy = calc_psnr(noisy, gt).item()
    psnr_denoise = calc_psnr(denoise, gt).item()
    psnr_gain = psnr_denoise - psnr_noisy

    print(f"\n📊 PSNR (越高越好):")
    print(f"  噪声图像:   {psnr_noisy:.2f} dB")
    print(f"  降噪图像:   {psnr_denoise:.2f} dB")
    print(f"  增益:       {psnr_gain:+.2f} dB {'✓改善' if psnr_gain > 0 else '✗变差'}")

    # ========== 2. MAE/MSE计算 ==========
    mae_noisy = (noisy - gt).abs().mean().item()
    mae_denoise = (denoise - gt).abs().mean().item()
    mae_improve = (mae_noisy - mae_denoise) / mae_noisy * 100

    mse_noisy = F.mse_loss(noisy, gt).item()
    mse_denoise = F.mse_loss(denoise, gt).item()
    mse_improve = (mse_noisy - mse_denoise) / mse_noisy * 100

    print(f"\n📉 误差指标 (越低越好):")
    print(f"  MAE - 噪声: {mae_noisy:.6f}, 降噪: {mae_denoise:.6f}, 改善: {mae_improve:+.1f}%")
    print(f"  MSE - 噪声: {mse_noisy:.6f}, 降噪: {mse_denoise:.6f}, 改善: {mse_improve:+.1f}%")

    # ========== 3. 逐帧分析（前5帧） ==========
    print(f"\n📹 逐帧PSNR分析 (前5帧):")
    print(f"{'帧号':<6} {'噪声PSNR':<12} {'降噪PSNR':<12} {'增益':<10} {'状态'}")
    print("-" * 50)

    for i in range(min(5, B * T)):
        psnr_n = calc_psnr(noisy[i:i + 1], gt[i:i + 1]).item()
        psnr_d = calc_psnr(denoise[i:i + 1], gt[i:i + 1]).item()
        gain = psnr_d - psnr_n
        status = "✓" if gain > 0 else "✗"
        print(f"{i:<6} {psnr_n:<12.2f} {psnr_d:<12.2f} {gain:+<10.2f} {status}")

    # ========== 4. 统计摘要 ==========
    psnrs_noisy = torch.stack([calc_psnr(noisy[i:i + 1], gt[i:i + 1]) for i in range(B * T)])
    psnrs_denoise = torch.stack([calc_psnr(denoise[i:i + 1], gt[i:i + 1]) for i in range(B * T)])
    gains = psnrs_denoise - psnrs_noisy

    improve_count = (gains > 0).sum().item()
    improve_rate = improve_count / (B * T) * 100

    print(f"\n📈 统计摘要:")
    print(f"  总帧数:     {B * T}")
    print(f"  改善帧数:   {improve_count} ({improve_rate:.1f}%)")
    print(f"  平均增益:   {gains.mean().item():+.2f} dB")
    print(f"  最大增益:   {gains.max().item():+.2f} dB")
    print(f"  最小增益:   {gains.min().item():+.2f} dB")

    # ========== 5. 快速结论 ==========
    print(f"\n🎯 快速结论:")
    if psnr_gain > 2.0 and improve_rate > 80:
        print(f"  ✅ 降噪效果优秀！PSNR提升{psnr_gain:.1f}dB，{improve_rate:.0f}%帧改善")
    elif psnr_gain > 0.5 and improve_rate > 60:
        print(f"  ✓ 降噪效果良好，PSNR提升{psnr_gain:.1f}dB，{improve_rate:.0f}%帧改善")
    elif psnr_gain > 0:
        print(f"  △ 降噪效果一般，PSNR仅提升{psnr_gain:.1f}dB")
    else:
        print(f"  ✗ 降噪效果不佳，PSNR下降{abs(psnr_gain):.1f}dB，需要调整参数")

    print("=" * 70)

    return {
        'psnr_noisy': psnr_noisy,
        'psnr_denoise': psnr_denoise,
        'psnr_gain': psnr_gain,
        'mae_improve': mae_improve,
        'improve_rate': improve_rate,
    }

import os
import subprocess
import shutil
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 导入你现有的模块
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2
from Denosie import split_bayer_gbrg, merge_bayer_gbrg, calculate_psnr


def run_vbm3d_operator(exe_path, bin_dir, input_dir, sigma, num_frames):
    """
    底层算子驱动：执行 VBM3D 并精准回收 %03d.tiff 结果
    """
    # 清理旧产物
    for f in bin_dir.glob("deno_*.tiff"): os.remove(f)
    for f in bin_dir.glob("bsic_*.tiff"): os.remove(f)

    input_pattern = str(input_dir / "in_%04d.tif").replace("\\", "/")
    args = [exe_path, "-i", input_pattern, "-f", "1", "-l", str(num_frames), "-sigma", f"{sigma:.2f}", "-add", "false"]

    # 运行算子
    subprocess.run(args, cwd=str(bin_dir), capture_output=True, check=True)

    # 回收 float32 数据
    results = []
    for i in range(1, num_frames + 1):
        prod_path = bin_dir / f"deno_{i:03d}.tiff"
        img = cv2.imread(str(prod_path), cv2.IMREAD_UNCHANGED)  # 读回 float32
        results.append(torch.from_numpy(img))
    return torch.stack(results)


def compare_vbm3d_strategies():
    # --- 1. 配置环境 ---
    bin_dir = Path(r"E:/Reasearch/Code/vbm3d_1/build/bin")
    exe_path = str(bin_dir / "VBM3Ddenoising.exe")
    workspace = Path(r"E:/vbm3d_comparison_workspace")

    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"
    scene_id, iso_val, num_frames = 1, 1600, 7

    if workspace.exists(): shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # --- 2. 加载原始数据 ---
    dataset = CRVDDataset(noisy_root, gt_root, scenes=[scene_id], iso_levels=[iso_val], sequence_mode=True)
    noisy_seq_np, gt_seq_np = dataset[0]  # (7, 1080, 1920)

    # --- 3. 策略 A：分通道处理 (Per-Channel) ---
    print("🚀 正在运行策略 A: 分通道处理...")
    noisy_tensor = torch.from_numpy(noisy_seq_np).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_seq_np).unsqueeze(0)
    noisy_ch_seq = split_bayer_gbrg(noisy_tensor)[0]  # (7, 4, 540, 960)
    gt_ch_seq = split_bayer_gbrg(gt_tensor)[0]

    denoised_ch_results = []
    ch_names = ["G1", "R", "B", "G2"]
    for c in range(4):
        ch_dir = workspace / f"ch_{ch_names[c]}"
        ch_dir.mkdir()
        for f in range(num_frames):
            cv2.imwrite(str(ch_dir / f"in_{f + 1:04d}.tif"), noisy_ch_seq[f, c].numpy().astype(np.uint16))

        # 计算该通道独立 Sigma
        sigma = torch.std(noisy_ch_seq[:, c] - gt_ch_seq[:, c]).item()
        print(f"  - 通道 {ch_names[c]} Sigma: {sigma:.2f}")
        denoised_ch_results.append(run_vbm3d_operator(exe_path, bin_dir, ch_dir, sigma, num_frames))

    # 合并
    merged_tensor = torch.stack(denoised_ch_results, dim=1).unsqueeze(0)
    seq_denoised_per_ch = merge_bayer_gbrg(merged_tensor)[0].numpy()

    # --- 4. 策略 B：全图处理 (Whole-Frame) ---
    print("🚀 正在运行策略 B: 全图不分通道处理...")
    whole_dir = workspace / "whole_frame"
    whole_dir.mkdir()
    for f in range(num_frames):
        cv2.imwrite(str(whole_dir / f"in_{f + 1:04d}.tif"), noisy_seq_np[f].astype(np.uint16))

    # 计算全图全局 Sigma
    sigma_whole = np.std(noisy_seq_np - gt_seq_np)
    print(f"  - 全图全局 Sigma: {sigma_whole:.2f}")
    seq_denoised_whole = run_vbm3d_operator(exe_path, bin_dir, whole_dir, sigma_whole, num_frames).numpy()

    # --- 5. 评价与可视化 ---
    isp = OpenCV_ISP2(show_preview=False)
    psnr_noisy = [calculate_psnr(noisy_seq_np[f], gt_seq_np[f]) for f in range(num_frames)]
    psnr_per_ch = [calculate_psnr(seq_denoised_per_ch[f], gt_seq_np[f]) for f in range(num_frames)]
    psnr_whole = [calculate_psnr(seq_denoised_whole[f], gt_seq_np[f]) for f in range(num_frames)]

    print("\n📈 最终 PSNR 对比 (平均值):")
    print(f"  Noisy:      {np.mean(psnr_noisy):.2f} dB")
    print(f"  Per-Channel: {np.mean(psnr_per_ch):.2f} dB")
    print(f"  Whole-Frame: {np.mean(psnr_whole):.2f} dB")

    # 转 RGB 预览
    with torch.no_grad():
        rgb_per_ch = isp(torch.from_numpy(seq_denoised_per_ch).unsqueeze(0))[0][3]  # 取中间帧
        rgb_whole = isp(torch.from_numpy(seq_denoised_whole).unsqueeze(0))[0][3]
        rgb_gt = isp(torch.from_numpy(gt_seq_np).unsqueeze(0))[0][3]

    # 绘图
    plt.figure(figsize=(16, 10))
    # PSNR 曲线
    plt.subplot(2, 1, 1)
    frames = range(1, 8)
    plt.plot(frames, psnr_noisy, 'o--', label='Noisy', color='gray')
    plt.plot(frames, psnr_per_ch, 's-', label='Strategy A: Per-Channel (Split)', color='blue')
    plt.plot(frames, psnr_whole, '^-', label='Strategy B: Whole-Frame (No Split)', color='green')
    plt.title("VBM3D 3DNR Strategy Comparison: PSNR Improvement")
    plt.ylabel("PSNR (dB)");
    plt.grid(True, alpha=0.3);
    plt.legend()

    # 视觉对比
    plt.subplot(1, 3, 1);
    plt.imshow(rgb_per_ch);
    plt.title(f"Per-Channel\\nPSNR: {psnr_per_ch[3]:.2f}");
    plt.axis('off')
    plt.subplot(1, 3, 2);
    plt.imshow(rgb_whole);
    plt.title(f"Whole-Frame\\nPSNR: {psnr_whole[3]:.2f}");
    plt.axis('off')
    plt.subplot(1, 3, 3);
    plt.imshow(rgb_gt);
    plt.title("Ground Truth");
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_vbm3d_strategies()
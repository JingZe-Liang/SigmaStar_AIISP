import os
import subprocess
import shutil
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 导入自定义模块
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2
from Denosie import split_bayer_gbrg, merge_bayer_gbrg, calculate_psnr


def run_vbm3d_and_save_to_desktop():
    # --- 1. 路径配置 ---
    bin_dir = Path(r"E:/Reasearch/Code/vbm3d_1/build/bin")
    exe_path = str(bin_dir / "VBM3Ddenoising.exe")
    workspace = Path(r"E:/vbm3d_final_task")

    # 目标保存路径：桌面
    save_root = Path(r"C:/Users/Jaime/Desktop/VBM3D_3DNR_Results")

    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"
    scene_id, iso_val, num_frames = 1, 1600, 7

    # 初始化文件夹
    if workspace.exists(): shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    if not save_root.exists(): save_root.mkdir(parents=True)

    # --- 2. 加载 CRVD 数据 ---
    dataset = CRVDDataset(noisy_root, gt_root, scenes=[scene_id], iso_levels=[iso_val], sequence_mode=True)
    noisy_seq_np, gt_seq_np = dataset[0]

    noisy_tensor = torch.from_numpy(noisy_seq_np).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_seq_np).unsqueeze(0)

    # 拆分通道
    noisy_ch_seq = split_bayer_gbrg(noisy_tensor)[0]
    gt_ch_seq = split_bayer_gbrg(gt_tensor)[0]

    ch_names = ["G1", "R", "B", "G2"]
    denoised_ch_results = []

    # --- 3. 算子处理流水线 (分通道策略) ---
    print(f"🚀 启动 VBM3D 高精度处理 (Scene {scene_id}, ISO {iso_val})...")

    for c in range(4):
        ch_dir = workspace / ch_names[c]
        ch_dir.mkdir()

        # 计算该通道独立 Sigma
        sigma = torch.std(noisy_ch_seq[:, c] - gt_ch_seq[:, c]).item()

        # 导出 16-bit 临时文件
        for f in range(num_frames):
            cv2.imwrite(str(ch_dir / f"in_{f + 1:04d}.tif"), noisy_ch_seq[f, c].numpy().astype(np.uint16))

        # 调用算子
        input_pattern = str(ch_dir / "in_%04d.tif").replace("\\", "/")
        # 清理旧产物
        for f in bin_dir.glob("deno_*.tiff"): os.remove(f)

        subprocess.run([exe_path, "-i", input_pattern, "-f", "1", "-l", str(num_frames),
                        "-sigma", f"{sigma:.2f}", "-add", "false"],
                       cwd=str(bin_dir), capture_output=True, check=True)

        # 读回 float32 结果
        ch_frames = []
        for f in range(1, num_frames + 1):
            img = cv2.imread(str(bin_dir / f"deno_{f:03d}.tiff"), cv2.IMREAD_UNCHANGED)
            ch_frames.append(torch.from_numpy(img))
        denoised_ch_results.append(torch.stack(ch_frames))

    # --- 4. 合并并保存结果 ---
    merged_tensor = torch.stack(denoised_ch_results, dim=1).unsqueeze(0)
    # 得到 7 帧最终的 RAW 去噪图像 (1080, 1920)
    denoised_raw_seq = merge_bayer_gbrg(merged_tensor)[0].numpy()

    print(f"\n💾 正在将 7 帧无损结果保存至: {save_root}")
    psnr_report = []

    for f in range(num_frames):
        # 计算 PSNR 增益
        p_noisy = calculate_psnr(noisy_seq_np[f], gt_seq_np[f])
        p_deno = calculate_psnr(denoised_raw_seq[f], gt_seq_np[f])
        psnr_report.append((p_noisy, p_deno))

        # 保存为 16-bit TIFF 以便你在桌面查看 (剪切到 12-bit 范围)
        output_frame = np.clip(denoised_raw_seq[f], 0, 4095).astype(np.uint16)
        save_path = save_root / f"VBM3D_Deno_Scene{scene_id}_ISO{iso_val}_Frame{f + 1}.tif"
        cv2.imwrite(str(save_path), output_frame)

        print(f"帧 {f + 1}: Noisy {p_noisy:.2f} -> Denoised {p_deno:.2f} (+{p_deno - p_noisy:+.2f} dB)")

    # --- 5. 可视化总结 ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 8), [p[0] for p in psnr_report], 'o--', label='Noisy')
    plt.plot(range(1, 8), [p[1] for p in psnr_report], 's-', label='VBM3D (Per-Channel)', color='red')
    plt.title("VBM3D 3DNR PSNR Comparison")
    plt.xlabel("Frame Index");
    plt.ylabel("PSNR (dB)");
    plt.grid(True);
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_vbm3d_and_save_to_desktop()
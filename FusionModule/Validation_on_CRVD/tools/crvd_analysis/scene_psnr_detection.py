import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

# 导入你的 ISP 工具
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2


def calculate_psnr_np(img1, img2, max_val=4095.0):
    """针对 12-bit RAW 数据计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def verify_all_denoise_datasets():
    # 1. 路径配置
    base_root = Path(r"E:\CRVD_dataset")
    dataset_paths = {
        "GT": base_root / "indoor_raw_gt",
        "2DNR_Bilateral": base_root / "2DNR_bilateral",
        "2DNR_BM3D": base_root / "2DNR_bm3d",
        "3DNR_VBM3D": base_root / "3DNR_vbm3d"
    }

    # 2. 检查路径有效性
    for name, path in dataset_paths.items():
        if not path.exists():
            print(f"❌ 找不到路径: {name} -> {path}")
            return

    # 3. 获取序列
    all_sequences = []
    for scene_dir in dataset_paths["GT"].glob("scene*"):
        for iso_dir in scene_dir.glob("ISO*"):
            all_sequences.append(Path(scene_dir.name) / iso_dir.name)

    if not all_sequences: return

    # 4. 抽样
    num_samples = min(10, len(all_sequences))
    sampled_rel_paths = random.sample(all_sequences, num_samples)
    isp = OpenCV_ISP2(show_preview=False)

    for i, rel_path in enumerate(sampled_rel_paths):
        print(f"\n[{i + 1}/{num_samples}] 正在计算 PSNR 并对比: {rel_path}")

        display_rows = {}
        psnr_table = {}  # 存储 PSNR 数值: {ds_name: [f1, f2...f7]}
        raw_gt_frames = []  # 用于计算基准

        # --- A. 先读取 GT 帧 ---
        gt_dir = dataset_paths["GT"] / rel_path
        for f_idx in range(1, 8):
            gt_file = list(gt_dir.glob(f"frame{f_idx}*slightly_denoised.tiff"))[0]
            raw_gt_frames.append(cv2.imread(str(gt_file), cv2.IMREAD_UNCHANGED).astype(np.float32))

        # --- B. 读取并计算其他数据集 ---
        for ds_name, ds_root in dataset_paths.items():
            current_dir = ds_root / rel_path
            raw_frames = []
            psnrs = []

            pattern = "*_temporal_denoised.tiff" if "3DNR" in ds_name else "*_slightly_denoised.tiff"

            for f_idx in range(1, 8):
                file_matches = list(current_dir.glob(f"frame{f_idx}{pattern}"))
                if not file_matches: break

                img = cv2.imread(str(file_matches[0]), cv2.IMREAD_UNCHANGED).astype(np.float32)
                raw_frames.append(img)

                # 计算与 GT 的 PSNR
                if ds_name == "GT":
                    psnrs.append(100.0)  # GT 自身为 100
                else:
                    psnrs.append(calculate_psnr_np(img, raw_gt_frames[f_idx - 1]))

            if len(raw_frames) == 7:
                raw_tensor = torch.from_numpy(np.stack(raw_frames)).unsqueeze(0)
                with torch.no_grad():
                    display_rows[ds_name] = isp(raw_tensor)[0]
                psnr_table[ds_name] = psnrs
            else:
                display_rows[ds_name] = None

        # --- C. 绘图与高亮逻辑 ---
        fig, axes = plt.subplots(4, 7, figsize=(24, 14))
        row_names = list(dataset_paths.keys())

        for col_idx in range(7):
            # 提取当前列（当前帧）除 GT 外的所有 PSNR
            col_scores = {name: psnr_table[name][col_idx] for name in row_names if
                          name != "GT" and psnr_table[name] is not None}

            if col_scores:
                # 排序获取第一和第二
                sorted_scores = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
                first_place = sorted_scores[0][0]
                second_place = sorted_scores[1][0] if len(sorted_scores) > 1 else None
            else:
                first_place = second_place = None

            for row_idx, ds_name in enumerate(row_names):
                ax = axes[row_idx, col_idx]
                frames_rgb = display_rows[ds_name]

                if frames_rgb is not None:
                    ax.imshow(frames_rgb[col_idx])

                    # 显示 PSNR 文字
                    score = psnr_table[ds_name][col_idx]
                    score_str = "Inf" if score > 90 else f"{score:.2f}dB"

                    # --- 高亮逻辑 ---
                    text_color = 'white'
                    bbox_props = dict(boxstyle="round,pad=0.3", alpha=0.7)

                    if ds_name == first_place:
                        bbox_props['facecolor'] = 'red'  # 第一名红色
                    elif ds_name == second_place:
                        bbox_props['facecolor'] = 'orange'  # 第二名黄色 (用orange效果比纯yellow清晰)
                    else:
                        bbox_props['facecolor'] = 'black'

                    ax.text(5, 5, score_str, color=text_color, fontsize=11, fontweight='bold',
                            va='top', ha='left', bbox=bbox_props, transform=ax.transData)

                    if col_idx == 0:
                        ax.set_ylabel(ds_name, fontsize=12, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, "MISSING", ha='center')

                ax.set_xticks([]);
                ax.set_yticks([])

        plt.suptitle(f"PSNR Comparison & Alignment: {rel_path}\n(Red: Best | Yellow: 2nd Best)", fontsize=18)
        plt.tight_layout()
        plt.show(block=False)
        input(f">>> [{i + 1}/{num_samples}] 按 Enter 查看下一个场景...")
        plt.close(fig)


if __name__ == "__main__":
    verify_all_denoise_datasets()
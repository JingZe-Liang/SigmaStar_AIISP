import re
from pathlib import Path

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")


H5_DIR = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\DarkData\smoke")
OUTPUT_DIR = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\Reports\DarkSmoke")
SAMPLES_PER_FILE = 3
TARGET_KEYS = ("2dnr", "3dnr", "noisy")


def natural_sort_key(text: str) -> list[int | str]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", text)]


def sample_frame_indices(total_frames: int, sample_count: int, exclude_first: bool = True) -> np.ndarray:
    """安全地抽样帧索引，避免总帧数过少时直接报错。"""
    start_index = 1 if exclude_first and total_frames > 1 else 0
    candidates = np.arange(start_index, total_frames)
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    real_sample_count = min(sample_count, len(candidates))
    return np.random.choice(candidates, real_sample_count, replace=False)


def analyze_single_frame(h5_file: h5py.File, target_key: str, frame_idx: int, file_name: str) -> tuple[float, float, bool]:
    """分析单帧并生成可视化 QA 报告。"""
    clean = h5_file["clean"][frame_idx].astype(np.float64)
    raw_test = h5_file[target_key][frame_idx]
    test = raw_test[1].astype(np.float64) if target_key == "noisy" else raw_test.astype(np.float64)

    shift, _ = cv2.phaseCorrelate(clean, test)
    pc_dx, pc_dy = shift

    residual = test - clean
    grad_x = cv2.Sobel(clean, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clean, cv2.CV_64F, 0, 1, ksize=3)
    grad_dx = -np.mean(residual * grad_x) / (np.mean(grad_x**2) + 1e-9)
    grad_dy = -np.mean(residual * grad_y) / (np.mean(grad_y**2) + 1e-9)

    is_aligned = abs(pc_dx) < 0.1 and abs(pc_dy) < 0.1

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(clean, cmap="gray")
    plt.title(f"GT (idx:{frame_idx})")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(test, cmap="gray")
    plt.title(f"Test ({target_key})")
    plt.axis("off")

    residual_vis = np.clip((residual + 100) / 200 * 255, 0, 255).astype(np.uint8)
    plt.subplot(2, 3, 3)
    plt.imshow(residual_vis, cmap="RdBu")
    plt.title("Residual")
    plt.axis("off")

    bins = np.linspace(0, 4096, 64)
    digitized = np.digitize(clean, bins)
    bin_centers: list[float] = []
    variances: list[float] = []
    for bin_idx in range(1, len(bins)):
        mask = digitized == bin_idx
        if int(np.sum(mask)) > 300:
            variances.append(float(np.var(residual[mask])))
            bin_centers.append(float((bins[bin_idx - 1] + bins[bin_idx]) / 2))

    plt.subplot(2, 3, 4)
    plt.scatter(bin_centers, variances, s=10, color="blue")
    plt.plot(bin_centers, variances, "r--")
    plt.title("Noise Profile")

    plt.subplot(2, 3, 6)
    plt.axis("off")
    status_text = "PASS" if is_aligned else "FAIL"
    info = (
        f"File: {file_name}\nTarget: {target_key} | Frame: {frame_idx}\n\n"
        f"Phase Shift: {pc_dx:.4f}, {pc_dy:.4f}\n"
        f"Grad Shift: {grad_dx:.4f}, {grad_dy:.4f}\n"
        f"Status: [{status_text}]"
    )
    plt.text(0, 0.5, info, fontsize=12, weight="bold", color="green" if is_aligned else "red")

    report_path = OUTPUT_DIR / f"{file_name}_{target_key}_idx{frame_idx}.png"
    plt.tight_layout()
    plt.savefig(report_path, dpi=120)
    plt.close()

    return float(pc_dx), float(pc_dy), is_aligned


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h5_files = sorted([file for file in H5_DIR.iterdir() if file.suffix.lower() == ".h5"], key=lambda path: natural_sort_key(path.name))

    total_tasks = len(h5_files) * SAMPLES_PER_FILE * len(TARGET_KEYS)
    print(f"--- 启动批量验收：共检测 {len(h5_files)} 个分片，预计 {total_tasks} 个任务 ---")
    progress_bar = tqdm(total=total_tasks, desc="Overall Batch QA", unit="task")

    all_results: list[tuple[str, int, str, float, float, bool]] = []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5_file:
            total_frames = int(len(h5_file["clean"]))
            sample_indices = sample_frame_indices(total_frames, SAMPLES_PER_FILE, exclude_first=True)
            if len(sample_indices) == 0:
                print(f"[跳过] {h5_path.name} 可用帧数不足，未执行抽样。")
                continue

            for frame_idx in sample_indices:
                for target_key in TARGET_KEYS:
                    if target_key in h5_file:
                        progress_bar.set_description(f"Processing {h5_path.name}")
                        dx, dy, status = analyze_single_frame(h5_file, target_key, int(frame_idx), h5_path.name)
                        all_results.append((h5_path.name, int(frame_idx), target_key, dx, dy, status))
                    progress_bar.update(1)

    progress_bar.close()

    print(f"\n{'文件名':<20} | {'帧号':<5} | {'项目':<6} | {'位移 (dx, dy)':<22} | {'状态'}")
    print("-" * 80)
    fail_count = 0
    for file_name, frame_idx, target_key, dx, dy, status in all_results:
        if not status:
            fail_count += 1
        print(f"{file_name:<20} | {frame_idx:<5} | {target_key:<6} | ({dx:>7.4f}, {dy:>7.4f}) | {'[PASS]' if status else '[FAIL]'}")

    print("\n--- 批量验收完成 ---")
    print(f"抽检总项：{len(all_results)} | 失败项：{fail_count}")
    print(f"详细报告图片目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

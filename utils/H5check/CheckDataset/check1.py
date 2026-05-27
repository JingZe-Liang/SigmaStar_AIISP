"""对单个 H5 文件中的单帧执行对齐 QA。"""

import os

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# =================【输入 / 输出配置】=================
# H5_FILE：待检查的单个 H5 文件路径。
# OUTPUT_DIR：QA 报告图片输出目录。
# IDX：要分析的帧索引。
H5_FILE = r"D:\DeepLearning\VideoDenoising\CheckDataset\h5_dataset\shard_0.h5"
OUTPUT_DIR = r"D:\DeepLearning\VideoDenoising\CheckDataset\Report"
IDX = 10
# ====================================================


def run_rigorous_qa_for_target(h5_file: h5py.File, target_key: str, idx: int) -> tuple[str, float, float, float, float, bool]:
    """针对单个目标执行对齐分析。"""
    clean = h5_file["clean"][idx].astype(np.float64)

    raw_test = h5_file[target_key][idx]
    if target_key == "noisy":
        # noisy 采用双帧形式，这里固定取当前帧 t 对应的通道 1。
        test = raw_test[1].astype(np.float64)
    else:
        test = raw_test.astype(np.float64)

    shift, confidence = cv2.phaseCorrelate(clean, test)
    pc_dx, pc_dy = shift

    residual = test - clean
    grad_x = cv2.Sobel(clean, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clean, cv2.CV_64F, 0, 1, ksize=3)

    corr_x = np.mean(residual * grad_x)
    corr_y = np.mean(residual * grad_y)
    var_grad_x = np.mean(grad_x**2) + 1e-9
    var_grad_y = np.mean(grad_y**2) + 1e-9

    gs_dx = -corr_x / var_grad_x
    gs_dy = -corr_y / var_grad_y
    is_aligned = abs(pc_dx) < 0.1 and abs(pc_dy) < 0.1

    bins = np.linspace(0, 4096, 128)
    digitized = np.digitize(clean, bins)
    bin_centers: list[float] = []
    variances: list[float] = []
    for bin_idx in range(1, len(bins)):
        mask = digitized == bin_idx
        if int(np.sum(mask)) > 500:
            variances.append(float(np.var(residual[mask])))
            bin_centers.append(float((bins[bin_idx - 1] + bins[bin_idx]) / 2))

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(clean, cmap="gray")
    plt.title("Clean (GT)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(test, cmap="gray")
    plt.title(f"Test ({target_key})")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    residual_vis = np.clip((residual + 100) / 200 * 255, 0, 255).astype(np.uint8)
    plt.imshow(residual_vis, cmap="RdBu")
    plt.title("Amplified Residual")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.scatter(bin_centers, variances, s=15, alpha=0.6, color="blue")
    plt.plot(bin_centers, variances, color="red", linestyle="--")
    plt.xlabel("Intensity")
    plt.ylabel("Variance")
    plt.title("Noise Profile")

    height, width = clean.shape
    plt.subplot(2, 3, 5)
    zoom = residual[height // 2 - 100 : height // 2 + 100, width // 2 - 100 : width // 2 + 100]
    plt.imshow(zoom, cmap="seismic")
    plt.title("Edge Area Zoom")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.axis("off")
    result_text = (
        f"QA Analysis: {target_key}\n\n"
        f"Phase Shift: {pc_dx:.4f}, {pc_dy:.4f}\n"
        f"Grad Shift: {gs_dx:.4f}, {gs_dy:.4f}\n"
        f"Confidence: {confidence:.3f}\n\n"
        f"Status: {'PASS' if is_aligned else 'FAIL'}\n"
        f"Conclusion: {'Aligned' if is_aligned else 'Shifted'}"
    )
    plt.text(0.1, 0.5, result_text, fontsize=14, weight="bold", color="green" if is_aligned else "red")

    report_name = f"QA_{target_key}_idx{idx}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, report_name), dpi=150)
    plt.close()

    return target_key, float(pc_dx), float(pc_dy), float(gs_dx), float(gs_dy), is_aligned


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_keys = ["2dnr", "3dnr", "noisy"]
    summary_results: list[tuple[str, float, float, float, float, bool]] = []

    # 【输出标记】终端会输出本次单帧 QA 的摘要结果。
    print(f"开始执行单帧 QA：{os.path.basename(H5_FILE)} (帧索引: {IDX})")
    print("-" * 70)

    with h5py.File(H5_FILE, "r") as h5_file:
        for key in target_keys:
            if key not in h5_file:
                print(f"跳过 {key}（数据集中不存在）")
                continue

            summary_results.append(run_rigorous_qa_for_target(h5_file, key, IDX))

    print(f"\n{'数据项':<10} | {'相位位移 (dx, dy)':<24} | {'梯度位移 (dx, dy)':<24} | {'状态'}")
    print("-" * 85)
    for name, pdx, pdy, gdx, gdy, status in summary_results:
        print(f"{name:<10} | ({pdx:>7.4f}, {pdy:>7.4f}) | ({gdx:>7.4f}, {gdy:>7.4f}) | {'[PASS]' if status else '[FAIL]'}")

    print(f"\n验收完成，报告图片已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import re
from pathlib import Path

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")


H5_DIR = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\CompanyData\scene1")
OUTPUT_DIR = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\Reports\test6")
SAMPLES_PER_FILE = 3
CURR_FRAME_IDX = 0
TARGET_KEYS = ("2dnr", "3dnr")


def natural_sort_key(text: str) -> list[int | str]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", text)]


def sample_frame_indices(total_frames: int, sample_count: int) -> np.ndarray:
    candidates = np.arange(total_frames)
    real_sample_count = min(sample_count, len(candidates))
    if real_sample_count == 0:
        return np.array([], dtype=np.int64)
    return np.random.choice(candidates, real_sample_count, replace=False)


def analyze_no_gt_frame(h5_file: h5py.File, target_key: str, frame_idx: int, file_name: str) -> tuple[float, float, bool]:
    """在没有 GT 的情况下，使用 noisy 当前帧作为相对对齐锚点。"""
    anchor = h5_file["noisy"][frame_idx, CURR_FRAME_IDX].astype(np.float64)

    if target_key == "noisy":
        test = h5_file["noisy"][frame_idx, 1].astype(np.float64)
    else:
        test = h5_file[target_key][frame_idx].astype(np.float64)

    shift, _ = cv2.phaseCorrelate(anchor, test)
    pc_dx, pc_dy = shift
    residual = test - anchor

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(anchor, cmap="gray")
    plt.title("Anchor (Noisy_t)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(test, cmap="gray")
    plt.title(f"Test ({target_key})")
    plt.axis("off")

    residual_vis = np.clip((residual + 100) / 200 * 255, 0, 255).astype(np.uint8)
    plt.subplot(2, 3, 3)
    plt.imshow(residual_vis, cmap="RdBu")
    plt.title("Difference Map")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.hist(residual.flatten(), bins=100, range=(-100, 100), color="gray", alpha=0.7)
    plt.title("Residual Distribution")
    plt.yscale("log")

    plt.subplot(2, 3, 6)
    plt.axis("off")
    is_aligned = abs(pc_dx) < 0.2 and abs(pc_dy) < 0.2
    info = (
        f"File: {file_name}\nTarget: {target_key} | idx: {frame_idx}\n\n"
        f"Rel. Shift to Noisy_t:\n"
        f"dx={pc_dx:.4f}, dy={pc_dy:.4f}\n\n"
        f"Status: [{'PASS' if is_aligned else 'FAIL'}]"
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

    all_results: list[tuple[str, int, str, float, float, bool]] = []
    progress_bar = tqdm(total=len(h5_files) * SAMPLES_PER_FILE * len(TARGET_KEYS))

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5_file:
            total_frames = int(len(h5_file["noisy"]))
            sample_indices = sample_frame_indices(total_frames, SAMPLES_PER_FILE)
            if len(sample_indices) == 0:
                print(f"[跳过] {h5_path.name} 可用帧数不足。")
                continue

            for frame_idx in sample_indices:
                for target_key in TARGET_KEYS:
                    if target_key in h5_file:
                        progress_bar.set_description(f"Scanning {h5_path.name}")
                        dx, dy, status = analyze_no_gt_frame(h5_file, target_key, int(frame_idx), h5_path.name)
                        all_results.append((h5_path.name, int(frame_idx), target_key, dx, dy, status))
                    progress_bar.update(1)

    progress_bar.close()

    print(f"\n{'文件名':<20} | {'帧号':<5} | {'项目':<6} | {'相对位移 (dx, dy)':<24} | {'状态'}")
    print("-" * 85)
    for file_name, frame_idx, target_key, dx, dy, status in all_results:
        print(f"{file_name:<20} | {frame_idx:<5} | {target_key:<6} | ({dx:>7.4f}, {dy:>7.4f}) | {'[PASS]' if status else '[FAIL]'}")


if __name__ == "__main__":
    main()
